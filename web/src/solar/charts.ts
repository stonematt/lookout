/**
 * The Solar tab's charts, drawn as SVG at their real pixel width (DESIGN.md,
 * "Chart grammar"): the 24-hour trace over its usual range, the daily bars,
 * a day's 15-minute bars, and the three heatmaps (year by month, month by
 * day, day by 15 minutes).
 *
 * Shared grammar, as on the Rain tab: hairline gridlines on y only, 12px
 * tabular ticks, labels haloed in the ground, the 135° hatch for what is not
 * in the record, and one claret mark for the headline's subject. Every chart
 * has a live readout below it (hover, tap, or focus and use the arrow keys),
 * so nothing relies on a title. The day charts are also pickers: a click,
 * or Enter on the focused chart, picks the day under the readout.
 */
import {
  NARROW,
  type Readout,
  canvas,
  chartLabel,
  hatchMissingDays,
  monthTicks,
  runs,
  scrub,
  yAxis,
  zeroTicks,
} from "../charts/svg.js";
import { svgEl } from "../dom.js";
import { MONTHS, stationFormat, addDays, daysBetween, formatTime, dayText, hourText, kwhPerM2, localDay, wattsPerM2 } from "../format.js";
import { dayStatus, gapSentence } from "../gaps.js";
import type { GapReason, SolarTab, SolarTrace } from "../snapshot.js";
import type { SolarDay } from "./text.js";
import {
  FIRST_SLOT,
  LAST_SLOT,
  lastDay,
  monthText,
  peakSlot,
  slotText,
  wh,
} from "./text.js";

/** Called with the day a chart picks. */
export type Pick = (day: string) => void;

/** How many shades the heatmaps use, from faint to full sun. */
export const STEPS = 5;

/** The daily bars cover this many days, ending today. */
export const BAR_DAYS = 90;

/**
 * The shade for a value on a scale up to `max`: 0 for none, else 1 to 5.
 *
 * @param value - The value.
 * @param max - The scale's top.
 * @returns The step.
 */
export function shade(value: number, max: number): number {
  if (value <= 0 || max <= 0) return 0;
  return Math.max(1, Math.min(STEPS, Math.ceil((value / max) * STEPS)));
}

/** The class for a shaded cell, with its text class for the value on it. */
function cellClass(step: number): string {
  return step ? `solar-cell solar-step-${step}` : "solar-cell solar-none";
}

/**
 * Make a chart pick the day under its readout on click or Enter.
 *
 * @param target - The chart host.
 * @param current - The day under the readout now, or null.
 * @param pick - Receives it.
 */
function pickable(target: HTMLElement, current: () => string | null, pick: Pick): void {
  target.onclick = () => {
    const day = current();
    if (day) pick(day);
  };
  target.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const day = current();
    if (!day) return;
    event.preventDefault();
    pick(day);
  });
}

/** Local hour and minute of an epoch ms at the station. */
const clockOf = () =>
  stationFormat("en-GB", { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });

/** The station-local [hour, minute] of an epoch ms. */
function localClock(ms: number): [number, number] {
  const [h, m] = clockOf().format(new Date(ms)).split(":").map(Number);
  return [h ?? 0, m ?? 0];
}

/**
 * The 24-hour view: the bold solar trace over the soft usual range, with
 * the dashed usual median, ending at the live reading.
 *
 * @param host - The figure; its contents are replaced.
 * @param trace - The snapshot's 24-hour view.
 * @param archiveNewestMs - The archive's newest row: after it only the live
 *   reading is known.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawTrace(
  host: HTMLElement,
  trace: SolarTrace,
  archiveNewestMs: number | null,
  width: number,
  readout: Readout,
): void {
  const narrow = width < NARROW;
  const height = narrow ? 200 : 240;
  const [left, right, top, bottom] = [44, width - (narrow ? 12 : 16), 26, height - 24];
  const values = trace.solar_wm2;
  const n = values.length;
  const high = trace.usual_high_wm2 ?? [];
  const top_value = Math.max(100, ...values.filter((v): v is number => v !== null), ...high.filter((v): v is number => v !== null));
  const ticks = zeroTicks(top_value, 4);
  const yMax = ticks[ticks.length - 1] ?? 1;
  const y = (v: number) => bottom - ((bottom - top) * v) / yMax;
  const step = (right - left) / (n - 1);
  const x = (i: number) => left + i * step;
  const stepMs = trace.step_minutes * 60_000;
  const at = (i: number) => trace.start_ms + i * stepMs;

  const { svg, hatch } = canvas(width, height, "Solar radiation over the last 24 hours against its usual range", "solar-svg");
  yAxis(svg, ticks, y, left, right, "solar", { text: "W/m²", place: "above" });

  // The usual range at the back, then its median.
  const low = trace.usual_low_wm2;
  const median = trace.usual_median_wm2;
  if (low && trace.usual_high_wm2) {
    const hi = trace.usual_high_wm2;
    for (const [a, b] of runs(n, (i) => low[i] !== null && hi[i] !== null)) {
      const upper = [];
      const lower = [];
      for (let i = a; i <= b; i++) {
        upper.push(`${x(i).toFixed(1)},${y(hi[i]!).toFixed(1)}`);
        lower.unshift(`${x(i).toFixed(1)},${y(low[i]!).toFixed(1)}`);
      }
      svg.append(svgEl("polygon", { points: [...upper, ...lower].join(" "), class: "solar-band" }));
    }
  }
  if (median) {
    for (const [a, b] of runs(n, (i) => median[i] !== null)) {
      const points = [];
      for (let i = a; i <= b; i++) points.push(`${x(i).toFixed(1)},${y(median[i]!).toFixed(1)}`);
      svg.append(svgEl("polyline", { points: points.join(" "), class: "solar-median" }));
    }
  }

  // The archived span ends at the archive's newest slot; after it, only live.
  const archived =
    archiveNewestMs === null ? n - 1 : Math.min(n - 1, Math.floor((archiveNewestMs - trace.start_ms) / stepMs));
  if (archived < n - 1) {
    const from = Math.max(0, archived);
    svg.append(svgEl("rect", { x: x(from), y: top, width: x(n - 1) - x(from), height: bottom - top, fill: hatch, rx: 2 }));
    const words = x(n - 1) - x(from) > 150 ? "only the live reading is known" : "live only";
    svg.append(chartLabel(x(n - 1), top - 8, words, "solar-label", "end"));
    const last = values[n - 1];
    const before = archived >= 0 ? values[archived] : null;
    if (last !== null && last !== undefined && before !== null && before !== undefined) {
      svg.append(svgEl("line", { x1: x(archived), y1: y(before), x2: x(n - 1), y2: y(last), class: "solar-live-join" }));
    }
  }
  // Readings missing inside the archived span are hatched, never bridged.
  for (const [a, b] of runs(archived + 1, (i) => values[i] === null)) {
    if (b - a < 1) continue;
    svg.append(svgEl("rect", { x: x(a), y: top, width: Math.max(2, x(b) - x(a)), height: bottom - top, fill: hatch, rx: 2 }));
  }
  for (const [a, b] of runs(archived + 1, (i) => values[i] !== null)) {
    const points = [];
    for (let i = a; i <= b; i++) points.push(`${x(i).toFixed(1)},${y(values[i]!).toFixed(1)}`);
    if (points.length === 1) points.push(points[0]!);
    svg.append(svgEl("polyline", { points: points.join(" "), class: "solar-now-line" }));
  }

  // Ticks every six hours, by the station's clock.
  for (let i = 0; i < n; i++) {
    const [h, m] = localClock(at(i));
    if (m !== 0 || h % 6 !== 0) continue;
    svg.append(svgEl("line", { x1: x(i), x2: x(i), y1: bottom, y2: bottom + 4, class: "solar-axis" }));
    svg.append(svgEl("text", { x: x(i), y: bottom + 16, class: "solar-tick", "text-anchor": "middle" }, hourText(h)));
  }

  const now = values[n - 1];
  if (now !== null && now !== undefined) {
    svg.append(svgEl("circle", { cx: x(n - 1), cy: y(now), r: 5, class: "solar-now-dot" }));
    svg.append(chartLabel(x(n - 1) - 8, Math.max(top + 4, y(now) - 10), `now ${wattsPerM2(now)}`, "solar-now-label", "end"));
  }

  host.replaceChildren(svg);
  scrub(
    host,
    n,
    (px) => Math.round((px - left) / step),
    (i) => {
      const clock = formatTime(at(i));
      const when = dayText(localDay(at(i)));
      const value = values[i];
      const usual =
        median && low && trace.usual_high_wm2 && median[i] !== null && median[i] !== undefined
          ? `; usually ${wattsPerM2(median[i]!)} (${Math.round(low[i] ?? 0)} to ${Math.round(trace.usual_high_wm2[i] ?? 0)})`
          : "";
      if (i === n - 1 && value !== null && value !== undefined) return readout(`Now, ${clock}: ${wattsPerM2(value)}, the live reading${usual}.`);
      if (i > archived) return readout(`${clock}: not archived yet; only the live reading is known${usual}.`);
      if (value === null || value === undefined) return readout(`${clock}: no reading in the record.`);
      readout(`${when}, ${clock}: ${wattsPerM2(value)}${usual}.`);
    },
  );
}

/** A day's sentence: its energy, or why it is not counted. */
function daySentence(day: string, energy: number | null, reason: GapReason | null, isToday: boolean): string {
  if (reason) return gapSentence(dayStatus(day, reason));
  const when = `${dayText(day, true)}${isToday ? " (today, so far)" : ""}`;
  return energy === null ? `${when}: not in the record.` : `${when}: ${kwhPerM2(energy)}. Select to see its 15 minutes.`;
}

/**
 * The daily bars: one bar per day of the last 90, the last 7 marked as the
 * headline's subject and the selected day ringed.
 *
 * @param host - The figure; its contents are replaced.
 * @param tab - The Solar tab's data.
 * @param selected - The selected day.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 * @param pick - Receives a picked day.
 */
export function drawDailyBars(
  host: HTMLElement,
  tab: SolarTab,
  selected: string,
  width: number,
  readout: Readout,
  pick: Pick,
): void {
  const all = tab.days.energy_kwh_m2;
  const offset = Math.max(0, all.length - BAR_DAYS);
  const energy = all.slice(offset);
  const reasons = tab.days.reason.slice(offset);
  const first = addDays(tab.days.first_day, offset);
  const count = energy.length;
  const height = 168;
  const [left, right, top, bottom] = [44, width - 6, 24, height - 22];
  const ticks = zeroTicks(Math.max(1, ...energy.filter((v): v is number => v !== null)));
  const yMax = ticks[ticks.length - 1] ?? 1;
  const y = (v: number) => bottom - ((bottom - top) * v) / yMax;
  const step = (right - left) / Math.max(1, count);
  const x = (i: number) => left + i * step;
  const pad = step > 4 ? 1 : 0;

  const { svg, hatch } = canvas(width, height, `Solar energy each day, ${dayText(first)} to today`, "solar-svg");
  yAxis(svg, ticks, y, left, right, "solar", { text: "kWh/m²", place: "above" });
  hatchMissingDays(svg, reasons, hatch, x, step, top, bottom, "solar-label");
  energy.forEach((value, i) => {
    if (value === null || value <= 0) return;
    const recent = i >= count - 7;
    const isToday = i === count - 1;
    svg.append(
      svgEl("rect", {
        x: x(i) + pad / 2,
        y: y(value),
        width: Math.max(1, step - pad),
        height: bottom - y(value),
        rx: step > 6 ? 2 : 0,
        class: isToday ? "solar-bar solar-bar-now" : recent ? "solar-bar solar-bar-week" : "solar-bar",
      }),
    );
  });
  if (count >= 7) {
    const a = x(count - 7);
    const b = x(count - 1) + step;
    svg.append(svgEl("line", { x1: a, x2: b, y1: top - 6, y2: top - 6, class: "solar-week-rule" }));
    svg.append(chartLabel(b, top - 11, "last 7 days", "solar-label", "end"));
  }
  const s = daysBetween(first, selected);
  if (s >= 0 && s < count) {
    svg.append(svgEl("rect", { x: x(s) - 1, y: top - 2, width: step + 2, height: bottom - top + 4, rx: 3, class: "solar-selected" }));
  }
  monthTicks(svg, first, count, x, bottom + 4, "solar");
  host.replaceChildren(svg);
  let under: string | null = null;
  scrub(
    host,
    count,
    (px) => Math.floor((px - left) / step),
    (i) => {
      under = addDays(first, i);
      readout(daySentence(under, energy[i] ?? null, reasons[i] ?? null, i === count - 1));
    },
  );
  pickable(host, () => under, pick);
}

/**
 * One day's 15-minute energy, 4 am to 10 pm, its brightest slot marked.
 *
 * @param host - The figure; its contents are replaced.
 * @param day - The day as read, with its slots.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawDaySlots(host: HTMLElement, day: SolarDay, width: number, readout: Readout): void {
  const slots = day.slots ?? [];
  const count = LAST_SLOT - FIRST_SLOT + 1;
  const height = 180;
  const [left, right, top, bottom] = [44, width - 6, 24, height - 22];
  const shown = slots.slice(FIRST_SLOT, LAST_SLOT + 1);
  const ticks = zeroTicks(Math.max(20, ...shown.filter((v): v is number => v !== null)));
  const yMax = ticks[ticks.length - 1] ?? 1;
  const y = (v: number) => bottom - ((bottom - top) * v) / yMax;
  const step = (right - left) / count;
  const x = (i: number) => left + (i - FIRST_SLOT) * step;
  const peak = peakSlot(slots);
  // Today, the slots after the last catalogued one are still to come.
  let lastKnown = -1;
  slots.forEach((v, i) => {
    if (v !== null) lastKnown = i;
  });

  const { svg, hatch } = canvas(width, height, `Solar energy each 15 minutes on ${dayText(day.day, true)}`, "solar-svg");
  yAxis(svg, ticks, y, left, right, "solar", { text: "Wh/m²", place: "above" });
  for (const [a, b] of runs(LAST_SLOT + 1, (i) => i >= FIRST_SLOT && slots[i] === null && (!day.isToday || i < lastKnown))) {
    svg.append(svgEl("rect", { x: x(a), y: top, width: x(b) + step - x(a), height: bottom - top, fill: hatch, rx: 2 }));
  }
  for (let i = FIRST_SLOT; i <= LAST_SLOT; i++) {
    const v = slots[i];
    if (v === null || v === undefined || v <= 0) continue;
    const isPeak = peak !== null && peak[0] === i;
    svg.append(
      svgEl("rect", {
        x: x(i) + 0.5,
        y: y(v),
        width: Math.max(1, step - 1),
        height: bottom - y(v),
        rx: 2,
        class: isPeak ? "solar-bar solar-bar-now" : "solar-bar",
      }),
    );
  }
  if (peak) {
    const cx = x(peak[0]) + step / 2;
    const anchor = cx > (left + right) / 2 ? "end" : "start";
    svg.append(chartLabel(cx + (anchor === "end" ? 4 : -4), Math.max(top - 6, y(peak[1]) - 8), `${slotText(peak[0])}, ${wh(peak[1])}`, "solar-now-label", anchor));
  }
  for (const h of [6, 9, 12, 15, 18, 21]) {
    const at = x(h * 4);
    svg.append(svgEl("line", { x1: at, x2: at, y1: bottom, y2: bottom + 4, class: "solar-axis" }));
    svg.append(svgEl("text", { x: at, y: bottom + 16, class: "solar-tick", "text-anchor": "middle" }, hourText(h)));
  }
  host.replaceChildren(svg);
  scrub(
    host,
    count,
    (px) => Math.floor((px - left) / step),
    (i) => {
      const slot = i + FIRST_SLOT;
      const v = slots[slot];
      const when = `${slotText(slot)} to ${slotText(slot + 1)}`;
      if (v === null || v === undefined) {
        return readout(day.isToday && slot > lastKnown ? `${when}: still to come.` : `${when}: no reading in the record.`);
      }
      readout(`${when}: ${wh(v)}.`);
    },
  );
}

/**
 * The year-by-month heatmap: years as rows (newest first), months across,
 * each cell shaded by its energy with the value in it where there is room.
 *
 * @param host - The figure; its contents are replaced.
 * @param tab - The Solar tab's data.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawYearMonth(host: HTMLElement, tab: SolarTab, width: number, readout: Readout): void {
  const byKey = new Map(tab.months.map((m) => [m.key, m.status]));
  const energy = new Map(tab.months.map((m) => [m.key, m.energy_kwh_m2]));
  const years = [...new Set(tab.months.map((m) => Number(m.key.slice(0, 4))))].sort((a, b) => b - a);
  const left = 44;
  const gap = 3;
  const cell = Math.min(52, Math.floor((width - left - 4) / 12) - gap);
  const rowH = Math.min(44, Math.max(24, cell));
  const top = 18;
  const height = top + years.length * (rowH + gap) + 4;
  const max = Math.max(1, ...tab.months.map((m) => m.energy_kwh_m2 ?? 0));
  const months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];

  const { svg, hatch } = canvas(width, height, "Solar energy each month, one row per year", "solar-svg");
  MONTHS.forEach((name, m) => {
    const cx = left + m * (cell + gap) + cell / 2;
    svg.append(svgEl("text", { x: cx, y: 12, class: "solar-tick", "text-anchor": "middle" }, cell >= 30 ? name : months[m]!));
  });
  years.forEach((year, r) => {
    const ry = top + r * (rowH + gap);
    svg.append(svgEl("text", { x: left - 6, y: ry + rowH / 2 + 4, class: "solar-tick", "text-anchor": "end" }, String(year)));
    for (let m = 0; m < 12; m++) {
      const key = `${year}-${String(m + 1).padStart(2, "0")}`;
      const status = byKey.get(key);
      if (!status) continue;
      const cx = left + m * (cell + gap);
      const value = energy.get(key) ?? null;
      const attrs = { x: cx, y: ry, width: cell, height: rowH, rx: 4 };
      if (status.reason === "before_archive") {
        svg.append(svgEl("rect", { ...attrs, class: "solar-before" }));
      } else if (value === null) {
        svg.append(svgEl("rect", { ...attrs, fill: hatch, class: status.reason === "in_progress" ? "solar-current" : "" }));
      } else {
        const level = shade(value, max);
        svg.append(svgEl("rect", { ...attrs, class: cellClass(level) }));
        if (cell >= 30) {
          const text = `${status.status === "prorated" ? "≈" : ""}${Math.round(value)}`;
          svg.append(svgEl("text", { x: cx + cell / 2, y: ry + rowH / 2 + 4, class: level >= 4 ? "solar-cell-text solar-cell-text-strong" : "solar-cell-text", "text-anchor": "middle" }, text));
        }
      }
    }
  });
  host.replaceChildren(svg);
  scrub(
    host,
    12,
    (px) => Math.floor((px - left) / (cell + gap)),
    (m, r) => {
      const year = years[r];
      if (year === undefined) return;
      const key = `${year}-${String(m + 1).padStart(2, "0")}`;
      const status = byKey.get(key);
      const value = energy.get(key) ?? null;
      if (!status) return readout(`${monthText(key)}: not in the record.`);
      if (value === null) return readout(gapSentence(status));
      readout(`${monthText(key)}: ${status.status === "prorated" ? "≈ " : ""}${kwhPerM2(value)}.`);
    },
    years.length,
    (py) => Math.floor((py - top) / (rowH + gap)),
  );
}

/** Every month from the record's first through today's, newest first. */
function monthKeys(first: string, last: string): string[] {
  const keys = [];
  let y = Number(last.slice(0, 4));
  let m = Number(last.slice(5, 7));
  const stop = first.slice(0, 7);
  for (;;) {
    const key = `${y}-${String(m).padStart(2, "0")}`;
    keys.push(key);
    if (key <= stop) break;
    m -= 1;
    if (m === 0) {
      m = 12;
      y -= 1;
    }
  }
  return keys;
}

/** Days in a "YYYY-MM" month. */
function monthLength(key: string): number {
  return new Date(Date.UTC(Number(key.slice(0, 4)), Number(key.slice(5, 7)), 0)).getUTCDate();
}

/**
 * The month-by-day heatmap: months as rows (newest first), days 1 to 31
 * across, each day shaded by its energy.
 *
 * @param host - The figure; its contents are replaced.
 * @param tab - The Solar tab's data.
 * @param selected - The selected day, ringed.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 * @param pick - Receives a picked day.
 */
export function drawMonthDay(host: HTMLElement, tab: SolarTab, selected: string, width: number, readout: Readout, pick: Pick): void {
  const today = lastDay(tab);
  const keys = monthKeys(tab.days.first_day, today);
  const left = 64;
  const right = width - 4;
  const step = (right - left) / 31;
  const rowH = width < NARROW ? 12 : 14;
  const top = 18;
  const height = top + keys.length * rowH + 4;
  const max = Math.max(1, ...tab.days.energy_kwh_m2.map((v) => v ?? 0));
  const dayAt = (r: number, d: number): string | null => {
    const key = keys[r];
    if (key === undefined || d < 1 || d > monthLength(key)) return null;
    const day = `${key}-${String(d).padStart(2, "0")}`;
    return day < tab.days.first_day || day > today ? null : day;
  };

  const { svg, hatch } = canvas(width, height, "Solar energy each day, one row per month", "solar-svg");
  for (const d of [1, 8, 15, 22, 29]) {
    svg.append(svgEl("text", { x: left + (d - 1) * step + step / 2, y: 12, class: "solar-tick", "text-anchor": "middle" }, String(d)));
  }
  keys.forEach((key, r) => {
    const ry = top + r * rowH;
    svg.append(svgEl("text", { x: left - 6, y: ry + rowH - 2, class: "solar-tick", "text-anchor": "end" }, monthText(key)));
    for (let d = 1; d <= 31; d++) {
      const day = dayAt(r, d);
      if (day === null) continue;
      const i = daysBetween(tab.days.first_day, day);
      const value = tab.days.energy_kwh_m2[i] ?? null;
      const attrs = { x: left + (d - 1) * step, y: ry, width: Math.max(1, step - 1), height: rowH - 1, rx: 2 };
      if (value === null) svg.append(svgEl("rect", { ...attrs, fill: hatch }));
      else svg.append(svgEl("rect", { ...attrs, class: cellClass(shade(value, max)) }));
      if (day === selected) svg.append(svgEl("rect", { ...attrs, class: "solar-selected" }));
    }
  });
  host.replaceChildren(svg);
  let under: string | null = null;
  scrub(
    host,
    31,
    (px) => Math.floor((px - left) / step),
    (d, r) => {
      under = dayAt(r, d + 1);
      if (under === null) return readout(`${monthText(keys[r] ?? "")}, day ${d + 1}: not in the record.`);
      const i = daysBetween(tab.days.first_day, under);
      readout(daySentence(under, tab.days.energy_kwh_m2[i] ?? null, tab.days.reason[i] ?? null, under === today));
    },
    keys.length,
    (py) => Math.floor((py - top) / rowH),
  );
  pickable(host, () => under, pick);
}

/**
 * The day-by-15-minutes heatmap: one column per day of the last 90, one
 * row per 15 minutes from 4 am (top) to 10 pm, each cell shaded by its
 * energy.
 *
 * @param host - The figure; its contents are replaced.
 * @param tab - The Solar tab's data.
 * @param selected - The selected day, ringed.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 * @param pick - Receives a picked day.
 */
export function drawDayGrid(host: HTMLElement, tab: SolarTab, selected: string, width: number, readout: Readout, pick: Pick): void {
  const recent = tab.recent;
  const count = recent.energy_wh_m2.length;
  const rows = LAST_SLOT - FIRST_SLOT + 1;
  const row = width < NARROW ? 3 : 4;
  const [left, right, top] = [56, width - 6, 4];
  const bottom = top + rows * row;
  const height = bottom + 22;
  const step = (right - left) / count;
  const x = (i: number) => left + i * step;
  const max = Math.max(1, ...recent.energy_wh_m2.flat().map((v) => v ?? 0));
  const offset = daysBetween(tab.days.first_day, recent.first_day);
  const reasons = recent.energy_wh_m2.map((_, i) => tab.days.reason[offset + i] ?? (offset + i < 0 ? "before_archive" : null));

  const { svg, hatch } = canvas(width, height, `Solar energy each 15 minutes, ${dayText(recent.first_day)} to today`, "solar-svg");
  recent.energy_wh_m2.forEach((slots, i) => {
    if (reasons[i] !== null) return;
    for (let s = FIRST_SLOT; s <= LAST_SLOT; s++) {
      const v = slots[s];
      if (v === null || v === undefined) continue;
      svg.append(svgEl("rect", { x: x(i), y: top + (s - FIRST_SLOT) * row, width: Math.max(1, step - (step > 4 ? 1 : 0)), height: row, class: cellClass(shade(v, max)) }));
    }
  });
  hatchMissingDays(svg, reasons, hatch, x, step, top, bottom, "solar-label");
  const s = daysBetween(recent.first_day, selected);
  if (s >= 0 && s < count) {
    svg.append(svgEl("rect", { x: x(s) - 1, y: top - 1, width: step + 2, height: bottom - top + 2, rx: 2, class: "solar-selected" }));
  }
  for (const h of [6, 12, 18]) {
    svg.append(svgEl("text", { x: left - 6, y: top + (h * 4 - FIRST_SLOT) * row + 4, class: "solar-tick", "text-anchor": "end" }, hourText(h)));
  }
  monthTicks(svg, recent.first_day, count, x, bottom + 4, "solar");
  host.replaceChildren(svg);
  let under: string | null = null;
  scrub(
    host,
    count,
    (px) => Math.floor((px - left) / step),
    (i, r) => {
      const day = addDays(recent.first_day, i);
      under = reasons[i] === "before_archive" ? null : day;
      const reason = reasons[i];
      if (reason) return readout(gapSentence(dayStatus(day, reason)));
      const slot = r + FIRST_SLOT;
      const v = recent.energy_wh_m2[i]?.[slot];
      const when = `${dayText(day, true)}, ${slotText(slot)}`;
      readout(v === null || v === undefined ? `${when}: no reading.` : `${when}: ${wh(v)}. Select to see the whole day.`);
    },
    rows,
    (py) => Math.floor((py - top) / row),
  );
  pickable(host, () => under, pick);
}

/**
 * The key for a heatmap's shades: what each step stands for.
 *
 * @param max - The scale's top.
 * @param unit - "kWh/m²" or "Wh/m²".
 * @returns One [class, words] pair per step.
 */
export function shadeKey(max: number, unit: string): [string, string][] {
  const out: [string, string][] = [];
  for (let s = 1; s <= STEPS; s++) {
    const upTo = (max * s) / STEPS;
    const text = unit === "Wh/m²" ? String(Math.round(upTo)) : kwhPerM2(upTo, { unit: false });
    out.push([`solar-step-${s}`, `up to ${text} ${unit}`]);
  }
  return out;
}
