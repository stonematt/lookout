/**
 * The temperature context (#81), three sections. They left the Overview
 * with the accordion (#472) and stay here as the temperature zoom's content
 * (#474); the Temperature mosaic tile (`mosaic-tiles.ts`) crops the first.
 *
 * 1. temperature across periods: one range bar per window (today, yesterday,
 *    7, 30, 90, 365 days, Full history), a whisker from the lowest to the
 *    highest reading, a bar over the middle half, and one "now" rule across
 *    every row; under it, the one next-unlock line (#470);
 * 2. the ten-day high-low list, each day's low-to-high bar over the usual
 *    low-to-high for its date, in miniature;
 * 3. "now against normal" strips for temperature and for rain so far today.
 *
 * Every number is the snapshot's (`lookout/snapshot/temperature.py`): this
 * module words and draws, it does not compute a comparison. History grows
 * (#470): a window or day the record does not cover yet (`shown` false) is
 * not drawn at all. A shown window or day the gap status marks invalid is
 * hatched with its reason, without "n/a" (a gap stays visible), and has no value
 * under it; a prorated one shows `≈` (`gapValue`), and the tile's source note
 * carries a hatched chip counting what is not compared.
 */
import "./temperature.css";

import { degrees, inches } from "../format.js";
import { LABEL_CHAR_PX, NARROW, aboveDot, niceTicks, pillLabel, scale } from "../charts/svg.js";
import { make, spoken, svgEl } from "../dom.js";
import { gapSentence, gapValue, isComparable, plainGapLabel, svgHatch } from "../gaps.js";
import type {
  DashboardSnapshot,
  GapStatus,
  NormalStrip,
  TemperatureDay,
  TemperaturePeriod,
} from "../snapshot.js";
import { gapReasons, hatchedCount, sourceNote, tileFinding, tileFolds } from "../tiles.js";
import type { SectionOptions } from "../tiles.js";
import { WINDOW_NAMES, isShown, nextUnlockLine } from "./history.js";
import type { OverviewSection } from "./index.js";

const SHORT_DATE = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  month: "short",
  day: "numeric",
});
const DAY_LABEL = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  weekday: "short",
  month: "short",
  day: "numeric",
});

// ---------- words ----------

/** An ISO day as a UTC date, so the zone cannot move it. */
const dayDate = (day: string) => new Date(`${day}T12:00:00Z`);

/** "2025, 2024 and 2023": years, newest first. */
export function yearsText(years: readonly number[]): string {
  if (years.length < 2) return years.length === 1 ? String(years[0]) : "";
  return `${years.slice(0, -1).join(", ")} and ${years.at(-1)}`;
}

/**
 * The window chart's rows: each window the record covers (every window, in
 * a snapshot written before Release A), then Full history when the snapshot
 * has it (#470).
 */
function periodRows(snapshot: DashboardSnapshot): TemperaturePeriod[] {
  const ctx = snapshot.temperature_context;
  const rows = ctx.periods.filter((p) => isShown(p.status));
  const full = ctx.full_history;
  if (full && isShown(full.status)) rows.push(full);
  return rows;
}

/** The "today" window, when the record covers it. */
export function todayPeriod(snapshot: DashboardSnapshot): TemperaturePeriod | null {
  const today = snapshot.temperature_context.periods.find((p) => p.status.key === "today");
  return today && isShown(today.status) ? today : null;
}

/**
 * Today in the ten-day list, which holds the usual low and high for this
 * date: the Usual row of the window range chart, in the zoom and on the
 * Temperature tile.
 */
export function todayInList(snapshot: DashboardSnapshot): TemperatureDay | null {
  const today = todayPeriod(snapshot);
  if (!today) return null;
  return (
    snapshot.temperature_context.ten_days.find((d) => d.day === today.status.first_day) ?? null
  );
}

/** The Usual row's low and high and its date, when the snapshot holds them. */
function usualRow(snapshot: DashboardSnapshot): { low: number; high: number; day: string } | null {
  const day = todayInList(snapshot);
  if (!day || day.usual_low === null || day.usual_high === null) return null;
  return { low: day.usual_low, high: day.usual_high, day: day.day };
}

/** The Usual row's span line: "Nov 10, earlier years". */
function usualSpan(day: string): string {
  return `${SHORT_DATE.format(dayDate(day))}, earlier years`;
}

/** The accessible sentence for the Usual row; null when there is no usual. */
export function usualSentence(snapshot: DashboardSnapshot): string | null {
  const usual = usualRow(snapshot);
  if (!usual) return null;
  return `Usual (${usualSpan(usual.day)}): low ${degrees(usual.low)} to high ${degrees(usual.high)}.`;
}

/** The ten-day list's days the record covers, oldest first. */
export function shownDays(snapshot: DashboardSnapshot): TemperatureDay[] {
  return snapshot.temperature_context.ten_days.filter((d) => isShown(d.status));
}

/**
 * The window's days, for a row label: "Jan 1 to Jan 6", or with the start
 * year when the window spans a new year, "Dec 31, 2025 to Jan 6", so a
 * 365-day window never reads backwards ("Jan 7 to Jan 6").
 */
function spanText(status: GapStatus): string {
  const first = SHORT_DATE.format(dayDate(status.first_day));
  if (status.first_day === status.last_day) return first;
  const spansYear = status.first_day.slice(0, 4) !== status.last_day.slice(0, 4);
  const start = spansYear ? `${first}, ${status.first_day.slice(0, 4)}` : first;
  return `${start} to ${SHORT_DATE.format(dayDate(status.last_day))}`;
}

/** How much of a window's readings `now` sits in, for the finding. */
function periodFinding(snapshot: DashboardSnapshot): string {
  const ctx = snapshot.temperature_context;
  const now = ctx.now;
  if (now === null) return "No current temperature to place in context";
  const p = ctx.periods.find((x) => x.status.key === ctx.finding);
  if (p && p.position !== null) {
    const name =
      p.status.key === "yesterday" ? "yesterday" : `the last ${WINDOW_NAMES[p.status.key]}`;
    return `${degrees(now)} now is ${p.position} the middle half of readings over ${name}`;
  }
  return `${degrees(now)} now, with no window complete enough to compare`;
}

// ---------- chart plumbing ----------

/** Draw into `host` now and again whenever its width changes. */
function mountChart(host: HTMLElement, draw: (width: number) => SVGSVGElement | null): void {
  const paint = () => {
    const width = Math.floor(host.clientWidth) || 640;
    const chart = draw(width);
    host.replaceChildren();
    if (chart) host.append(chart);
  };
  paint();
  if (typeof ResizeObserver !== "undefined") {
    let seen = host.clientWidth;
    new ResizeObserver(() => {
      if (Math.abs(host.clientWidth - seen) < 2) return;
      seen = host.clientWidth;
      paint();
    }).observe(host);
  }
}

function root(width: number, height: number, label: string): SVGSVGElement {
  const el = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "tc-svg" });
  el.setAttribute("role", "img");
  el.setAttribute("aria-label", label);
  return el;
}

/** Ticks and their faint grid lines across `rows` of height `h`. */
function axis(
  g: SVGGElement,
  ticks: number[],
  x: (v: number) => number,
  h: number,
  label: (v: number) => string,
): void {
  for (const v of ticks) {
    g.append(svgEl("line", { class: "tc-grid", x1: x(v), x2: x(v), y1: 0, y2: h }));
    g.append(
      svgEl("text", { class: "tc-tick", x: x(v), y: h + 17, "text-anchor": "middle" }, label(v)),
    );
  }
}

/** The hatch over a row whose window cannot be compared, with its reason on a pill. */
function hatchRow(
  g: SVGGElement,
  hatch: string,
  status: GapStatus,
  x0: number,
  x1: number,
  cy: number,
  half: number,
): void {
  g.append(
    svgEl("rect", {
      class: "tc-hatch",
      x: x0,
      y: cy - half,
      width: Math.max(0, x1 - x0),
      height: half * 2,
      rx: 4,
      fill: hatch,
    }),
  );
  // On its pill, drawn after the now rule so the rule never crosses it.
  const reason = plainGapLabel(status);
  if (reason) g.append(pillLabel((x0 + x1) / 2, cy + 4, reason, "tc-reason", "middle"));
}

/** Pick `ticks` that fit `[lo, hi]` and widen the range to hold them. */
function padded(values: number[], pad: number): { lo: number; hi: number; ticks: number[] } {
  let lo = Math.min(...values) - pad;
  let hi = Math.max(...values) + pad;
  const ticks = niceTicks(lo, hi);
  lo = Math.min(lo, ticks[0] ?? lo);
  hi = Math.max(hi, ticks.at(-1) ?? hi);
  return { lo, hi, ticks };
}

// ---------- 1. temperature across periods ----------

/** The accessible text for the periods chart: one sentence per row. */
export function periodSentence(p: TemperaturePeriod): string {
  const name = WINDOW_NAMES[p.status.key] ?? p.status.key;
  if (
    !isComparable(p.status) ||
    p.low === null ||
    p.high === null ||
    p.q1 === null ||
    p.q3 === null
  ) {
    return `${name}: ${gapSentence(p.status)}`;
  }
  const approx = p.status.status === "prorated" ? "about " : "";
  return (
    `${name} (${spanText(p.status)}): ${approx}low ${degrees(p.low)}, middle half ${degrees(p.q1)} to ${degrees(p.q3)}, ` +
    `high ${degrees(p.high)}.${p.status.status === "prorated" ? ` ${gapSentence(p.status)}` : ""}`
  );
}

/**
 * The window range chart: one row per window, the now rule across them.
 * With `withUsual` (the temperature zoom, spec #460 story 35) a Usual row
 * leads, the usual low to high for this date in earlier years, ruled off
 * from the windows under it.
 */
function periodsChart(
  snapshot: DashboardSnapshot,
  width: number,
  withUsual = false,
): SVGSVGElement | null {
  const ctx = snapshot.temperature_context;
  const now = ctx.now;
  const periods = periodRows(snapshot);
  const usual = withUsual ? usualRow(snapshot) : null;
  // The windows sit one row down under a Usual row.
  const lead = usual ? 1 : 0;
  // On a phone each row's label sits above its bar, so the bar gets the
  // whole width; from 520px of chart width the labels are a column.
  const narrow = width < NARROW;
  const rowH = narrow ? 50 : 46;
  // The label column fits the longest span ("Nov 1, 2025 to Oct 31"), so
  // a whisker at the plot's left edge never runs under it.
  const longest = Math.max(
    0,
    ...periods.map((p) => spanText(p.status).length),
    usual ? usualSpan(usual.day).length : 0,
  );
  const left = narrow ? 0 : Math.max(118, Math.ceil(longest * LABEL_CHAR_PX) + 8);
  const f = { l: left, r: narrow ? 0 : 12, t: 34, b: 26 };
  // A row's bar: centred in the row beside its label, under it on a phone.
  const barY = (i: number) => (narrow ? i * rowH + 36 : i * rowH + rowH / 2);
  const iw = width - f.l - f.r;
  const ih = rowH * (periods.length + lead);
  const height = f.t + ih + f.b;

  const numbers = periods.flatMap((p) =>
    isComparable(p.status) && p.low !== null && p.high !== null ? [p.low, p.high] : [],
  );
  if (now !== null) numbers.push(now);
  if (usual) numbers.push(usual.low, usual.high);
  if (numbers.length === 0 && periods.every((p) => isComparable(p.status) === false)) {
    numbers.push(40, 60);
  }
  const { lo, hi, ticks } = padded(numbers, 3);
  // Room at each end for the widest low or high value label: its characters
  // at the chart's label size, the 7px gap to the dot and the dot's radius,
  // never less than the 34px a two-digit value needs. Full history's span
  // is the widest a station's record gives, so a "≈ 102°" or "≈ −12°" must fit.
  const widest = Math.max(
    0,
    ...periods.flatMap((p) =>
      p.low === null || p.high === null || p.q1 === null || p.q3 === null
        ? []
        : [
            (gapValue(p.status, degrees(p.low, 0)) ?? "").length,
            (gapValue(p.status, degrees(p.high, 0)) ?? "").length,
          ],
    ),
  );
  const margin = Math.max(34, Math.ceil(widest * LABEL_CHAR_PX) + 12);
  const x = scale(lo, hi, margin, iw - margin);

  const el = root(width, height, periodFinding(snapshot));
  const hatch = svgHatch(el);
  const g = svgEl("g", { transform: `translate(${f.l},${f.t})` });
  const labels = svgEl("g", { transform: `translate(0,${f.t})` });
  el.append(g, labels);
  axis(g, ticks, x, ih, (v) => degrees(v, 0));
  if (usual) {
    const cy = barY(0);
    const y = narrow ? 18 : cy - 2;
    labels.append(
      svgEl("text", { class: "tc-row", x: narrow ? 0 : 2, y }, "Usual"),
      narrow
        ? svgEl("text", { class: "tc-sub", x: 52, y }, usualSpan(usual.day))
        : svgEl("text", { class: "tc-sub", x: 2, y: cy + 13 }, usualSpan(usual.day)),
      // The rule between the usual and the windows under it.
      svgEl("line", { class: "tc-divider", x1: 0, x2: width, y1: rowH, y2: rowH }),
    );
    g.append(
      svgEl("rect", {
        class: "tc-usual",
        "data-row": "usual",
        x: x(usual.low),
        y: cy - 7,
        width: Math.max(2, x(usual.high) - x(usual.low)),
        height: 14,
        rx: 4,
      }),
    );
  }

  if (now !== null) {
    const nx = x(now);
    g.append(svgEl("line", { class: "tc-now", x1: nx, x2: nx, y1: -8, y2: ih }));
    g.append(svgEl("circle", { class: "tc-now-dot", cx: nx, cy: -8, r: 4 }));
    // The note is 13px bold, a little wider than a 12.5px label.
    const note = `now ${degrees(now)}`;
    const right = nx + 8 + note.length * LABEL_CHAR_PX * 1.15 > iw + f.r;
    // On a phone the note sits above its dot.
    const at = narrow
      ? aboveDot(nx, -8, note, 0, iw, 12 - f.t)
      : { x: right ? nx - 8 : nx + 8, y: -14, anchor: right ? "end" : "start" };
    g.append(svgEl("text", { class: "tc-note", x: at.x, y: at.y, "text-anchor": at.anchor }, note));
  }
  periods.forEach((p, row) => {
    const i = row + lead;
    const cy = barY(i);
    const name = WINDOW_NAMES[p.status.key] ?? p.status.key;
    if (narrow) {
      // The name and its span on one line above the bar. The row name is
      // 13px, a little wider than a 12.5px label.
      const y = i * rowH + 18;
      const after = Math.ceil(name.length * LABEL_CHAR_PX * 1.1) + 8;
      labels.append(
        svgEl("text", { class: "tc-row", x: 0, y }, name),
        svgEl("text", { class: "tc-sub", x: after, y }, spanText(p.status)),
      );
    } else {
      labels.append(
        svgEl("text", { class: "tc-row", x: 2, y: cy - 2 }, name),
        svgEl("text", { class: "tc-sub", x: 2, y: cy + 13 }, spanText(p.status)),
      );
    }
    if (p.low === null || p.high === null || p.q1 === null || p.q3 === null) {
      hatchRow(g, hatch, p.status, 0, iw, cy, narrow ? 10 : 15);
      return;
    }
    const lowText = gapValue(p.status, degrees(p.low, 0))!;
    const highText = gapValue(p.status, degrees(p.high, 0))!;
    g.append(svgEl("line", { class: "tc-whisker", x1: x(p.low), x2: x(p.high), y1: cy, y2: cy }));
    g.append(
      svgEl("rect", {
        class: "tc-iqr",
        x: x(p.q1),
        y: cy - 6,
        width: Math.max(2, x(p.q3) - x(p.q1)),
        height: 12,
        rx: 3,
      }),
    );
    g.append(
      svgEl("circle", { class: "tc-end", cx: x(p.low), cy, r: 2.6 }),
      svgEl("circle", { class: "tc-end", cx: x(p.high), cy, r: 2.6 }),
    );
    g.append(
      svgEl(
        "text",
        { class: "tc-value", x: x(p.low) - 7, y: cy + 4, "text-anchor": "end" },
        lowText,
      ),
    );
    g.append(
      svgEl(
        "text",
        { class: "tc-value", x: x(p.high) + 7, y: cy + 4, "text-anchor": "start" },
        highText,
      ),
    );
  });

  return el;
}

// ---------- 2. the ten-day high-low list ----------

/** The temperature ramp's colour at `t`, blended cool to mild to warm. */
export function rampColour(t: number): string {
  const stops: [number, string][] = [
    [28, "--temp-cool"],
    [56, "--temp-mild"],
    [84, "--temp-warm"],
  ];
  if (t <= stops[0]![0]) return `var(${stops[0]![1]})`;
  if (t >= stops.at(-1)![0]) return `var(${stops.at(-1)![1]})`;
  for (let i = 1; i < stops.length; i += 1) {
    const [t1, c1] = stops[i]!;
    const [t0, c0] = stops[i - 1]!;
    if (t <= t1) {
      const share = Math.round(((t - t0) / (t1 - t0)) * 100);
      return `color-mix(in srgb, var(${c1}) ${share}%, var(${c0}))`;
    }
  }
  return `var(${stops.at(-1)![1]})`;
}

/** The accessible text for one day of the list. */
export function daySentence(d: TemperatureDay): string {
  const when = DAY_LABEL.format(dayDate(d.day));
  if (!isComparable(d.status) || d.low === null || d.high === null) {
    return `${when}: ${gapSentence(d.status)}`;
  }
  const usual =
    d.usual_low !== null && d.usual_high !== null
      ? ` Usual for the date: ${degrees(d.usual_low)} to ${degrees(d.usual_high)}, over ${d.usual_days} days in ${yearsText(d.usual_years)}.`
      : " No earlier year to set it against.";
  return `${when}: low ${degrees(d.low)}, high ${degrees(d.high)}.${usual}`;
}

function daysChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const ctx = snapshot.temperature_context;
  const shown = shownDays(snapshot);
  if (shown.length === 0) return null;
  // Newest first: today leads a phone's scroll.
  const days = [...shown].reverse();
  const narrow = width < NARROW;
  // On a phone each day's label sits above its bar, so the bar gets the
  // whole width, as in the periods chart; from 520px it is a column.
  const rowH = narrow ? 46 : 32;
  // The day column fits the longest name ("Wed, Nov 30" in 13px), so a
  // hatched row starts clear of it, past the name's halo.
  const longest = Math.max(0, ...days.map((d) => DAY_LABEL.format(dayDate(d.day)).length));
  const left = narrow ? 0 : Math.max(84, Math.ceil(longest * LABEL_CHAR_PX * 1.1) + 12);
  const f = { l: left, r: narrow ? 0 : 12, t: 8, b: 26 };
  const iw = width - f.l - f.r;
  const ih = rowH * days.length;

  const numbers = days.flatMap((d) =>
    [d.low, d.high, d.usual_low, d.usual_high].filter((v): v is number => v !== null),
  );
  if (numbers.length === 0) numbers.push(40, 60);
  const { lo, hi, ticks } = padded(numbers, 3);
  const x = scale(lo, hi, 22, iw - 22);

  const el = root(
    width,
    f.t + ih + f.b,
    `The last ten days' lows and highs against the usual for each date in ${yearsText(ctx.ten_day_years)}`,
  );
  const hatch = svgHatch(el);
  const g = svgEl("g", { transform: `translate(${f.l},${f.t})` });
  const labels = svgEl("g", { transform: `translate(0,${f.t})` });
  el.append(g, labels);
  axis(g, ticks, x, ih, (v) => degrees(v, 0));

  days.forEach((d, i) => {
    // The bar's centre: mid-row beside its label, under it on a phone.
    const cy = narrow ? i * rowH + 32 : i * rowH + rowH / 2;
    const today = i === 0;
    labels.append(
      svgEl(
        "text",
        {
          class: `tc-row${today ? " tc-row-today" : ""}`,
          x: narrow ? 0 : 2,
          y: narrow ? i * rowH + 14 : cy + 4,
        },
        DAY_LABEL.format(dayDate(d.day)),
      ),
    );
    if (d.usual_low !== null && d.usual_high !== null) {
      g.append(
        svgEl("rect", {
          class: "tc-usual",
          x: x(d.usual_low),
          y: cy - 10,
          width: Math.max(2, x(d.usual_high) - x(d.usual_low)),
          height: 20,
          rx: 3,
        }),
      );
    }
    if (d.low === null || d.high === null) {
      hatchRow(g, hatch, d.status, 0, iw, cy, 11);
      return;
    }
    const gradient = svgEl("linearGradient", {
      id: `tc-day-${i}-${d.day}`,
      x1: 0,
      x2: 1,
      y1: 0,
      y2: 0,
    });
    const stops = [d.low, d.high];
    for (let t = Math.ceil(d.low / 7) * 7; t < d.high; t += 7) stops.push(t);
    stops.sort((a, b) => a - b);
    for (const t of stops) {
      const stop = svgEl("stop", { offset: d.high === d.low ? 0 : (t - d.low) / (d.high - d.low) });
      stop.setAttribute("style", `stop-color:${rampColour(t)}`);
      gradient.append(stop);
    }
    g.append(gradient);
    g.append(
      svgEl("rect", {
        class: "tc-day",
        x: x(d.low),
        y: cy - 5,
        width: Math.max(4, x(d.high) - x(d.low)),
        height: 10,
        rx: 5,
        fill: `url(#${gradient.id})`,
      }),
    );
    g.append(
      svgEl(
        "text",
        { class: "tc-value", x: x(d.low) - 6, y: cy + 4, "text-anchor": "end" },
        degrees(d.low, 0),
      ),
    );
    g.append(
      svgEl(
        "text",
        { class: "tc-value", x: x(d.high) + 6, y: cy + 4, "text-anchor": "start" },
        degrees(d.high, 0),
      ),
    );
  });
  return el;
}

// ---------- 3. now against normal ----------

/** The claim line above a strip: the percentage and the sample. */
export function stripClaim(strip: NormalStrip): string {
  if (strip.missing === "no_reading") {
    return strip.kind === "rain"
      ? "No rain reading to place against usual."
      : "No temperature reading to place against usual.";
  }
  if (strip.missing === "no_comparison") {
    return "No earlier year is complete enough to set this against.";
  }
  const n = strip.values.length;
  const below = strip.below_pct ?? 0;
  const equal = strip.equal_pct ?? 0;
  const sample = `${n} past days within a week of this date in ${yearsText(strip.years)}`;
  if (strip.kind === "temperature") {
    return below >= 50
      ? `Warmer than ${below}% of ${sample}, at this hour.`
      : `Cooler than ${strip.above_pct ?? 0}% of ${sample}, at this hour.`;
  }
  if (strip.now === 0) {
    return `Dry so far today, as on ${equal}% of ${sample}, by this time of day.`;
  }
  return `More rain so far than ${below}% of ${sample}, by this time of day.`;
}

function nowLabel(strip: NormalStrip): string {
  const v = strip.now ?? 0;
  return strip.kind === "rain" ? `today ${inches(v, { digits: 2 })} so far` : `now ${degrees(v)}`;
}

/** A spread for a dot that is the same on every draw. */
const jitter = (i: number) => ((i * 7) % 11) / 10;

function stripChart(strip: NormalStrip, width: number): SVGSVGElement | null {
  if (strip.missing !== null || strip.now === null) return null;
  const f = { l: 10, r: 14, t: 26, b: 26 };
  const iw = width - f.l - f.r;
  const ih = 104 - f.t - f.b;
  const rain = strip.kind === "rain";
  const all = [...strip.values, strip.now];
  let lo: number;
  let hi: number;
  let ticks: number[];
  if (rain) {
    lo = 0;
    hi = Math.max(...all, 0.1);
    ticks = [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4].filter((t) => t <= hi * 1.001);
  } else {
    ({ lo, hi, ticks } = padded(all, 2));
  }
  // Rain is on a square-root scale, so the many dry and light days spread.
  const map = rain ? (v: number) => Math.sqrt(v) : (v: number) => v;
  const x = scale(map(lo) - (rain ? 0.03 * map(hi) : 0), map(hi), 0, iw);
  const at = (v: number) => x(map(v));

  const el = root(width, 104, `${stripClaim(strip)} ${nowLabel(strip)}.`);
  const g = svgEl("g", { transform: `translate(${f.l},${f.t})` });
  el.append(g);
  g.append(svgEl("line", { class: "tc-base", x1: 0, x2: iw, y1: ih, y2: ih }));
  for (const t of ticks) {
    g.append(svgEl("line", { class: "tc-base", x1: at(t), x2: at(t), y1: ih, y2: ih + 4 }));
    g.append(
      svgEl(
        "text",
        {
          class: "tc-tick",
          x: at(t),
          y: ih + 18,
          "text-anchor": t === ticks[0] ? "start" : "middle",
        },
        rain ? (t === 0 ? "0" : t === ticks.at(-1) ? `${t} in` : String(t)) : degrees(t, 0),
      ),
    );
  }
  strip.values.forEach((v, i) => {
    g.append(
      svgEl("circle", {
        class: `tc-dot tc-dot-${strip.kind}`,
        cx: at(v),
        cy: 6 + jitter(i) * (ih - 14),
        r: 4,
      }),
    );
  });
  const nx = at(strip.now);
  g.append(svgEl("line", { class: "tc-now", x1: nx, x2: nx, y1: -6, y2: ih }));
  const right = nx + 140 > iw + f.r;
  g.append(
    svgEl(
      "text",
      {
        class: "tc-note",
        x: right ? nx - 6 : nx + 6,
        y: -10,
        "text-anchor": right ? "end" : "start",
      },
      nowLabel(strip),
    ),
  );
  return el;
}

// ---------- the tiles ----------

/** Sentences as a list for screen readers alone, one item each. */
function spokenList(sentences: string[]): HTMLElement {
  return spoken(
    sentences.map((s) => make("li", "", s)),
    "ul",
  );
}

function chartHost(part: string): HTMLElement {
  const el = document.createElement("div");
  el.className = "tc-chart";
  el.dataset.part = part;
  return el;
}

/** The headline of the ten-day list, naming the compared years newest first. */
export function daysHeadline(snapshot: DashboardSnapshot): string {
  const years = yearsText(snapshot.temperature_context.ten_day_years);
  return years
    ? `The last ten days, against the usual for their dates in ${years}`
    : "The last ten days, with no earlier year to set them against";
}

/**
 * The window range chart under its finding; the tile form folds its source.
 * In the zoom (`options.zoom`) the chart gains the Usual row.
 */
function renderPeriods(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  options: SectionOptions = {},
): void {
  const periods = periodRows(snapshot);
  host.dataset.block = "periods";
  const chart = chartHost("periods-chart");
  const unlock = nextUnlockLine(snapshot);
  const usual = options.zoom ? usualSentence(snapshot) : null;
  host.append(
    tileFinding(periodFinding(snapshot), options),
    chart,
    ...(unlock ? [unlock] : []),
    spokenList([...(usual ? [usual] : []), ...periods.map(periodSentence)]),
  );
  mountChart(chart, (w) => periodsChart(snapshot, w, options.zoom));
  if (options.zoom) return;
  host.append(
    tileFolds(
      sourceNote(snapshot, {
        what: "five-minute readings, station time",
        method:
          "Each row runs from the lowest to the highest reading in the window; the bar covers the middle half of its readings. " +
          "Five-minute readings, station time, with the live reading as now. A window with a few days missing from the record is marked ≈; one with too many is hatched and not compared.",
        gaps: gapReasons([[hatchedCount(periods.map((p) => p.status)), "window", "windows"]]),
        lag: false,
      }),
    ),
  );
}

/** The ten-day list under its finding; the tile form folds its source. */
function renderTenDays(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  options: SectionOptions = {},
): void {
  const days = shownDays(snapshot);
  host.dataset.block = "ten-days";
  const chart = chartHost("days-chart");
  host.append(
    tileFinding(daysHeadline(snapshot), options),
    chart,
    spokenList([...days].reverse().map(daySentence)),
  );
  mountChart(chart, (w) => daysChart(snapshot, w));
  if (options.zoom) return;
  host.append(
    tileFolds(
      sourceNote(snapshot, {
        what: "daily low and high",
        method:
          "Newest first. Each bar runs from the day's low to its high; the pale band behind it is the usual low to high for that date. " +
          "Usual is the median daily low and median daily high over the days within a week either side of the date, in each earlier year whose record there is complete. Today is the day so far.",
        gaps: gapReasons([[hatchedCount(days.map((d) => d.status)), "day", "days"]]),
        lag: false,
      }),
    ),
  );
}

function stripHeadline(strip: NormalStrip): string {
  const years = yearsText(strip.years);
  return `Now against usual for this date in ${years}`;
}

/** The now-against-this-date strips; the tile form folds its source. */
function renderStrips(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  options: SectionOptions = {},
): void {
  const ctx = snapshot.temperature_context;
  host.dataset.block = "strips";
  host.append(
    tileFinding(
      ctx.strips[0]?.missing === null && ctx.strips[0]
        ? stripHeadline(ctx.strips[0])
        : "Now against usual",
      options,
    ),
  );
  const stripHosts: [NormalStrip, HTMLElement][] = [];
  // Strips with no comparison share one sentence; say it once.
  let saidNoComparison = false;
  for (const strip of ctx.strips) {
    if (strip.missing === "no_comparison") {
      if (saidNoComparison) continue;
      saidNoComparison = true;
    }
    const figure = document.createElement("figure");
    figure.className = "tc-strip";
    figure.dataset.strip = strip.kind;
    const claim = document.createElement("figcaption");
    claim.className = "tc-claim";
    claim.dataset.part = "claim";
    claim.textContent = stripClaim(strip);
    const chart = chartHost(`${strip.kind}-strip`);
    figure.append(claim);
    if (strip.missing === null) {
      figure.append(chart);
      stripHosts.push([strip, chart]);
    }
    host.append(figure);
  }
  const uncompared = ctx.strips.filter((strip) => strip.missing === "no_comparison").length;
  if (!options.zoom) {
    host.append(
      tileFolds(
        sourceNote(snapshot, {
          what: "past days at this hour",
          method:
            "Each dot is a past day within a week of this date; the rule is now. " +
            "Temperature dots are each past day's mean at this hour. Rain dots are each past day's rain by this time of day, on a square-root scale, so today's total so far is set among like totals.",
          gaps: gapReasons([[uncompared, "strip", "strips"]]),
          lag: false,
        }),
      ),
    );
  }
  for (const [strip, chart] of stripHosts) mountChart(chart, (w) => stripChart(strip, w));
}

/**
 * The temperature zoom's window range chart (#474): the Usual row on top,
 * then each window the record covers and Full history; no source fold.
 */
export function renderPeriodsZoom(host: HTMLElement, snapshot: DashboardSnapshot): void {
  renderPeriods(host, snapshot, { zoom: true });
}

/** The temperature zoom's now-against-this-date strips; no source fold. */
export function renderStripsZoom(host: HTMLElement, snapshot: DashboardSnapshot): void {
  renderStrips(host, snapshot, { zoom: true });
}

/** The Last 10 days zoom's high-low list (#477); no source fold. */
export function renderTenDaysZoom(host: HTMLElement, snapshot: DashboardSnapshot): void {
  renderTenDays(host, snapshot, { zoom: true });
}

/** Temperature across periods: one range bar per window, the now rule across them. */
export const temperaturePeriods: OverviewSection = {
  id: "temperature-periods",
  label: "Temperature in context",
  kicker: "Temperature, in context",
  span: 2,
  render: renderPeriods,
};

/** "Now against normal": today's temperature and rain among past days of this date. */
export const againstNormal: OverviewSection = {
  id: "against-normal",
  label: "Now against this date",
  kicker: "Now against this date",
  // Half the four-column row each, with the ten-day list, under the
  // full-width now tile (#468), so the almanac row stays whole.
  span: 2,
  render: renderStrips,
};

/** The ten-day high-low list. */
export const tenDays: OverviewSection = {
  id: "ten-days",
  label: "The last ten days",
  kicker: "The last ten days",
  span: 2,
  render: renderTenDays,
};
