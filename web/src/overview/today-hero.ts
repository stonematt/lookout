/**
 * The usual band hero (#468, spec #460 stories 1–8): the fixed top's chart
 * of today's temperature against the usual band for this date. Not the
 * leading tile (`hero.ts`, "What leads"), which the feature tile replaces.
 *
 * - The band is the middle half of each station-local hour's readings and
 *   the dashed median their median (`temperature_context.usual_band`), drawn
 *   midnight to midnight: measured history, not a forecast. Its basis
 *   (earlier years, the station's recent days, or the median only) is
 *   Python's; the label only words it.
 * - Today's line is the condition ribbon's 15-minute slots that fall in the
 *   snapshot's today (the "today" period's day), ending at the orange now
 *   dot. Where the live reading is ahead of the archive, it is joined by a
 *   dotted line.
 * - Today's rain stands as quarter-hour bars on a track along the foot, one
 *   rain colour, heavier rain a taller bar (`rainStep`), never over the band;
 *   a slot the archive cannot speak for is hatched there.
 *
 * A snapshot written before the band (ADR-0008) draws today's line and rain
 * with no band, median or basis label.
 *
 * Every value is the snapshot's: this module places and words, and decides
 * nothing. Offers `todayChart` (the chart block), `todayChartName` (its
 * accessible name), `bandBasisLabel`, `bandBasisWords` and `todayMissingMs`.
 */
import { degrees, formatStamp, hourText, localDay, stationFormat } from "../format.js";
import { RAIN_STEP_COUNT, rainStep } from "../charts/rain-step.js";
import {
  NARROW,
  aboveDot,
  canvas,
  drawSized,
  niceTicks,
  scale,
  yTickSteps,
} from "../charts/svg.js";
import { make, svgEl } from "../dom.js";
import { yearsText } from "./temperature.js";
import type { DashboardSnapshot, UsualBand, UsualHour } from "../snapshot.js";

/** Room above the plot for the now label. */
const TOP = 22;
/** The y tick labels' gutter. */
const LEFT = 34;
/** Room right of the plot, so the end dot's ring is not clipped. */
const RIGHT_PAD = 6;
/** The plot's height: taller once there is width for it. */
const PLOT_H = { narrow: 128, wide: 150 } as const;
/** The rain track under the plot: its gap from the plot, and its height. */
const RAIN_GAP = 6;
const RAIN_H = 14;
/** The time-of-day labels under the rain track. */
const X_LABELS_H = 22;

// ---------- the snapshot's today ----------

/** The clock-time formatter in the station's zone. */
const clockOf = () =>
  stationFormat("en-US", { hour: "numeric", minute: "numeric", hourCycle: "h23" });

/**
 * A time's place on the station's wall clock, in hours since midnight.
 *
 * @param ms - Epoch milliseconds UTC.
 * @returns 0 up to 24.
 */
function clockHours(ms: number): number {
  const parts = clockOf().formatToParts(ms);
  const value = (type: string) => Number(parts.find((p) => p.type === type)?.value ?? 0);
  return value("hour") + value("minute") / 60;
}

/**
 * The snapshot's today: the "today" period's day, which Python's clock set.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns "YYYY-MM-DD".
 */
function todayOf(snapshot: DashboardSnapshot): string {
  const period = snapshot.temperature_context.periods.find((p) => p.status.key === "today");
  return period?.status.first_day ?? localDay(snapshot.built_at_ms);
}

/** One of today's ribbon slots. */
interface Slot {
  /** Its index in the ribbon. */
  i: number;
  /** When it starts. */
  start: number;
  /** Where its reading sits on the clock, in hours: mid-slot, or the live reading's time. */
  at: number;
  /** The slot's middle on the clock, in hours, where its rain bar stands. */
  mid: number;
  temperature: number | null;
  rain: number | null;
  /** The ribbon's last slot: the live reading. */
  live: boolean;
}

/**
 * Today's ribbon slots, oldest first.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The slots in the snapshot's today; empty without a ribbon.
 */
function todaySlots(snapshot: DashboardSnapshot): Slot[] {
  const ribbon = snapshot.condition_ribbon;
  if (!ribbon) return [];
  const today = todayOf(snapshot);
  const slotMs = ribbon.slot_minutes * 60_000;
  const n = ribbon.temperature_f.length;
  const slots: Slot[] = [];
  for (let i = 0; i < n; i++) {
    const start = ribbon.start_ms + i * slotMs;
    if (localDay(start) !== today) continue;
    const live = i === n - 1;
    // The live slot sits at its reading's time; an archived one mid-slot.
    const at =
      live && snapshot.observed_at_ms !== null ? snapshot.observed_at_ms : start + slotMs / 2;
    const temperature = live
      ? (snapshot.current.temperature_f ?? ribbon.temperature_f[i] ?? null)
      : (ribbon.temperature_f[i] ?? null);
    slots.push({
      i,
      start,
      at: clockHours(at),
      mid: clockHours(start + slotMs / 2),
      temperature,
      rain: ribbon.rain_in[i] ?? null,
      live,
    });
  }
  return slots;
}

/**
 * How much of today the archive should hold but has no rain value for: a gap
 * in the record. The unarchived tail is the archive lag's, not a gap.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns Milliseconds; 0 with no gap, no ribbon or nothing archived.
 */
export function todayMissingMs(snapshot: DashboardSnapshot): number {
  const ribbon = snapshot.condition_ribbon;
  const archived = snapshot.archive_newest_ms;
  if (!ribbon || archived === null) return 0;
  const gaps = todaySlots(snapshot).filter((s) => s.rain === null && s.start <= archived);
  return gaps.length * ribbon.slot_minutes * 60_000;
}

// ---------- words ----------

/**
 * The band's recent days in words.
 *
 * @param days - How many days the band is over.
 * @returns "your last day", "your last 14 days".
 */
const lastDays = (days: number) => (days === 1 ? "your last day" : `your last ${days} days`);

/**
 * The band's label on the chart, from its basis.
 *
 * @param band - The snapshot's usual band.
 * @returns E.g. "usual, this week in 2025 and 2024", "usual, your last 14
 *   days", "median, your last 4 days".
 */
export function bandBasisLabel(band: UsualBand): string {
  switch (band.basis) {
    case "earlier_years":
      return `usual, this week in ${yearsText(band.years)}`;
    case "recent_days":
      return `usual, ${lastDays(band.days)}`;
    case "median_only":
      return `median, ${lastDays(band.days)}`;
  }
}

/**
 * What the band is over, in a sentence's words.
 *
 * @param band - The snapshot's usual band.
 * @returns E.g. "the middle half and median of 30 days within a week of this
 *   date in 2025 and 2024".
 */
export function bandBasisWords(band: UsualBand): string {
  switch (band.basis) {
    case "earlier_years":
      return `the middle half and median of ${band.days} days within a week of this date in ${yearsText(band.years)}`;
    case "recent_days":
      return `the middle half and median of ${lastDays(band.days)}`;
    case "median_only":
      return `the median of ${lastDays(band.days)}`;
  }
}

/**
 * The chart's accessible name: what is drawn, from the snapshot.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns One or two sentences.
 */
export function todayChartName(snapshot: DashboardSnapshot): string {
  const slots = todaySlots(snapshot);
  const live = slots.find((s) => s.live && s.temperature !== null);
  const band = snapshot.temperature_context.usual_band ?? null;
  const against = band
    ? `against the usual for each hour, ${bandBasisWords(band)}.`
    : "; no usual band in this snapshot yet.";
  const known = slots.some((s) => s.temperature !== null);
  const line = !known
    ? "Today's temperature has no reading yet"
    : live && snapshot.observed_at_ms !== null
      ? `Today's temperature from midnight to ${degrees(live.temperature!)} at ${formatStamp(snapshot.observed_at_ms)}`
      : "Today's temperature from midnight";
  const sentence = band ? `${line}, ${against}` : `${line}${against}`;
  if (!slots.length) return sentence;
  const wet = slots.filter((s) => rainStep(s.rain) !== null).length;
  const rain = wet
    ? ` Rain fell in ${wet} quarter ${wet === 1 ? "hour" : "hours"} today.`
    : " No rain on record today.";
  return sentence + rain;
}

// ---------- drawing ----------

/** A band hour placed on the clock: mid-hour, or held to a midnight. */
interface HourPoint {
  at: number;
  hour: UsualHour;
}

/**
 * Runs of consecutive hours that have what a mark needs, each hour at its
 * middle; a run that starts at hour 0 or ends at hour 23 is held flat to that
 * midnight, so the band spans the whole day.
 *
 * @param hours - The band's 24 hours.
 * @param has - Whether an hour has what the mark needs.
 * @returns The runs, left to right.
 */
function hourRuns(hours: readonly UsualHour[], has: (h: UsualHour) => boolean): HourPoint[][] {
  const out: HourPoint[][] = [];
  let run: HourPoint[] | null = null;
  for (const hour of hours) {
    if (!has(hour)) {
      run = null;
      continue;
    }
    if (!run) {
      run = hour.hour === 0 ? [{ at: 0, hour }] : [];
      out.push(run);
    }
    run.push({ at: hour.hour + 0.5, hour });
    if (hour.hour === 23) run.push({ at: 24, hour });
  }
  return out;
}

/**
 * Runs of today's archived slots with a reading: one polyline each.
 *
 * @param slots - Today's slots.
 * @returns The runs, oldest first; the live slot is never in one.
 */
function lineRuns(slots: readonly Slot[]): Slot[][] {
  const out: Slot[][] = [];
  let run: Slot[] | null = null;
  for (const s of slots) {
    if (s.live || s.temperature === null) {
      run = null;
      continue;
    }
    if (!run || run.at(-1)!.i !== s.i - 1) {
      run = [];
      out.push(run);
    }
    run.push(s);
  }
  return out;
}

const fixed = (v: number) => +v.toFixed(1);

/**
 * Draw the chart at a width into its host.
 *
 * @param host - The chart block.
 * @param snapshot - The dashboard snapshot.
 * @param width - Pixels.
 */
function draw(host: HTMLElement, snapshot: DashboardSnapshot, width: number): void {
  const band = snapshot.temperature_context.usual_band ?? null;
  const slots = todaySlots(snapshot);
  const live = slots.find((s) => s.live && s.temperature !== null) ?? null;
  const narrow = width < NARROW;
  const plotH = narrow ? PLOT_H.narrow : PLOT_H.wide;
  const plotBottom = TOP + plotH;
  const rainTop = plotBottom + RAIN_GAP;
  const height = rainTop + RAIN_H + X_LABELS_H;
  const left = LEFT;
  const right = width - RIGHT_PAD;
  const { svg, hatch } = canvas(width, height, todayChartName(snapshot), "today-svg");
  const x = scale(0, 24, left, right);

  // The y scale holds today, now and the whole band.
  const values: number[] = slots.map((s) => s.temperature).filter((t): t is number => t !== null);
  for (const h of band?.hours ?? []) {
    for (const v of [h.q1, h.median, h.q3]) if (v !== null) values.push(v);
  }
  const lo = values.length ? Math.min(...values) - 1 : 40;
  const hi = values.length ? Math.max(...values) + 1 : 60;
  const ticks = niceTicks(lo, hi, yTickSteps(width));
  // Three ticks on a phone: a third above when the span holds only two.
  if (narrow && ticks.length === 2) ticks.push(+(2 * ticks[1]! - ticks[0]!).toPrecision(6));
  const y = scale(Math.min(lo, ticks[0] ?? lo), Math.max(hi, ticks.at(-1) ?? hi), plotBottom, TOP);

  // The plot's box, for tests and the eye: no fill.
  svg.append(
    svgEl("rect", { x: left, y: TOP, width: right - left, height: plotH, class: "today-plot" }),
  );
  for (const tick of ticks) {
    const at = fixed(y(tick));
    svg.append(svgEl("line", { x1: left, x2: right, y1: at, y2: at, class: "today-grid" }));
    svg.append(
      svgEl(
        "text",
        { x: left - 6, y: at + 4, class: "today-tick", "text-anchor": "end" },
        degrees(tick, 0),
      ),
    );
  }

  // The band, then its median over it.
  if (band) {
    const at = (p: HourPoint, v: number) => `${fixed(x(p.at))},${fixed(y(v))}`;
    for (const run of hourRuns(band.hours, (h) => h.q1 !== null && h.q3 !== null)) {
      const top = run.map((p) => at(p, p.hour.q3!));
      const bottom = run.map((p) => at(p, p.hour.q1!));
      svg.append(
        svgEl("path", {
          d: `M${top.join(" L")} L${[...bottom].reverse().join(" L")} Z`,
          class: "today-band",
        }),
      );
      // Edged along its top and foot only: no edge stands at a midnight.
      for (const edge of [top, bottom]) {
        svg.append(svgEl("polyline", { points: edge.join(" "), class: "today-band-edge" }));
      }
    }
    for (const run of hourRuns(band.hours, (h) => h.median !== null)) {
      const points = run.map((p) => at(p, p.hour.median!)).join(" ");
      svg.append(svgEl("polyline", { points, class: "today-median" }));
    }
    svg.append(
      svgEl(
        "text",
        { x: right, y: plotBottom - 6, class: "today-basis", "text-anchor": "end" },
        bandBasisLabel(band),
      ),
    );
  }

  // Today's line, joined to the live reading.
  const point = (s: Slot) => `${fixed(x(s.at))},${fixed(y(s.temperature!))}`;
  const segments = lineRuns(slots);
  const last = segments.at(-1)?.at(-1) ?? null;
  if (live && last && last.i === live.i - 1) segments.at(-1)!.push(live);
  for (const seg of segments) {
    const points = seg.map(point);
    // A lone reading still needs a length to show.
    if (points.length === 1) points.push(points[0]!);
    svg.append(svgEl("polyline", { points: points.join(" "), class: "today-line" }));
  }
  if (live && last && last.i !== live.i - 1) {
    svg.append(svgEl("polyline", { points: `${point(last)} ${point(live)}`, class: "today-join" }));
  }

  // The rain track along the foot: a bar per wet slot, a hatch per gap in the record
  // (the unarchived tail is told by the dotted join, not hatched here).
  const slotW = x(0.25) - x(0);
  svg.append(
    svgEl("line", {
      x1: left,
      x2: right,
      y1: rainTop + RAIN_H,
      y2: rainTop + RAIN_H,
      class: "today-rain-floor",
    }),
  );
  const archived = snapshot.archive_newest_ms;
  const rain = svgEl("g", { class: "today-rain" });
  for (const s of slots) {
    const from = fixed(x(s.mid) - slotW / 2);
    if (s.rain === null) {
      if (archived === null || s.start > archived) continue;
      rain.append(
        svgEl("rect", {
          x: from,
          y: rainTop,
          width: fixed(slotW),
          height: RAIN_H,
          fill: hatch,
          class: "today-rain-hatch",
        }),
      );
      continue;
    }
    const step = rainStep(s.rain);
    if (step === null) continue;
    const h = fixed((step / RAIN_STEP_COUNT) * RAIN_H);
    const bar = svgEl("rect", {
      x: from,
      y: fixed(rainTop + RAIN_H - h),
      width: fixed(Math.max(1, slotW - 0.5)),
      height: h,
    });
    bar.dataset.rain = String(step);
    rain.append(bar);
  }
  svg.append(rain);

  // The times of day; the closing midnight once there is room for it.
  const labelY = rainTop + RAIN_H + 17;
  for (const [hour, text] of [
    ...[0, 6, 12, 18].map((h) => [h, hourText(h)]),
    ...(narrow ? [] : [[24, hourText(0)]]),
  ] as [number, string][]) {
    // The opening midnight centres under the plot's edge, over the y
    // gutter, so it keeps clear of "6 am" on a phone.
    const anchor = hour === 24 ? "end" : "middle";
    svg.append(
      svgEl(
        "text",
        { x: fixed(x(hour)), y: labelY, class: "today-x", "text-anchor": anchor },
        text,
      ),
    );
  }

  // The now dot and its label, last, over everything.
  if (live) {
    const cx = fixed(x(live.at));
    const cy = fixed(y(live.temperature!));
    svg.append(svgEl("circle", { cx, cy, r: 5.5, class: "today-dot" }));
    const text = `now ${degrees(live.temperature!)}`;
    const place = aboveDot(cx, cy, text, left, width, 13);
    svg.append(
      svgEl(
        "text",
        { x: place.x, y: place.y, class: "today-now", "text-anchor": place.anchor },
        text,
      ),
    );
  }

  host.replaceChildren(svg);
}

/**
 * The hero's chart block, drawn at its width (and again when that changes).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The block, `data-part="today-chart"`; append it, then call the
 *   returned `start` once it is in the page so it measures its width.
 */
export function todayChart(snapshot: DashboardSnapshot): { block: HTMLElement; start(): void } {
  const block = make("div", "today-chart");
  block.dataset.part = "today-chart";
  return { block, start: () => void drawSized(block, (w) => draw(block, snapshot, w)) };
}
