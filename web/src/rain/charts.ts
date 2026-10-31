/**
 * The Rain tab's three charts, drawn as SVG at their real pixel width
 * (DESIGN.md, "Chart grammar"): the daily bars, the water-year curves and
 * the hour-by-day heatmap.
 *
 * Shared grammar: hairline gridlines on y only, 12px tabular ticks, labels on
 * the marks haloed in the ground, the 135° hatch for a day not in the record
 * (with its reason in place where there is room), and one claret mark for
 * the subject of the headline. Every chart has a live readout below it:
 * hover, tap, or focus the chart and use the arrow keys, and the readout
 * says what the mark under it means, so nothing relies on a title (phones
 * do not show one).
 */
import { gapLabel, gapSentence } from "../gaps.js";
import { rainStep } from "../overview/ribbon.js";
import type { GapReason, GapStatus, RainTab, RecentRain, WaterYearCurve } from "../snapshot.js";
import { addDays, dayText, hourText, inches } from "./text.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/** Width used before the chart's host has been laid out (a hidden tab). */
export const FALLBACK_WIDTH = 640;

/** About how wide one character of a 12.5px chart label is. */
const LABEL_CHAR_PX = 6.8;

/** Below this chart width, month ticks thin out and labels shorten. */
const NARROW = 520;

/** Month starts as places in a water year, Oct first. */
const MONTH_PLACES = [0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335];
const WATER_MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"];

/** Oct 1 of a non-leap water year: a place's month and day count from it. */
const REFERENCE_OCT_1 = "2000-10-01";

/** What a chart writes to its readout for one position. */
export type Readout = (text: string) => void;

let patternCount = 0;

/** An SVG element with attributes. */
function el<K extends keyof SVGElementTagNameMap>(
  name: K,
  attrs: Record<string, string | number> = {},
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

/** A chart's SVG, with its own hatch pattern; returns the svg and the fill. */
function canvas(width: number, height: number, label: string): { svg: SVGSVGElement; hatch: string } {
  const svg = el("svg", {
    class: "rain-svg",
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": label,
  });
  const id = `rain-hatch-${++patternCount}`;
  const pattern = el("pattern", {
    id,
    width: 8,
    height: 8,
    patternUnits: "userSpaceOnUse",
    patternTransform: "rotate(45)",
  });
  pattern.append(el("rect", { width: 8, height: 8, class: "rain-hatch-b" }));
  pattern.append(el("rect", { width: 4, height: 8, class: "rain-hatch-a" }));
  const defs = el("defs");
  defs.append(pattern);
  svg.append(defs);
  return { svg, hatch: `url(#${id})` };
}

/**
 * Round ticks from 0 up past `max`.
 *
 * @param max - The largest value to show.
 * @param count - About how many ticks.
 * @returns The ticks, ascending, starting at 0.
 */
export function niceTicks(max: number, count = 4): number[] {
  const top = max > 0 ? max : 0.1;
  const raw = top / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw;
  const ticks = [];
  for (let v = 0; v < top + step - 1e-9; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return ticks;
}

/** "0.5", "1", "2.5": a tick value without trailing zeros. */
function tickText(value: number): string {
  return String(Math.round(value * 100) / 100);
}

/** A one-day gap status, so a day's reason reads like any window's. */
export function dayStatus(day: string, reason: GapReason): GapStatus {
  return {
    key: day,
    first_day: day,
    last_day: day,
    days: 1,
    missing_days: 1,
    status: "invalid",
    reason,
  };
}

/** Horizontal gridlines with their tick labels; the top one says the unit. */
function yAxis(
  svg: SVGSVGElement,
  ticks: number[],
  y: (v: number) => number,
  left: number,
  right: number,
): void {
  ticks.forEach((tick, i) => {
    const at = y(tick);
    svg.append(el("line", { x1: left, x2: right, y1: at, y2: at, class: i ? "rain-grid" : "rain-axis" }));
    const unit = i === ticks.length - 1 ? " in" : "";
    svg.append(el("text", { x: left - 6, y: at + 4, class: "rain-tick", "text-anchor": "end" }, tickText(tick) + unit));
  });
}

/**
 * Make an element scrub through positions: hover, tap, or arrow keys.
 *
 * @param target - The focusable chart host.
 * @param count - How many positions (columns).
 * @param at - The position under a pointer x, given the target's box.
 * @param show - Writes the readout for a position.
 * @param rows - For a grid, how many rows; up and down move between them.
 * @param rowAt - The row under a pointer y.
 */
function scrub(
  target: HTMLElement,
  count: number,
  at: (x: number) => number,
  show: (index: number, row: number) => void,
  rows = 1,
  rowAt: (y: number) => number = () => 0,
): void {
  let index = count - 1;
  let row = 0;
  const update = (i: number, r: number) => {
    index = Math.max(0, Math.min(count - 1, i));
    row = Math.max(0, Math.min(rows - 1, r));
    show(index, row);
  };
  target.tabIndex = 0;
  target.onpointermove = target.onpointerdown = (event: PointerEvent) => {
    const box = target.getBoundingClientRect();
    update(at(event.clientX - box.left), rowAt(event.clientY - box.top));
  };
  target.onkeydown = (event: KeyboardEvent) => {
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    };
    if (event.key === "Home") update(0, row);
    else if (event.key === "End") update(count - 1, row);
    else {
      const move = moves[event.key];
      if (!move) return;
      update(index + move[0], row + move[1]);
    }
    event.preventDefault();
  };
}

/** A label haloed in the ground, so it reads over marks. */
function label(x: number, y: number, text: string, cls = "rain-label", anchor = "start"): SVGTextElement {
  return el("text", { x, y, class: cls, "text-anchor": anchor }, text);
}

/** Runs of consecutive indices for which `keep` holds: [first, last]. */
function runs(count: number, keep: (i: number) => boolean): [number, number][] {
  const found: [number, number][] = [];
  let start = -1;
  for (let i = 0; i <= count; i++) {
    if (i < count && keep(i)) {
      if (start < 0) start = i;
    } else if (start >= 0) {
      found.push([start, i - 1]);
      start = -1;
    }
  }
  return found;
}

/** Month ticks under a row of days starting at `first`. */
function dayTicks(svg: SVGSVGElement, first: string, count: number, x: (i: number) => number, y: number): void {
  for (let i = 0; i < count; i++) {
    const day = addDays(first, i);
    if (day.slice(8) !== "01") continue;
    const month = dayText(day).split(" ")[0];
    svg.append(el("line", { x1: x(i), x2: x(i), y1: y - 4, y2: y, class: "rain-axis" }));
    svg.append(el("text", { x: x(i) + 2, y: y + 13, class: "rain-tick" }, month));
  }
}

/** Hatch the days not in the record, with the reason in place where it fits. */
function hatchDays(
  svg: SVGSVGElement,
  recent: RecentRain,
  hatch: string,
  x: (i: number) => number,
  step: number,
  top: number,
  bottom: number,
): void {
  const missing = (i: number) => recent.reason[i] !== null && recent.reason[i] !== "before_archive";
  for (const [a, b] of runs(recent.reason.length, missing)) {
    const width = x(b) + step - x(a);
    svg.append(el("rect", { x: x(a), y: top, width, height: bottom - top, fill: hatch, rx: 2 }));
    const words = gapLabel(dayStatus(recent.first_day, recent.reason[a] as GapReason));
    if (words && width >= words.length * LABEL_CHAR_PX + 8) {
      svg.append(label(x(a) + width / 2, top + 14, words, "rain-label", "middle"));
    }
  }
  const before = runs(recent.reason.length, (i) => recent.reason[i] === "before_archive");
  for (const [, b] of before) {
    if (b + 1 >= recent.reason.length) continue;
    const at = x(b + 1);
    svg.append(el("line", { x1: at, x2: at, y1: top, y2: bottom, class: "rain-edge" }));
    const words = `record begins ${dayText(addDays(recent.first_day, b + 1))}`;
    const anchor = at > 140 ? "end" : "start";
    svg.append(label(anchor === "end" ? at - 4 : at + 4, top + 14, words, "rain-label", anchor));
  }
}

/** One day's sentence: its total, or why it is not counted. */
function daySentence(recent: RecentRain, i: number, isToday: boolean): string {
  const day = addDays(recent.first_day, i);
  const reason = recent.reason[i];
  if (reason) return gapSentence(dayStatus(day, reason));
  const total = recent.rain_in[i] ?? 0;
  const when = `${dayText(day, true)}${isToday ? " (today, live)" : ""}`;
  return total > 0 ? `${when}: ${inches(total)} of rain.` : `${when}: no rain.`;
}

/**
 * The daily bars: one bar per day of the recent window, today marked.
 *
 * @param host - The figure to draw in; its contents are replaced.
 * @param recent - The snapshot's recent days.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawDailyBars(host: HTMLElement, recent: RecentRain, width: number, readout: Readout): void {
  const count = recent.rain_in.length;
  const height = 168;
  const [left, right, top, bottom] = [40, width - 6, 22, height - 22];
  const values = recent.rain_in.filter((v): v is number => v !== null);
  const ticks = niceTicks(Math.max(0.1, ...values));
  const yMax = ticks[ticks.length - 1] ?? 1;
  const y = (v: number) => bottom - ((bottom - top) * v) / yMax;
  const step = (right - left) / count;
  const x = (i: number) => left + i * step;
  const pad = step > 4 ? 1 : 0;

  const { svg, hatch } = canvas(width, height, `Rain each day, ${dayText(recent.first_day)} to today`);
  yAxis(svg, ticks, y, left, right);
  hatchDays(svg, recent, hatch, x, step, top, bottom);
  recent.rain_in.forEach((value, i) => {
    if (value === null || value <= 0) return;
    const isToday = i === count - 1;
    svg.append(
      el("rect", {
        x: x(i) + pad / 2,
        y: y(value),
        width: Math.max(1, step - pad),
        height: bottom - y(value),
        class: isToday ? "rain-bar rain-bar-now" : "rain-bar",
      }),
    );
  });
  const today = recent.rain_in[count - 1];
  if (today !== null && today !== undefined) {
    const at = x(count - 1) + step / 2;
    svg.append(el("line", { x1: at, x2: at, y1: top - 4, y2: bottom, class: "rain-now-rule" }));
    svg.append(label(at, top - 8, `today ${inches(today)}`, "rain-now-label", "end"));
  }
  dayTicks(svg, recent.first_day, count, x, bottom + 4);
  host.replaceChildren(svg);
  scrub(host, count, (px) => Math.floor((px - left) / step), (i) => readout(daySentence(recent, i, i === count - 1)));
}

/** A curve's value at a place, as shown: "≈ 3.10 in", or null. */
function curveValue(curve: WaterYearCurve, place: number): string | null {
  const value = curve.cumulative_in[place];
  if (value === null || value === undefined) return null;
  return `${curve.prorated[place] ? "≈ " : ""}${inches(value)}`;
}

/**
 * The water-year curves: cumulative rain from Oct 1, one line per year, the
 * current year bold and marked "now".
 *
 * @param host - The figure to draw in; its contents are replaced.
 * @param tab - The Rain tab's data.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawYearCurves(host: HTMLElement, tab: RainTab, width: number, readout: Readout): void {
  const narrow = width < NARROW;
  const height = narrow ? 240 : 300;
  // The right margin holds the earlier years' names: "2024 ≈ 18.4 in".
  const [left, right, top, bottom] = [40, width - (narrow ? 92 : 112), 18, height - 24];
  const shown = tab.water_years.filter((c) => c.cumulative_in.some((v) => v !== null));
  const values = shown.flatMap((c) => c.cumulative_in.filter((v): v is number => v !== null));
  const ticks = niceTicks(Math.max(1, ...values));
  const yMax = ticks[ticks.length - 1] ?? 1;
  const y = (v: number) => bottom - ((bottom - top) * v) / yMax;
  const step = (right - left) / 364;
  const x = (place: number) => left + place * step;

  const { svg } = canvas(width, height, "Rain since Oct 1, one line per water year");
  yAxis(svg, ticks, y, left, right);
  MONTH_PLACES.forEach((place, i) => {
    svg.append(el("line", { x1: x(place), x2: x(place), y1: bottom, y2: bottom + 4, class: "rain-axis" }));
    if (narrow && i % 2) return;
    svg.append(el("text", { x: x(place) + 2, y: bottom + 16, class: "rain-tick" }, WATER_MONTHS[i]));
  });

  const current = tab.water_years[tab.water_years.length - 1];
  const earlier = shown.filter((c) => c !== current).reverse();
  const draw = (curve: WaterYearCurve, cls: string) => {
    const keep = (i: number) => curve.cumulative_in[i] !== null;
    for (const [a, b] of runs(curve.cumulative_in.length, keep)) {
      const points = [];
      for (let i = a; i <= b; i++) points.push(`${x(i).toFixed(1)},${y(curve.cumulative_in[i]!).toFixed(1)}`);
      if (points.length === 1) points.push(points[0]);
      svg.append(el("polyline", { points: points.join(" "), class: cls }));
    }
  };

  // Earlier years: thin lines, named at the right edge, nudged apart.
  const ends: { y: number; text: string }[] = [];
  earlier.forEach((curve, i) => {
    draw(curve, `rain-year rain-year-${Math.min(i, 2)}`);
    const last = curve.cumulative_in.length - 1;
    const value = curveValue(curve, last);
    if (value !== null) {
      ends.push({ y: y(curve.cumulative_in[last]!), text: `${curve.water_year} ${value}` });
      return;
    }
    // A line cut short by a gap is named where it stops.
    let stop = last;
    while (stop > 0 && curve.cumulative_in[stop] === null) stop--;
    const at = curve.cumulative_in[stop];
    if (at === null || at === undefined) return;
    const text = `${curve.water_year} ${curveValue(curve, stop)}${narrow ? "" : ", then not compared"}`;
    const room = text.length * LABEL_CHAR_PX;
    const anchor = x(stop) - 6 - room >= 0 ? "end" : "start";
    const lx = anchor === "end" ? x(stop) - 6 : Math.min(x(stop) + 6, width - room);
    svg.append(label(lx, Math.max(top + 12, y(at) - 8), text, "rain-label", anchor));
  });
  ends.sort((a, b) => a.y - b.y);
  let floor = -Infinity;
  for (const end of ends) {
    end.y = Math.max(end.y, floor + 15);
    floor = end.y;
    svg.append(label(right + 6, end.y + 4, end.text));
  }

  // The current year: the bold line, its dot and "now" over it.
  if (current && shown.includes(current)) {
    draw(current, "rain-year rain-year-now");
    const place = current.cumulative_in.length - 1;
    const value = current.cumulative_in[place];
    if (value !== null && value !== undefined) {
      const cx = x(place);
      svg.append(el("circle", { cx, cy: y(value), r: 5, class: "rain-now-dot" }));
      const text = `${current.water_year} now ${curveValue(current, place)}`;
      const anchor = cx > (left + right) / 2 ? "end" : "start";
      svg.append(label(cx + (anchor === "end" ? -8 : 8), Math.max(top + 4, y(value) - 10), text, "rain-now-label", anchor));
    }
  }

  host.replaceChildren(svg);
  scrub(
    host,
    365,
    (px) => Math.round((px - left) / step),
    (place) => {
      const parts = [current, ...earlier]
        .filter((c): c is WaterYearCurve => Boolean(c))
        .filter((c) => place < c.cumulative_in.length)
        .map((c) => `${c.water_year} ${curveValue(c, place) ?? "not compared"}`);
      readout(`By ${dayText(addDays(REFERENCE_OCT_1, place))}: ${parts.length ? parts.join("; ") : "no year reaches this date yet"}.`);
    },
  );
}

/** Hour rows: midnight at the top. */
const HOUR_TICKS = [0, 6, 12, 18];

/**
 * The heatmap: one column per day of the recent window, one row per hour of
 * the day, each cell shaded by its rain in four steps.
 *
 * @param host - The figure to draw in; its contents are replaced.
 * @param recent - The snapshot's recent days.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawHourGrid(host: HTMLElement, recent: RecentRain, width: number, readout: Readout): void {
  const count = recent.hourly_in.length;
  const row = width < NARROW ? 8 : 10;
  const [left, right, top] = [56, width - 6, 4];
  const bottom = top + 24 * row;
  const height = bottom + 22;
  const step = (right - left) / count;
  const x = (i: number) => left + i * step;

  const { svg, hatch } = canvas(width, height, `Rain by the hour, ${dayText(recent.first_day)} to today`);
  recent.hourly_in.forEach((hours, i) => {
    if (recent.reason[i] !== null) return;
    hours.forEach((rain, h) => {
      if (rain === null) return;
      const level = rainStep(rain, 60);
      svg.append(
        el("rect", {
          x: x(i),
          y: top + h * row,
          width: Math.max(1, step - (step > 4 ? 1 : 0)),
          height: row - 1,
          class: level ? `rain-cell rain-step-${level}` : "rain-cell rain-dry",
        }),
      );
    });
  });
  hatchDays(svg, recent, hatch, x, step, top, bottom);
  HOUR_TICKS.forEach((h) => {
    svg.append(el("text", { x: left - 6, y: top + h * row + row, class: "rain-tick", "text-anchor": "end" }, hourText(h)));
  });
  dayTicks(svg, recent.first_day, count, x, bottom + 4);
  host.replaceChildren(svg);
  scrub(
    host,
    count,
    (px) => Math.floor((px - left) / step),
    (i, h) => {
      const day = addDays(recent.first_day, i);
      const reason = recent.reason[i];
      if (reason) return readout(gapSentence(dayStatus(day, reason)));
      const rain = recent.hourly_in[i]?.[h];
      const when = `${dayText(day, true)}, ${hourText(h)}`;
      if (rain === null || rain === undefined) return readout(`${when}: still to come.`);
      readout(rain > 0 ? `${when}: ${inches(rain)} of rain.` : `${when}: no rain.`);
    },
    24,
    (py) => Math.floor((py - top) / row),
  );
}

/**
 * Draw a chart at its host's width, and again whenever that width changes
 * (a hidden tab has no width until it is opened).
 *
 * @param host - The chart's host.
 * @param draw - Draws at a width.
 */
export function drawSized(host: HTMLElement, draw: (width: number) => void): void {
  let drawn = 0;
  const redraw = () => {
    const width = Math.floor(host.clientWidth) || FALLBACK_WIDTH;
    if (width === drawn) return;
    drawn = width;
    draw(width);
  };
  redraw();
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(() => {
      if (host.clientWidth > 0) redraw();
    }).observe(host);
  }
}
