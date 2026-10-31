/**
 * The scaffolding every hand-drawn SVG chart shares (DESIGN.md, "Chart
 * grammar"): elements, a canvas with its own hatch, haloed labels, the y
 * axis, round ticks, scales, scrubbing for the live readout, and drawing at
 * the host's real width. The Overview's hero lead and temperature context,
 * the Rain tab and the Solar tab draw with it; each passes its own class
 * prefix, so its stylesheet keeps its own look.
 */
import { MONTHS, addDays } from "../format.js";
import { reasonLabel, svgHatch } from "../gaps.js";
import type { GapReason } from "../snapshot.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/** Width used before the chart's host has been laid out (a hidden tab). */
export const FALLBACK_WIDTH = 640;

/** About how wide one character of a 12.5px chart label is. */
export const LABEL_CHAR_PX = 6.8;

/** Below this chart width, ticks thin out and labels shorten. */
export const NARROW = 520;

/** What a chart writes to its readout for one position. */
export type Readout = (text: string) => void;

/**
 * An SVG element with attributes and, optionally, text.
 *
 * @param tag - The element name.
 * @param attrs - Its attributes.
 * @param text - Its text.
 * @returns The element.
 */
export function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * A chart's SVG at its pixel size, named for screen readers, with its own
 * hatch pattern.
 *
 * @param width - Pixels.
 * @param height - Pixels.
 * @param label - The chart's accessible name.
 * @param cls - The SVG's class, e.g. "rain-svg".
 * @returns The svg and the hatch's fill.
 */
export function canvas(
  width: number,
  height: number,
  label: string,
  cls: string,
): { svg: SVGSVGElement; hatch: string } {
  const svg = svgEl("svg", {
    class: cls,
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": label,
  });
  return { svg, hatch: svgHatch(svg) };
}

/**
 * A label haloed in the ground (by its class), so it reads over marks.
 *
 * @param x - Its anchor.
 * @param y - Its baseline.
 * @param text - Its words.
 * @param cls - Its class, e.g. "rain-label".
 * @param anchor - `text-anchor`.
 * @returns The text element.
 */
export function chartLabel(
  x: number,
  y: number,
  text: string,
  cls: string,
  anchor = "start",
): SVGTextElement {
  return svgEl("text", { x, y, class: cls, "text-anchor": anchor }, text);
}

/**
 * Hatch the days not in the record over a row of day columns, with the
 * reason in place where it fits. A day before the archive is left to the
 * chart (it has no hatch: the record had not begun).
 *
 * @param svg - The chart.
 * @param reasons - Each day's reason, null for a day in the record.
 * @param hatch - The chart's hatch fill.
 * @param x - Day index to its column's left edge.
 * @param step - A column's width.
 * @param top - The plot's top.
 * @param bottom - The plot's bottom.
 * @param labelCls - The reason's class, e.g. "rain-label".
 */
export function hatchMissingDays(
  svg: SVGSVGElement,
  reasons: (GapReason | null)[],
  hatch: string,
  x: (i: number) => number,
  step: number,
  top: number,
  bottom: number,
  labelCls: string,
): void {
  const missing = (i: number) => reasons[i] !== null && reasons[i] !== "before_archive";
  for (const [a, b] of runs(reasons.length, missing)) {
    const width = x(b) + step - x(a);
    svg.append(svgEl("rect", { x: x(a), y: top, width, height: bottom - top, fill: hatch, rx: 2 }));
    const words = reasonLabel(reasons[a] as GapReason);
    if (width >= words.length * LABEL_CHAR_PX + 8) {
      svg.append(chartLabel(x(a) + width / 2, top + 14, words, labelCls, "middle"));
    }
  }
}

/**
 * Month ticks under a row of day columns: a mark and the month's name at
 * each 1st.
 *
 * @param svg - The chart.
 * @param first - The first column's day, "YYYY-MM-DD".
 * @param count - How many columns.
 * @param x - Day index to its column's left edge.
 * @param y - The ticks' top.
 * @param prefix - The chart family's class prefix, e.g. "rain".
 */
export function monthTicks(
  svg: SVGSVGElement,
  first: string,
  count: number,
  x: (i: number) => number,
  y: number,
  prefix: string,
): void {
  for (let i = 0; i < count; i++) {
    const day = addDays(first, i);
    if (day.slice(8) !== "01") continue;
    svg.append(svgEl("line", { x1: x(i), x2: x(i), y1: y - 4, y2: y, class: `${prefix}-axis` }));
    svg.append(svgEl("text", { x: x(i) + 2, y: y + 13, class: `${prefix}-tick` }, MONTHS[Number(day.slice(5, 7)) - 1]!));
  }
}

/** How a y axis names its unit. */
export interface AxisUnit {
  /** The unit, e.g. " in", "kWh/m²". */
  text: string;
  /** After the top tick's number, or on a line of its own above the axis. */
  place: "tick" | "above";
}

/**
 * Horizontal gridlines with their tick labels: the bottom one an axis, the
 * rest hairlines, classed `<prefix>-axis`, `<prefix>-grid`, `<prefix>-tick`.
 *
 * @param svg - The chart.
 * @param ticks - Values, ascending.
 * @param y - Value to pixel.
 * @param left - The plot's left edge; tick labels end just before it.
 * @param right - The plot's right edge.
 * @param prefix - The chart family's class prefix, e.g. "rain".
 * @param unit - Where the unit goes.
 */
export function yAxis(
  svg: SVGSVGElement,
  ticks: number[],
  y: (v: number) => number,
  left: number,
  right: number,
  prefix: string,
  unit: AxisUnit,
): void {
  ticks.forEach((tick, i) => {
    const at = y(tick);
    svg.append(svgEl("line", { x1: left, x2: right, y1: at, y2: at, class: i ? `${prefix}-grid` : `${prefix}-axis` }));
    const suffix = unit.place === "tick" && i === ticks.length - 1 ? unit.text : "";
    const text = String(Math.round(tick * 100) / 100) + suffix;
    svg.append(svgEl("text", { x: left - 6, y: at + 4, class: `${prefix}-tick`, "text-anchor": "end" }, text));
  });
  if (unit.place === "above") {
    // Above the axis, where it never clips at phone width.
    const topTick = ticks[ticks.length - 1] ?? 0;
    svg.append(svgEl("text", { x: 0, y: Math.max(12, y(topTick) - 12), class: `${prefix}-tick` }, unit.text));
  }
}

/**
 * Round ticks from 0 up past `max`.
 *
 * @param max - The largest value to show.
 * @param count - About how many ticks.
 * @returns The ticks, ascending, starting at 0.
 */
export function zeroTicks(max: number, count = 4): number[] {
  const top = max > 0 ? max : 0.1;
  const raw = top / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw;
  const ticks = [];
  for (let v = 0; v < top + step - 1e-9; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return ticks;
}

/**
 * Up to five round ticks over [lo, hi].
 *
 * @param lo - The low end.
 * @param hi - The high end.
 * @returns The ticks, ascending.
 */
export function niceTicks(lo: number, hi: number): number[] {
  const span = hi - lo || 1;
  const raw = span / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= 5) ?? raw;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(Number(v.toPrecision(6)));
  return ticks;
}

/**
 * A linear scale from a domain to a range.
 *
 * @returns Value to pixel; a flat domain maps everything to `r0`.
 */
export const scale = (d0: number, d1: number, r0: number, r1: number) => (v: number) =>
  d1 === d0 ? r0 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

/**
 * Runs of consecutive indices for which `keep` holds.
 *
 * @param count - How many indices.
 * @param keep - Whether an index belongs.
 * @returns Each run as [first, last].
 */
export function runs(count: number, keep: (i: number) => boolean): [number, number][] {
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
export function scrub(
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
