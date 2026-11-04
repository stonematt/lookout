/**
 * The scaffolding every hand-drawn SVG chart shares (DESIGN.md, "Chart
 * grammar"): elements, a canvas with its own hatch, haloed labels, the y
 * axis, round ticks, scales, scrubbing for the live readout, and drawing at
 * the host's real width. The Overview's hero lead and temperature context,
 * the Rain tab and the Solar tab draw with it; each passes its own class
 * prefix, so its stylesheet keeps its own look.
 */
import { svgEl } from "../dom.js";
import { MONTHS, addDays, dayText } from "../format.js";
import { reasonLabel, svgHatch } from "../gaps.js";
import type { GapReason } from "../snapshot.js";

/** Width used before the chart's host has been laid out (a hidden tab). */
export const FALLBACK_WIDTH = 640;

/** About how wide one character of a 12.5px chart label is. */
export const LABEL_CHAR_PX = 6.8;

/**
 * Below this chart width a chart draws its phone layout (DESIGN.md, "The
 * Chart Width Rule"): ticks thin out, labels shorten, the now label sits
 * above its dot, and a chart of days shows its last 30.
 */
export const NARROW = 520;

/**
 * How many steps a chart of this width divides its y axis into, about
 * (`zeroTicks`' count): two on a phone, so three ticks.
 */
export const yTickSteps = (width: number): number => (width < NARROW ? 2 : 4);

/** Days a chart of days shows by default at phone width (the 30 / 90 switch). */
export const PHONE_DAYS = 30;

/** What a chart writes to its readout for one position. */
export type Readout = (text: string) => void;

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
 * Whether a day column is hatched as not in the record: any reason but
 * before the archive, which the chart leaves blank. A tile's source chip
 * counts the same days.
 *
 * @param reason - The day's reason, null for a day in the record.
 * @returns True when the day is hatched.
 */
export function missingDay(reason: GapReason | null): boolean {
  return reason !== null && reason !== "before_archive";
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
  for (const [a, b] of runs(reasons.length, (i) => missingDay(reasons[i] ?? null))) {
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
    svg.append(
      svgEl(
        "text",
        { x: x(i) + 2, y: y + 13, class: `${prefix}-tick` },
        MONTHS[Number(day.slice(5, 7)) - 1]!,
      ),
    );
  }
}

/**
 * Date ticks under a short row of day columns (a month or so): today's
 * date at the right, then every two weeks back, e.g. "Oct 5 · Oct 19 ·
 * Nov 2". A longer row is ticked by month (`monthTicks`).
 *
 * @param svg - The chart.
 * @param first - The first column's day, "YYYY-MM-DD".
 * @param count - How many columns.
 * @param x - Day index to its column's left edge.
 * @param step - A column's width.
 * @param y - The ticks' top.
 * @param prefix - The chart family's class prefix, e.g. "rain".
 */
export function dayTicks(
  svg: SVGSVGElement,
  first: string,
  count: number,
  x: (i: number) => number,
  step: number,
  y: number,
  prefix: string,
): void {
  if (count > 45) {
    monthTicks(svg, first, count, x, y, prefix);
    return;
  }
  for (let i = count - 1; i >= 0; i -= 14) {
    const at = x(i) + step / 2;
    const text = dayText(addDays(first, i));
    // The newest ends at the plot's right edge; one too close to the left
    // edge to centre starts there.
    const anchor =
      i === count - 1 ? "end" : at - (text.length * LABEL_CHAR_PX) / 2 < x(0) ? "start" : "middle";
    const lx = anchor === "end" ? x(i) + step : anchor === "start" ? x(0) : at;
    svg.append(svgEl("line", { x1: at, x2: at, y1: y - 4, y2: y, class: `${prefix}-axis` }));
    svg.append(
      svgEl("text", { x: lx, y: y + 13, class: `${prefix}-tick`, "text-anchor": anchor }, text),
    );
  }
}

/** Where a now label goes on a phone: centred over its dot, kept inside. */
export interface AboveDot {
  x: number;
  y: number;
  anchor: "start" | "middle" | "end";
}

/**
 * The now label above its dot (the phone layout): centred over it, or
 * ending (or starting) at it where centring would run off the plot.
 *
 * @param cx - The dot's centre.
 * @param cy - The dot's centre.
 * @param text - The label.
 * @param left - The plot's left edge.
 * @param right - The plot's right edge (the label may reach the chart's).
 * @param minY - The highest baseline the chart has room for.
 * @returns The label's anchor point and text-anchor.
 */
export function aboveDot(
  cx: number,
  cy: number,
  text: string,
  left: number,
  right: number,
  minY: number,
): AboveDot {
  // The annotation is 13px bold, a little wider than a 12.5px label.
  const half = (text.length * LABEL_CHAR_PX * 1.15) / 2;
  const y = Math.max(minY, cy - 12);
  if (cx + half > right) return { x: Math.min(cx + 6, right), y, anchor: "end" };
  if (cx - half < left) return { x: Math.max(cx - 6, left), y, anchor: "start" };
  return { x: cx, y, anchor: "middle" };
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
    svg.append(
      svgEl("line", {
        x1: left,
        x2: right,
        y1: at,
        y2: at,
        class: i ? `${prefix}-grid` : `${prefix}-axis`,
      }),
    );
    const suffix = unit.place === "tick" && i === ticks.length - 1 ? unit.text : "";
    const text = String(Math.round(tick * 100) / 100) + suffix;
    svg.append(
      svgEl(
        "text",
        { x: left - 6, y: at + 4, class: `${prefix}-tick`, "text-anchor": "end" },
        text,
      ),
    );
  });
  if (unit.place === "above") {
    // Above the axis, where it never clips at phone width.
    const topTick = ticks[ticks.length - 1] ?? 0;
    svg.append(
      svgEl("text", { x: 0, y: Math.max(12, y(topTick) - 12), class: `${prefix}-tick` }, unit.text),
    );
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
 * Round ticks over [lo, hi]: up to five, or as few as a phone wants.
 *
 * @param lo - The low end.
 * @param hi - The high end.
 * @param steps - At most how many steps the span holds; 2 gives three ticks.
 * @returns The ticks, ascending.
 */
export function niceTicks(lo: number, hi: number, steps = 5): number[] {
  const span = hi - lo || 1;
  const raw = span / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step =
    [1, 2, 2.5, 5, 10, 20, 25, 50].map((m) => m * mag).find((s) => span / s <= steps) ?? raw;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step)
    ticks.push(Number(v.toPrecision(6)));
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
 * @returns Draws again at the last width (a switch changed what is shown).
 */
export function drawSized(host: HTMLElement, draw: (width: number) => void): () => void {
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
  return () => draw(drawn || FALLBACK_WIDTH);
}
