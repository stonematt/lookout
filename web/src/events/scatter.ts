/**
 * The Rain Events page's scatter (#666, board EventsBC): every completed
 * event of this water year and the three before it as a dot placed by its
 * total, one row per water year, newest on top. The dot's size and ramp step
 * are its category, which Python decides (`ScatterYear.category`); a size
 * key sits under the chart. The x axis starts at zero and runs on a square
 * root, so a wet station's many small events spread out instead of piling
 * into one column at zero (its ticks say so). Within a row the dots
 * swarm up and down from its centre line so no two overlap at any width
 * (`layoutScatter`); a crowded row grows taller rather than letting dots
 * cover each other. The median event of the catalog (the "usual event", as
 * the feature words it) is a dashed rule, and the live event, which the
 * catalog has not closed, is the one `now` ring under this year's dots.
 *
 * - `layoutScatter`: the chart's geometry at a width, pure, for tests.
 * - `scatterMarkup`: the block's heading, chart host and size key.
 * - `drawScatter`: draw the chart into its host at the host's width.
 */
import {
  NOTE_SIZE,
  canvas,
  chartLabel,
  drawSized,
  NARROW,
  pillLabel,
  zeroTicks,
} from "../charts/svg.js";
import type { Html } from "../dom.js";
import { html, svgEl } from "../dom.js";
import { dayText, inches, localDay } from "../format.js";
import type { DashboardSnapshot, EventCategory, ScatterYear } from "../snapshot.js";

/** Each category's dot radius, smallest first: the size key's too. */
export const RADIUS: Readonly<Record<EventCategory, number>> = {
  sprinkle: 2,
  shower: 3.5,
  rain: 5,
  storm: 7,
};

/** The categories in the size key's order, smallest first. */
const KEY: readonly EventCategory[] = ["sprinkle", "shower", "rain", "storm"];

/** Room left of the plot for a row's water year. */
const YEAR_W = 44;
/** Room above the plot for the usual event's label. */
const TOP = 22;
/** Room under the baseline for the tick labels. */
const AXIS_H = 22;
/** The least space between two dots' edges. */
const DOT_GAP = 1;
/** The space between two rows' outermost dots. */
const ROW_GAP = 10;
/** The now ring's radius, and the space above it. */
const NOW_R = 5;
const NOW_GAP = 8;

/** One dot, placed. */
export interface PlacedDot {
  cx: number;
  cy: number;
  r: number;
  category: EventCategory;
  startMs: number;
  totalIn: number;
}

/** One water year's row, placed. */
export interface PlacedRow {
  waterYear: number;
  /** Its centre line: the year label and the swarm's middle. */
  y: number;
  dots: PlacedDot[];
}

/** The scatter's geometry at one width. */
export interface ScatterLayout {
  width: number;
  height: number;
  /** The x of zero and of the axis' last tick. */
  x0: number;
  x1: number;
  /** The baseline's y. */
  baseline: number;
  ticks: { x: number; value: number }[];
  rows: PlacedRow[];
  /** The usual event's rule, or null with no median. */
  usual: { x: number; valueIn: number } | null;
  /** The live event's ring under this year's row, or null. */
  now: { cx: number; cy: number; valueIn: number } | null;
}

/** What the scatter draws, from the snapshot. */
export interface ScatterInput {
  rows: ScatterYear[];
  /** The catalog's median completed event, or null. */
  usualIn: number | null;
  /** The live event's total, when it started in the top row's water year. */
  liveIn: number | null;
}

/**
 * The scatter's input from a snapshot, or null when it has none: a snapshot
 * written before #666, or no row with a dot and no live event.
 *
 * @param section - The snapshot's Rain Events section.
 * @returns The rows, the usual event and the live total.
 */
export function scatterInput(section: DashboardSnapshot["rain_events"]): ScatterInput | null {
  const rows = section.scatter ?? [];
  const live = section.events.find((e) => e.live);
  const liveIn = live && rows[0]?.water_year === live.water_year ? live.total_in : null;
  if (!rows.some((row) => row.start_ms.length) && liveIn === null) return null;
  return { rows, usualIn: section.all_time.median_in, liveIn };
}

/**
 * Offsets from a row's centre line that keep dots apart: each dot, by total
 * from the smallest, takes the nearest offset, up or down, where it touches
 * no dot already placed.
 *
 * @param xs - Each dot's x.
 * @param rs - Each dot's radius.
 * @returns Each dot's offset, in the input's order.
 */
export function swarm(xs: readonly number[], rs: readonly number[]): number[] {
  const order = xs.map((_, i) => i).sort((a, b) => xs[a]! - xs[b]! || rs[b]! - rs[a]!);
  const placed: { x: number; y: number; r: number }[] = [];
  const offsets = new Array<number>(xs.length).fill(0);
  for (const i of order) {
    const x = xs[i]!;
    const r = rs[i]!;
    const near = placed.filter((p) => Math.abs(p.x - x) < p.r + r + DOT_GAP);
    const clear = (y: number) =>
      near.every((p) => Math.hypot(p.x - x, p.y - y) >= p.r + r + DOT_GAP - 1e-9);
    // The candidates are the centre and each offset that just touches a
    // neighbour, above or below it: the nearest clear one wins.
    const candidates = [0];
    for (const p of near) {
      const dx = p.x - x;
      const dy = Math.sqrt(Math.max(0, (p.r + r + DOT_GAP) ** 2 - dx * dx));
      candidates.push(p.y - dy, p.y + dy);
    }
    candidates.sort((a, b) => Math.abs(a) - Math.abs(b) || a - b);
    // Past every tangent: step outward until clear (a crowded corner).
    let y = candidates.find(clear);
    for (let step = 1; y === undefined; step++) {
      if (clear(-step)) y = -step;
      else if (clear(step)) y = step;
    }
    offsets[i] = y;
    placed.push({ x, y, r });
  }
  return offsets;
}

/**
 * The scatter's geometry at a width: the x scale from zero, each row's dots
 * swarmed clear of each other, the rows stacked as tall as their swarms.
 *
 * @param input - What to draw.
 * @param width - The chart's pixel width.
 * @returns The layout.
 */
export function layoutScatter(input: ScatterInput, width: number): ScatterLayout {
  const largest = Math.max(
    0,
    ...input.rows.flatMap((row) => row.total_in),
    input.liveIn ?? 0,
    input.usualIn ?? 0,
  );
  const nice = zeroTicks(largest, width < NARROW ? 3 : 5);
  const top = nice[nice.length - 1]!;
  const x0 = YEAR_W + RADIUS.storm;
  const x1 = width - RADIUS.storm - 1;
  // A square-root scale from zero: most events are small, so a linear one
  // piles a wet station's sprinkles into one column taller than the page.
  const x = (v: number) => x0 + Math.sqrt(Math.max(0, v) / top) * (x1 - x0);
  const values = sqrtTicks(top, x, width < NARROW ? 40 : 56);

  let y = TOP;
  const rows: PlacedRow[] = input.rows.map((row, index) => {
    const xs = row.total_in.map(x);
    const rs = row.category.map((c) => RADIUS[c] ?? RADIUS.shower);
    const offsets = swarm(xs, rs);
    const above = Math.max(RADIUS.storm, ...offsets.map((o, i) => rs[i]! - o));
    const below = Math.max(RADIUS.storm, ...offsets.map((o, i) => rs[i]! + o));
    const centre = y + ROW_GAP / 2 + above;
    y = centre + below + ROW_GAP / 2;
    if (index === 0 && input.liveIn !== null) y += NOW_GAP + NOW_R * 2;
    return {
      waterYear: row.water_year,
      y: centre,
      dots: xs.map((cx, i) => ({
        cx,
        cy: centre + offsets[i]!,
        r: rs[i]!,
        category: row.category[i]!,
        startMs: row.start_ms[i]!,
        totalIn: row.total_in[i]!,
      })),
    };
  });
  const first = rows[0];
  const now =
    input.liveIn === null || !first
      ? null
      : {
          cx: x(input.liveIn),
          cy:
            Math.max(...first.dots.map((d) => d.cy + d.r), first.y + RADIUS.storm) +
            NOW_GAP +
            NOW_R,
          valueIn: input.liveIn,
        };
  const baseline = y;
  return {
    width,
    height: baseline + AXIS_H,
    x0,
    x1,
    baseline,
    ticks: values.map((value) => ({ x: x(value), value })),
    rows,
    usual: input.usualIn === null ? null : { x: x(input.usualIn), valueIn: input.usualIn },
    now,
  };
}

/** Round values a square-root axis may tick, in inches. */
const TICK_STEPS = [0, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20];

/**
 * The ticks of a square-root axis: zero, the top, and the round values
 * between them at least `gap` pixels from their neighbours.
 *
 * @param top - The axis' last value.
 * @param x - Value to pixel.
 * @param gap - The least pixels between two ticks.
 * @returns The ticks, ascending.
 */
export function sqrtTicks(top: number, x: (v: number) => number, gap: number): number[] {
  const ticks = [0];
  for (const v of TICK_STEPS) {
    if (v <= 0 || v >= top) continue;
    if (x(v) - x(ticks[ticks.length - 1]!) >= gap && x(top) - x(v) >= gap) ticks.push(v);
  }
  ticks.push(top);
  return ticks;
}

/** A tick's words: "0", "0.5", and the last with its unit, "1.5 in". */
function tickText(value: number, last: boolean): string {
  const text = String(value);
  return last ? `${text} in` : text;
}

/** The chart's name for screen readers: each row's count and largest event. */
function scatterLabel(input: ScatterInput): string {
  const rows = input.rows.map((row) => {
    const n = row.start_ms.length;
    if (!n) return `${row.water_year}: no completed events`;
    const most = row.total_in.indexOf(Math.max(...row.total_in));
    const day = dayText(localDay(row.start_ms[most]!));
    const count = `${n} event${n === 1 ? "" : "s"}`;
    return `${row.water_year}: ${count}, the largest ${inches(row.total_in[most]!, { digits: 2 })} on ${day}`;
  });
  const live =
    input.liveIn === null ? "" : ` Raining now: ${inches(input.liveIn, { digits: 2 })} so far.`;
  return `Every completed rain event as a dot placed by its total, one row per water year, the dot's size its category. ${rows.join(". ")}.${live}`;
}

/**
 * The scatter block's markup: its heading with the number of water years,
 * the chart host, and the size key in place under it.
 *
 * @param input - What the chart draws.
 * @returns The markup.
 */
export function scatterMarkup(input: ScatterInput): Html {
  const years = input.rows.length;
  const key = KEY.map((c) => html`<li><span class="ev-sc-swatch ev-sc-${c}"></span>${c}</li>`);
  return html`<section class="ev-scatter" data-part="scatter" aria-labelledby="ev-sc-title">
<div class="ev-sc-head"><h3 class="ev-sc-title" id="ev-sc-title">By size</h3><span class="ev-sc-years">${years} water year${years === 1 ? "" : "s"}</span></div>
<div class="ev-sc-chart"></div>
<ul class="ev-sc-key" aria-label="Dot size: the event's category">${key}</ul>
</section>`;
}

/**
 * Draw the scatter into its host at the host's width, and again whenever
 * that width changes.
 *
 * @param host - The chart's host (`.ev-sc-chart`).
 * @param input - What to draw.
 */
export function drawScatter(host: HTMLElement, input: ScatterInput): void {
  drawSized(host, (width) => host.replaceChildren(scatterSvg(layoutScatter(input, width), input)));
}

/** The scatter's SVG from its layout. */
function scatterSvg(layout: ScatterLayout, input: ScatterInput): SVGSVGElement {
  const { svg } = canvas(layout.width, layout.height, scatterLabel(input), "ev-sc-svg");
  const { baseline, x0, x1 } = layout;
  if (layout.usual) {
    const { x, valueIn } = layout.usual;
    svg.append(svgEl("line", { class: "ev-sc-usual", x1: x, x2: x, y1: TOP - 6, y2: baseline }));
    const text = `usual event ${inches(valueIn, { digits: 2 })}`;
    const fits = x + 4 + text.length * 6.8 <= layout.width;
    svg.append(
      chartLabel(fits ? x + 4 : x - 4, TOP - 10, text, "ev-sc-label", fits ? "start" : "end"),
    );
  }
  for (const row of layout.rows) {
    svg.append(chartLabel(0, row.y + 4, String(row.waterYear), "ev-sc-year"));
    for (const dot of row.dots) {
      const circle = svgEl("circle", {
        class: `ev-sc-dot ev-sc-${dot.category}`,
        cx: dot.cx,
        cy: dot.cy,
        r: dot.r,
        "data-start": dot.startMs,
      });
      circle.append(
        svgEl(
          "title",
          {},
          `${dayText(localDay(dot.startMs), true)}: ${inches(dot.totalIn, { digits: 2 })}, ${dot.category}`,
        ),
      );
      svg.append(circle);
    }
  }
  svg.append(svgEl("line", { class: "ev-sc-axis", x1: x0, x2: x1, y1: baseline, y2: baseline }));
  layout.ticks.forEach(({ x, value }, i) => {
    const last = i === layout.ticks.length - 1;
    svg.append(
      chartLabel(x, baseline + 16, tickText(value, last), "ev-sc-tick", last ? "end" : "middle"),
    );
  });
  if (layout.now) {
    const { cx, cy, valueIn } = layout.now;
    svg.append(svgEl("circle", { class: "ev-sc-now", cx, cy, r: NOW_R }));
    const text = `now ${inches(valueIn, { digits: 2 })}`;
    const fits = cx + NOW_R + 6 + text.length * NOTE_SIZE.charPx <= layout.width;
    svg.append(
      pillLabel(
        fits ? cx + NOW_R + 6 : cx - NOW_R - 6,
        cy + 4.5,
        text,
        "ev-sc-now-label",
        fits ? "start" : "end",
        NOTE_SIZE,
      ),
    );
  }
  return svg;
}
