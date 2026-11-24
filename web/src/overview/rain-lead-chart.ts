/**
 * The rain tile's dual chart (Overview C, #661; boards MainDark, Desktop,
 * StoppedC, DesktopStopped): the 3 hours ending now, rain every 5 minutes
 * as columns in `rain-bar` (left axis) and the event's running total as a
 * line in `rain-line` (right axis), labelled in place ("0.31 in so far").
 * The two axes keep the dual form's fixed 10:1 ratio and rescale together
 * (`charts/dual.ts`). Under the baseline a bracket: while raining the last
 * hour and its rain ("last hour 0.08 in"); stopped, the dry stretch from
 * last rain to now ("dry 40 min", #660). Under it the times: the window's
 * start, the event's start, last rain when stopped, and now, any that would
 * touch another dropped.
 *
 * The chart is drawn at its real width (`drawSized`), so its text stays at
 * 12px at 320px, where the boards' scaled viewBox fell below. A slot with
 * no reading is hatched, never read as dry. Every number is the snapshot's
 * `rain_now` (`last_3_hours`, Python's, from the counter helper).
 *
 * The chart sits inside the tile's now link, which says its finding in
 * words, so the chart and its legend are hidden from screen readers.
 *
 * Offers `storyChart`.
 */
import { DUAL_RATIO, dualScale, dualTickText } from "../charts/dual.js";
import { LABEL_CHAR_PX, drawSized, runs, scale, yTickSteps } from "../charts/svg.js";
import { make, svgEl } from "../dom.js";
import { inches, timeParts } from "../format.js";
import { svgHatch } from "../gaps.js";
import type { RainNow, RainSlots } from "../snapshot.js";
import { stoppedAgo } from "./feature.js";

const MINUTE_MS = 60_000;
/** The last-hour bracket's span: the rate's own hour. */
const HOUR_MS = 60 * MINUTE_MS;

/** The chart's rows, top down: label room, plot, bracket, its words, times. */
const TOP = 26;
const PLOT = 72;
const BASE = TOP + PLOT;
const BRACKET = BASE + 8;
const BRACKET_TEXT = BRACKET + 17;
const TIME_TEXT = BRACKET_TEXT + 19;
const HEIGHT = TIME_TEXT + 4;
/** Bold 12px labels run a little wider than regular ones. */
const BOLD_CHAR_PX = LABEL_CHAR_PX * 1.08;

/** A tick's text without its leading zero, as the boards set them: ".03 in". */
const tickText = (ticks: number[]) => dualTickText(ticks, "in").map((t) => t.replace(/^0\./, "."));

/** A label's width at 12px. */
const textWidth = (text: string, bold = false) =>
  text.length * (bold ? BOLD_CHAR_PX : LABEL_CHAR_PX);

/** A time label's place, for the drop-what-touches pass. */
interface TimeLabel {
  x: number;
  text: string;
  anchor: "start" | "middle" | "end";
}

const span = (label: TimeLabel): [number, number] => {
  const w = textWidth(label.text);
  if (label.anchor === "start") return [label.x, label.x + w];
  if (label.anchor === "end") return [label.x - w, label.x];
  return [label.x - w / 2, label.x + w / 2];
};

/**
 * The time labels that fit, in priority order: each one that would touch
 * one already placed, or run off the chart, is dropped.
 *
 * @param wanted - The labels, most wanted first.
 * @param width - The chart's width.
 * @returns The labels kept.
 */
function fitTimes(wanted: TimeLabel[], width: number): TimeLabel[] {
  const kept: TimeLabel[] = [];
  for (const label of wanted) {
    const [a, b] = span(label);
    if (a < 0 || b > width) continue;
    if (kept.some((k) => a < span(k)[1] + 8 && b > span(k)[0] - 8)) continue;
    kept.push(label);
  }
  return kept;
}

/**
 * Draw the chart at a width.
 *
 * @param rain - The snapshot's now story.
 * @param slots - Its chart's slots.
 * @param nowMs - The build time: the window's end.
 * @param width - Pixels.
 * @returns The SVG.
 */
function draw(rain: RainNow, slots: RainSlots, nowMs: number, width: number): SVGSVGElement {
  const n = slots.rain_in.length;
  const slotMs = slots.slot_minutes * MINUTE_MS;
  const end = slots.start_ms + n * slotMs;
  const totals = slots.total_in.filter((v): v is number => v !== null);
  const columns = slots.rain_in.filter((v): v is number => v !== null);
  const dual = dualScale(Math.max(0, ...totals), Math.max(0, ...columns), yTickSteps(width));
  const left = Math.max(...tickText(dual.columnTicks).map((t) => textWidth(t))) + 8;
  const right = width - Math.max(...tickText(dual.totalTicks).map((t) => textWidth(t))) - 8;
  const x = scale(slots.start_ms, end, left, right);
  const yColumn = scale(0, dual.columnTop, BASE, TOP);
  const yTotal = scale(0, dual.totalTop, BASE, TOP);

  const svg = svgEl("svg", {
    class: "rain-lead-svg",
    width,
    height: HEIGHT,
    viewBox: `0 0 ${width} ${HEIGHT}`,
    "aria-hidden": "true",
  });
  const hatch = svgHatch(svg);

  // The grid: one line per tick, both axes' labels on it (10:1).
  const leftText = tickText(dual.columnTicks);
  const rightText = tickText(dual.totalTicks);
  dual.totalTicks.forEach((tick, i) => {
    const at = yTotal(tick);
    svg.append(
      svgEl("line", {
        class: i ? "rain-lead-grid" : "rain-lead-axis",
        x1: left,
        x2: right,
        y1: at,
        y2: at,
      }),
    );
    if (i === 0) return;
    svg.append(
      svgEl(
        "text",
        { class: "rain-lead-tick", x: left - 6, y: at + 4, "text-anchor": "end" },
        leftText[i]!,
      ),
      svgEl("text", { class: "rain-lead-tick", x: right + 6, y: at + 4 }, rightText[i]!),
    );
  });

  // A slot with no reading: hatched over the plot, never drawn as dry.
  for (const [a, b] of runs(n, (i) => slots.rain_in[i] === null)) {
    svg.append(
      svgEl("rect", {
        class: "rain-lead-gap",
        x: x(slots.start_ms + a * slotMs),
        y: TOP,
        width: Math.max(1, x(slots.start_ms + (b + 1) * slotMs) - x(slots.start_ms + a * slotMs)),
        height: PLOT,
        fill: hatch,
      }),
    );
  }

  // The columns, each in its slot.
  const slotW = (right - left) / n;
  const barW = Math.max(2, Math.min(14, slotW * 0.7));
  slots.rain_in.forEach((v, i) => {
    if (v === null || v <= 0) return;
    const top = Math.min(yColumn(v), BASE - 1.5);
    svg.append(
      svgEl("rect", {
        class: "rain-lead-col",
        x: x(slots.start_ms + i * slotMs) + (slotW - barW) / 2,
        y: top,
        width: barW,
        height: BASE - top,
        rx: Math.min(1.5, barW / 2),
      }),
    );
  });

  // The running total, from where the event enters the window.
  const first = slots.total_in.findIndex((v) => v !== null);
  let finalTotal: number | null = null;
  if (first >= 0) {
    const from = slots.total_in[first]! - (slots.rain_in[first] ?? 0);
    const startMs = Math.max(slots.start_ms + first * slotMs, rain.event_start_ms ?? 0);
    const points = [`${x(startMs).toFixed(1)},${yTotal(Math.max(0, from)).toFixed(1)}`];
    for (let i = first; i < n; i++) {
      const v = slots.total_in[i];
      if (v === null || v === undefined) continue;
      points.push(`${x(slots.start_ms + (i + 1) * slotMs).toFixed(1)},${yTotal(v).toFixed(1)}`);
      finalTotal = v;
    }
    const d = `M${points.join(" L")}`;
    svg.append(
      svgEl("path", { class: "rain-lead-halo", d }),
      svgEl("path", { class: "rain-lead-line", d }),
    );
  }

  // The line's label in place, by its dot at last rain.
  const total = rain.event_total_in ?? finalTotal;
  if (finalTotal !== null && total !== null) {
    const lastMs = Math.min(end, Math.max(slots.start_ms, rain.last_rain_ms ?? nowMs));
    // The dot sits on the line at last rain: the total at the end of its slot.
    const slot = Math.min(n - 1, Math.max(0, Math.ceil((lastMs - slots.start_ms) / slotMs) - 1));
    const atLast = slots.total_in[slot] ?? finalTotal;
    const cx = x(Math.min(end, slots.start_ms + (slot + 1) * slotMs));
    const cy = yTotal(atLast);
    svg.append(svgEl("circle", { class: "rain-lead-dot", cx, cy, r: 3.5 }));
    const text = `${inches(total, { digits: 2 })} ${rain.raining ? "so far" : "total"}`;
    const w = textWidth(text, true);
    const y = Math.max(14, cy - 9);
    const before = cx - 8 - w >= 0;
    svg.append(
      svgEl(
        "text",
        {
          class: "rain-lead-label",
          x: before ? cx - 8 : Math.min(cx + 8, width - w),
          y,
          "text-anchor": before ? "end" : "start",
        },
        text,
      ),
    );
  }

  // The bracket: the last hour while raining, the dry stretch once stopped.
  const fromMs = rain.raining
    ? Math.max(slots.start_ms, nowMs - HOUR_MS)
    : Math.max(slots.start_ms, Math.min(nowMs, rain.last_rain_ms ?? nowMs));
  const words = rain.raining
    ? `last hour ${inches(rain.rate_in_hr ?? 0, { digits: 2 })}`
    : `dry ${stoppedAgo(nowMs - (rain.last_rain_ms ?? nowMs))}`;
  const bx0 = x(fromMs);
  const bx1 = x(Math.min(end, nowMs));
  svg.append(
    svgEl("line", { class: "rain-lead-bracket", x1: bx0, x2: bx1, y1: BRACKET, y2: BRACKET }),
    svgEl("line", {
      class: "rain-lead-bracket",
      x1: bx0,
      x2: bx0,
      y1: BRACKET - 4,
      y2: BRACKET + 4,
    }),
    svgEl("line", {
      class: "rain-lead-bracket",
      x1: bx1,
      x2: bx1,
      y1: BRACKET - 4,
      y2: BRACKET + 4,
    }),
  );
  const bw = textWidth(words, true);
  const mid = (bx0 + bx1) / 2;
  const bracketX = Math.min(Math.max(mid, bw / 2), width - bw / 2);
  svg.append(
    svgEl(
      "text",
      { class: "rain-lead-bracket-text", x: bracketX, y: BRACKET_TEXT, "text-anchor": "middle" },
      words,
    ),
  );

  // The times: now first, then the event's start, last rain, the window's start.
  const clock = (ms: number) => timeParts(ms).clock;
  const wanted: TimeLabel[] = [{ x: right, text: clock(end), anchor: "end" }];
  const startMs = rain.event_start_ms;
  if (startMs !== null && startMs > slots.start_ms && startMs < end) {
    wanted.push({ x: x(startMs), text: `${clock(startMs)} start`, anchor: "start" });
  }
  if (!rain.raining && rain.last_rain_ms !== null && rain.last_rain_ms > slots.start_ms) {
    wanted.push({ x: x(rain.last_rain_ms), text: clock(rain.last_rain_ms), anchor: "middle" });
  }
  const opening = timeParts(slots.start_ms);
  wanted.push({ x: left, text: `${opening.clock} ${opening.period}`, anchor: "start" });
  for (const label of fitTimes(wanted, width)) {
    svg.append(
      svgEl(
        "text",
        { class: "rain-lead-time", x: label.x, y: TIME_TEXT, "text-anchor": label.anchor },
        label.text,
      ),
    );
  }
  return svg;
}

/**
 * The chart's legend: the two marks and their axes, in words.
 *
 * @param rain - The snapshot's now story.
 * @returns The legend, hidden from screen readers.
 */
function legend(rain: RainNow): HTMLElement {
  const node = make("div", "rain-lead-legend");
  node.setAttribute("aria-hidden", "true");
  const item = (swatch: string, words: string) => {
    const it = make("span", "rain-lead-key");
    it.append(make("span", swatch), words);
    return it;
  };
  node.append(item("rain-lead-swatch-col", "rain every 5 min (left)"));
  if (rain.event_start_ms !== null) {
    const since = timeParts(rain.event_start_ms).clock;
    node.append(item("rain-lead-swatch-line", `total since ${since} (right)`));
  }
  return node;
}

/**
 * The tile's chart and legend, drawn at the host's width and again when it
 * changes.
 *
 * @param rain - The snapshot's now story.
 * @param slots - Its chart's slots.
 * @param nowMs - The build time.
 * @returns The block and a start function to call once it is in the page.
 */
export function storyChart(
  rain: RainNow,
  slots: RainSlots,
  nowMs: number,
): { block: HTMLElement; start: () => void } {
  const block = make("div", "rain-lead-chart");
  block.dataset.part = "chart";
  block.dataset.ratio = String(DUAL_RATIO);
  const plot = make("div", "rain-lead-plot");
  block.append(plot, legend(rain));
  return {
    block,
    start: () => {
      drawSized(plot, (width) => plot.replaceChildren(draw(rain, slots, nowMs, width)));
    },
  };
}
