/**
 * One rain event's chart, drawn with Plotly (loaded on first use, so the
 * page's first screen never waits for it) in the open event row
 * (`drawEventChart`, #419): the dual form of spec #647, the rain as columns
 * and the running total as a line on one time axis.
 *
 * - The columns: the rain between readings gathered into the finest bins
 *   (5 minutes up) that keep the columns apart at the chart's width, in
 *   `rain-bar`, on the left axis (`rainColumns`), labelled in place ("rain
 *   every 15 min").
 * - The line: the event rain so far as a bold stepped line in `rain-line` on
 *   a halo of the tile's surface (a running total is a line, never a filled
 *   area), on the right axis, against the median event's total, dashed. It
 *   ends in a dot labelled with its total ("0.31 in so far"); the live
 *   event's dot is the `now` mark.
 * - The two axes keep a fixed 10:1 ratio and rescale together
 *   (`web/src/charts/dual.ts`), so the columns and the line share one grid.
 *
 * The time axis reads plain times ("6 pm", "noon"), the date said on the
 * first tick and again where the day turns.
 *
 * DESIGN.md's chart grammar: no line, column or value across a gap. Each
 * reading that follows a gap of more than 10 minutes starts a hatched span
 * labelled "n/a: gap in record"; the live reading past the archive's end is
 * joined by a dotted line over a hatched span, "only the live total is
 * known".
 */
import { stationFormat, dayText, hourText, inches, inchesPerHour } from "../format.js";
import { dualScale, dualTickText } from "../charts/dual.js";
import type { DualScale } from "../charts/dual.js";
import { FALLBACK_WIDTH, NARROW, PILL_PAD } from "../charts/svg.js";
import { plotlyHatch } from "../gaps.js";
import type { EventCurve, RainEventRow } from "../snapshot.js";

type PlotlyModule = typeof import("plotly.js-dist-min");

let plotly: Promise<PlotlyModule> | null = null;

/** Plotly, imported once. */
function loadPlotly(): Promise<PlotlyModule> {
  plotly ??= import("plotly.js-dist-min").then(
    (module) => (module as unknown as { default?: PlotlyModule }).default ?? module,
  );
  return plotly;
}

/** The wall-clock formatter in the station's zone. */
const wallClock = () =>
  stationFormat("en-CA", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  });

/**
 * Epoch ms as the station's wall-clock time Plotly plots, e.g.
 * `2026-01-08 14:00:00`.
 *
 * @param ms - Epoch milliseconds UTC.
 * @returns The local time string.
 */
export function stationWallClock(ms: number): string {
  const p = Object.fromEntries(
    wallClock()
      .formatToParts(ms)
      .map((x) => [x.type, x.value]),
  );
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}`;
}

/**
 * Station wall-clock time as if it were UTC, so steps land on local hours.
 *
 * @param ms - Epoch milliseconds UTC.
 * @returns The wall-clock time's ms, read as UTC.
 */
function wallMs(ms: number): number {
  return Date.parse(`${stationWallClock(ms).replace(" ", "T")}Z`);
}

/** A `wallMs` time as the wall-clock string Plotly plots. */
function wallText(wall: number): string {
  return new Date(wall).toISOString().slice(0, 19).replace("T", " ");
}

/** A `wallMs` time as a clock reading: "4 pm", "4:15 pm". */
function clockText(wall: number): string {
  const at = new Date(wall);
  const [hour, minute] = [at.getUTCHours(), at.getUTCMinutes()];
  if (minute === 0) return hourText(hour);
  return `${hour % 12 || 12}:${String(minute).padStart(2, "0")} ${hour < 12 ? "am" : "pm"}`;
}

/** Tick steps in minutes, finest first. */
const TICK_STEPS = [15, 30, 60, 120, 180, 360, 720, 1440];

/** The plot width one tick label needs, in px. */
const TICK_PX = 56;

/** The chart's side margins, in px: the column ticks left, the total's right. */
const MARGIN = { l: 48, r: 52 };

/** The event time axis's ticks: Plotly's x values and their labels. */
export interface TimeTicks {
  values: string[];
  text: string[];
}

/**
 * The event time axis: plain times at the finest step whose ticks fit the
 * plot, the date said on the first tick and in place of each midnight.
 *
 * @param fromMs - The axis start, epoch ms UTC.
 * @param toMs - The axis end.
 * @param plotPx - The plot's width in px.
 * @returns The tick values (station wall-clock strings) and labels.
 */
export function timeTicks(fromMs: number, toMs: number, plotPx: number): TimeTicks {
  const [from, to] = [wallMs(fromMs), wallMs(toMs)];
  const room = Math.max(2, Math.floor(plotPx / TICK_PX));
  let stamps: number[] = [];
  for (const minutes of TICK_STEPS) {
    const step = minutes * 60_000;
    stamps = [];
    for (let t = Math.ceil(from / step) * step; t <= to; t += step) stamps.push(t);
    if (stamps.length <= room) break;
  }
  // An event of many days: thin the daily ticks to fit.
  const every = Math.ceil(stamps.length / room);
  stamps = stamps.filter((_, i) => i % every === 0);
  const values = stamps.map(wallText);
  const text = stamps.map((t, i) => {
    const date = dayText(values[i]!.slice(0, 10));
    if (t % 86_400_000 === 0) return date;
    const time = clockText(t);
    return i === 0 ? `${time}<br>${date}` : time;
  });
  return { values, text };
}

/** A gap span in a curve: the readings either side of it. */
export interface GapSpan {
  fromMs: number;
  toMs: number;
  /** The live reading past the archive's end, not a gap in the record. */
  liveTail: boolean;
}

/**
 * The spans of a curve not covered by readings.
 *
 * @param curve - The curve.
 * @param live - Whether the event is live (its last gap is the live tail).
 * @returns Each gap, oldest first.
 */
export function gapSpans(curve: EventCurve, live: boolean): GapSpan[] {
  const spans: GapSpan[] = [];
  const last = curve.t_ms.length - 1;
  curve.gap_minutes.forEach((gap, i) => {
    if (gap === null || i === 0) return;
    spans.push({
      fromMs: curve.t_ms[i - 1]!,
      toMs: curve.t_ms[i]!,
      liveTail: live && i === last,
    });
  });
  return spans;
}

/**
 * The accumulation line's points, broken (a null) at every gap so no line
 * is drawn across one. The live tail is left out: it is drawn dotted.
 *
 * @param curve - The curve.
 * @param spans - Its gaps.
 * @returns x (station time strings) and y.
 */
export function accumulationLine(
  curve: EventCurve,
  spans: GapSpan[],
): { x: string[]; y: (number | null)[] } {
  const breaks = new Set(spans.map((s) => s.toMs));
  const tail = spans.find((s) => s.liveTail);
  const x: string[] = [];
  const y: (number | null)[] = [];
  curve.t_ms.forEach((t, i) => {
    if (tail && t === tail.toMs) return;
    if (breaks.has(t)) {
      x.push(stationWallClock(t - 1));
      y.push(null);
    }
    x.push(stationWallClock(t));
    y.push(curve.event_in[i] ?? null);
  });
  return { x, y };
}

/**
 * A CSS custom property's value, e.g. `--rain-line`.
 *
 * @param name - The property.
 * @param from - Where to read it: the root by default; the chart's host for
 *   `--surface`, which a tile sets for what sits in it.
 * @returns Its value.
 */
function token(name: string, from: Element = document.documentElement): string {
  return getComputedStyle(from).getPropertyValue(name).trim();
}

/**
 * The chart's layout: the time axis, the columns' axis on the left with the
 * grid, the total's on the right, tick for tick on the same grid.
 */
function dualLayout(
  ticks: TimeTicks,
  xRange: [string, string],
  scale: DualScale,
): Partial<Plotly.Layout> {
  const tick = { family: token("--sans"), size: 12, color: token("--ink-3") };
  const y = { zeroline: false, fixedrange: true, tickfont: tick, tickmode: "array" as const };
  return {
    height: CHART_HEIGHT,
    margin: { ...MARGIN, t: 8, b: 44 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: token("--sans"), size: 12, color: token("--ink-2") },
    showlegend: false,
    // The hatches and the columns share an axis: drawn over, not side by side.
    barmode: "overlay",
    hovermode: "closest",
    hoverlabel: {
      bgcolor: token("--tile"),
      bordercolor: token("--rule-strong"),
      font: { family: token("--sans"), color: token("--ink") },
    },
    xaxis: {
      type: "date",
      showgrid: false,
      zeroline: false,
      fixedrange: true,
      linecolor: token("--rule-strong"),
      tickfont: tick,
      tickmode: "array",
      tickvals: ticks.values,
      ticktext: ticks.text,
      hoverformat: "%b %-d, %-I:%M %p",
      range: xRange,
    },
    yaxis: {
      ...y,
      range: [0, scale.columnTop],
      gridcolor: token("--rule"),
      tickvals: scale.columnTicks,
      ticktext: dualTickText(scale.columnTicks, "in"),
    },
    yaxis2: {
      ...y,
      overlaying: "y",
      side: "right",
      range: [0, scale.totalTop],
      showgrid: false,
      tickvals: scale.totalTicks,
      ticktext: dualTickText(scale.totalTicks, "in"),
    },
  };
}

/** A hatched column over each gap span, the height of the plot. */
function hatches(spans: GapSpan[], top: number, surface: string): Plotly.Data {
  return {
    type: "bar",
    x: spans.map((s) => stationWallClock((s.fromMs + s.toMs) / 2)),
    y: spans.map(() => top),
    width: spans.map((s) => s.toMs - s.fromMs),
    marker: {
      color: surface,
      line: { width: 0 },
      pattern: plotlyHatch(token("--hatch"), surface),
    },
    hoverinfo: "skip",
  };
}

/**
 * A label's pill, as near the SVG charts' `pillLabel` as Plotly comes: a
 * solid box of the tile's surface 4px past the text. Plotly cannot round
 * its corners.
 */
function pill(surface: string): Partial<Plotly.Annotation> {
  return { bgcolor: surface, borderpad: PILL_PAD, borderwidth: 0 };
}

/** The in-place label over a gap span, when it is wide enough to hold one. */
function gapLabels(
  spans: GapSpan[],
  widthMs: number,
  surface: string,
): Partial<Plotly.Annotation>[] {
  return spans
    .filter((s) => (s.toMs - s.fromMs) / widthMs > 0.12)
    .map((s) => ({
      // The live tail sits at the right edge: anchor its label there.
      x: stationWallClock(s.liveTail ? s.toMs : (s.fromMs + s.toMs) / 2),
      xanchor: s.liveTail ? ("right" as const) : ("center" as const),
      y: 0,
      yanchor: "bottom",
      yshift: 6,
      text: s.liveTail ? "only the live <br>total is known" : "n/a: gap in record",
      showarrow: false,
      font: { size: 12, color: token("--ink-2") },
      ...pill(surface),
    }));
}

/** What the chart is drawn against. */
export interface EventChartOptions {
  /** The catalog's median event total, drawn dashed; null for none. */
  medianIn: number | null;
}

/** The chart's height, which the loading state keeps too. */
const CHART_HEIGHT = 240;

/** Column bins in minutes, finest first. */
const COLUMN_STEPS = [5, 10, 15, 30, 60, 120, 180, 360];

/** The narrowest bin, in px, that still draws as a column apart from the next. */
const COLUMN_PX = 4;

/** Of each bin, the share its column fills: the rest is the gap between columns. */
const COLUMN_FILL = 0.7;

/** The rain between readings, gathered into bins of one width. */
export interface RainColumns {
  /** The bin's width in minutes. */
  minutes: number;
  /** Each column's middle, as station wall-clock strings. */
  x: string[];
  /** The rain in each bin, in inches. */
  y: number[];
  /** Each column's hover: its start, its rain and its rate. */
  text: string[];
}

/**
 * The event's rain as columns: the rain between readings (the rate times
 * the time since the reading before; none after a gap) summed into the
 * finest bins of station time that draw at least `COLUMN_PX` wide, so the
 * columns stay apart on a long event and a phone.
 *
 * @param curve - The curve.
 * @param spanMs - The time axis's span.
 * @param plotPx - The plot's width in px.
 * @returns The bin width and each bin with rain in it, oldest first.
 */
export function rainColumns(curve: EventCurve, spanMs: number, plotPx: number): RainColumns {
  const minutes =
    COLUMN_STEPS.find((m) => ((m * 60_000) / spanMs) * plotPx >= COLUMN_PX) ??
    COLUMN_STEPS[COLUMN_STEPS.length - 1]!;
  const bin = minutes * 60_000;
  const sums = new Map<number, number>();
  curve.t_ms.forEach((t, i) => {
    const rate = curve.rate_in_hr[i];
    if (rate === null || rate === undefined) return;
    // The first reading's rate runs from the reading before the event.
    const since = i === 0 ? t - 300_000 : curve.t_ms[i - 1]!;
    const start = Math.floor((wallMs(t) - 1) / bin) * bin;
    sums.set(start, (sums.get(start) ?? 0) + (rate * (t - since)) / 3_600_000);
  });
  const starts = [...sums.keys()].sort((a, b) => a - b);
  const y = starts.map((s) => Math.round(sums.get(s)! * 1e4) / 1e4);
  return {
    minutes,
    x: starts.map((s) => wallText(s + bin / 2)),
    y,
    text: starts.map((s, i) => {
      const rate = inchesPerHour(y[i]! / (minutes / 60));
      return `${dayText(wallText(s).slice(0, 10))}, ${clockText(s)}, ${minutes} min: ${inches(y[i]!)} (${rate})`;
    }),
  };
}

/** The columns' in-place name: "rain every 15 min", "rain each hour". */
export function columnWords(minutes: number): string {
  if (minutes < 60) return `rain every ${minutes} min`;
  return minutes === 60 ? "rain each hour" : `rain every ${minutes / 60} h`;
}

/**
 * Draw an event's chart into a host: its rain as columns and its running
 * total as a line, against the median event, the two axes at the fixed
 * dual ratio.
 *
 * @param host - The chart element.
 * @param curve - The event's curve.
 * @param row - The event, for its total and whether it is live.
 * @param options - The median event's total.
 * @returns When it is drawn.
 */
export async function drawEventChart(
  host: HTMLElement,
  curve: EventCurve,
  row: RainEventRow,
  options: EventChartOptions,
): Promise<void> {
  const Plotly = await loadPlotly();
  const spans = gapSpans(curve, row.live);
  const lastIn = curve.event_in[curve.event_in.length - 1] ?? row.total_in;
  const endMs = curve.t_ms[curve.t_ms.length - 1] ?? row.end_ms;
  const config: Partial<Plotly.Config> = { displayModeBar: false, responsive: true };
  // One time range, from the reading before the first; the plot is the host
  // less the side margins.
  const [fromMs, toMs] = [curve.t_ms[0]! - 300_000, endMs + 60_000];
  const xRange: [string, string] = [stationWallClock(fromMs), stationWallClock(toMs)];
  const width = host.clientWidth || FALLBACK_WIDTH;
  const plotPx = width - MARGIN.l - MARGIN.r;
  const ticks = timeTicks(fromMs, toMs, plotPx);
  // Drawn again at its real width, and in a new colour scheme.
  watchRedraw(host, () => drawEventChart(host, curve, row, options));
  // The tile's surface, for the hatch's ground, the line's halo and every
  // label's pill.
  const surface = token("--surface", host);
  const rain = token("--rain-line");

  const line = accumulationLine(curve, spans);
  const columns = rainColumns(curve, toMs - fromMs, plotPx);
  const median = options.medianIn;
  const tallest = Math.max(0, ...columns.y);
  const lineMax = Math.max(lastIn, median ?? 0, ...line.y.filter((v): v is number => v !== null));
  // Three y ticks on a phone (under 520px of chart, the Chart Width Rule).
  const scale = dualScale(lineMax, tallest, width < NARROW ? 2 : 4);
  const tail = spans.find((s) => s.liveTail);

  const traces: Plotly.Data[] = [
    hatches(spans, scale.columnTop, surface),
    {
      type: "bar",
      x: columns.x,
      y: columns.y,
      width: columns.x.map(() => columns.minutes * 60_000 * COLUMN_FILL),
      marker: { color: token("--rain-bar"), line: { width: 0 } },
      text: columns.text,
      textposition: "none",
      hovertemplate: "%{text}<extra></extra>",
    },
  ];
  // The line and its marks read on the total's axis, over the columns.
  const onTotal = { yaxis: "y2" } as const;
  if (median !== null) {
    // The median event's total: a dashed level under the bold line.
    traces.push({
      ...onTotal,
      type: "scatter",
      mode: "lines",
      x: xRange,
      y: [median, median],
      line: { color: token("--ink-2"), width: 1.5, dash: "dash" },
      hoverinfo: "skip",
    });
  }
  // A halo of the tile's surface lifts the line off the columns under it.
  traces.push({
    ...onTotal,
    type: "scatter",
    mode: "lines",
    x: line.x,
    y: line.y,
    line: { color: surface, width: 7, shape: "hv" },
    connectgaps: false,
    hoverinfo: "skip",
  });
  traces.push({
    ...onTotal,
    type: "scatter",
    mode: "lines",
    x: line.x,
    y: line.y,
    line: { color: rain, width: 3.5, shape: "hv" },
    connectgaps: false,
    hovertemplate: "%{x|%b %-d, %-I:%M %p}: %{y:.2f} in so far<extra></extra>",
  });
  if (tail) {
    const before = curve.event_in[curve.t_ms.indexOf(tail.fromMs)] ?? null;
    traces.push({
      ...onTotal,
      type: "scatter",
      mode: "lines",
      x: [stationWallClock(tail.fromMs), stationWallClock(tail.toMs)],
      y: [before, lastIn],
      line: { color: rain, width: 3, dash: "dot" },
      hoverinfo: "skip",
    });
  }
  traces.push({
    ...onTotal,
    type: "scatter",
    mode: "markers",
    x: [stationWallClock(endMs)],
    y: [lastIn],
    marker: {
      color: row.live ? token("--now") : rain,
      size: 11,
      line: { color: surface, width: 2 },
    },
    // The end dot sits on the plot's right edge: draw all of it.
    cliponaxis: false,
    hovertemplate: `${row.live ? "now: " : ""}%{y:.2f} in<extra></extra>`,
  });

  const layout = dualLayout(ticks, xRange, scale);
  const widthMs = Math.max(1, endMs - curve.t_ms[0]!);
  layout.annotations = [
    ...gapLabels(spans, widthMs, surface),
    {
      yref: "y2",
      x: stationWallClock(endMs),
      y: lastIn,
      xanchor: "right",
      yanchor: "bottom",
      yshift: 8,
      text: `${inches(lastIn, { digits: 2 })} ${row.live ? "so far" : "total"}`,
      showarrow: false,
      font: { size: 13, color: row.live ? token("--now") : token("--ink") },
      ...pill(surface),
    },
  ];
  if (tallest > 0) {
    // The columns named in place, over the tallest, toward the middle.
    const at = columns.y.indexOf(tallest);
    const right = at > columns.y.length / 2;
    layout.annotations.push({
      x: columns.x[at],
      xanchor: right ? "right" : "left",
      y: tallest,
      yanchor: "bottom",
      yshift: 4,
      text: columnWords(columns.minutes),
      showarrow: false,
      font: { size: 12, color: token("--ink-2") },
      ...pill(surface),
    });
  }
  if (median !== null) {
    layout.annotations.push({
      yref: "y2",
      x: xRange[0],
      xanchor: "left",
      y: median,
      yanchor: "bottom",
      yshift: 2,
      text: `median ${inches(median, { digits: 2 })}`,
      showarrow: false,
      font: { size: 12, color: token("--ink-2") },
      ...pill(surface),
    });
  }
  await Plotly.react(host, traces, layout, config);
}

/** Each watched chart's latest redraw, by its host. */
const redraws = new WeakMap<HTMLElement, () => unknown>();

/**
 * Redraw a Plotly chart when it can next be drawn right: when its host
 * takes a new width (a chart first drawn hidden, in a closed row or a tab
 * not yet open, had no width, so it got the fallback's desktop ticks), and
 * when the colour scheme changes (Plotly bakes the theme's colours in when
 * it draws). A host watched again keeps only its latest redraw (each redraw
 * watches its host again); one no longer in the page stops listening at the
 * next change.
 *
 * @param host - The chart's host.
 * @param redraw - Draws it again, at its width now, in the theme now.
 */
function watchRedraw(host: HTMLElement, redraw: () => unknown): void {
  const known = redraws.has(host);
  redraws.set(host, redraw);
  if (known) return;
  if (typeof ResizeObserver === "function") {
    let drawn = host.clientWidth;
    new ResizeObserver(() => {
      const width = host.clientWidth;
      if (width === 0 || width === drawn) return;
      drawn = width;
      void redraws.get(host)?.();
    }).observe(host);
  }
  if (typeof matchMedia !== "function") return;
  const scheme = matchMedia("(prefers-color-scheme: dark)");
  const onChange = () => {
    if (host.isConnected) void redraws.get(host)?.();
    else scheme.removeEventListener?.("change", onChange);
  };
  scheme.addEventListener?.("change", onChange);
}
