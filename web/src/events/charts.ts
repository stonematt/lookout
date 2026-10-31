/**
 * One rain event's two charts, drawn with Plotly (loaded on first use, so
 * the page's first screen never waits for it):
 *
 * - Accumulation: the event rain so far as a bold stepped line over a soft
 *   fill, ending in a dot with its total. The live event's dot is the claret
 *   "now" mark; a completed event's is in the rain hue.
 * - Rate: the rain rate between readings as zero-based columns.
 *
 * DESIGN.md's chart grammar: no line, column or value across a gap. Each
 * reading that follows a gap of more than 10 minutes starts a hatched span
 * labelled "n/a: gap in record"; the live reading past the archive's end is
 * joined by a dotted line over a hatched span, "only the live total is
 * known".
 */
import { STATION_TIME_ZONE, inches } from "../format.js";
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

const PACIFIC = new Intl.DateTimeFormat("en-CA", {
  timeZone: STATION_TIME_ZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
});

/**
 * Epoch ms as the Pacific wall-clock time Plotly plots, e.g.
 * `2026-01-08 14:00:00`.
 *
 * @param ms - Epoch milliseconds UTC.
 * @returns The local time string.
 */
export function pacific(ms: number): string {
  const p = Object.fromEntries(PACIFIC.formatToParts(ms).map((x) => [x.type, x.value]));
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}`;
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
 * @returns x (Pacific time strings) and y.
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
      x.push(pacific(t - 1));
      y.push(null);
    }
    x.push(pacific(t));
    y.push(curve.event_in[i] ?? null);
  });
  return { x, y };
}

/** A CSS custom property's value on the root, e.g. `--rain-3`. */
function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** The parts both charts share. */
function baseLayout(height: number, ySuffix: string): Partial<Plotly.Layout> {
  const tick = { family: token("--sans"), size: 12, color: token("--ink-3") };
  return {
    height,
    margin: { l: 52, r: 12, t: 8, b: 44 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: token("--sans"), size: 12, color: token("--ink-2") },
    showlegend: false,
    bargap: 0,
    hovermode: "closest",
    hoverlabel: {
      bgcolor: token("--surface"),
      bordercolor: token("--rule-strong"),
      font: { family: token("--sans"), color: token("--ink") },
    },
    xaxis: {
      type: "date",
      showgrid: false,
      zeroline: false,
      fixedrange: true,
      linecolor: token("--rule"),
      tickfont: tick,
      hoverformat: "%b %-d, %-I:%M %p",
    },
    yaxis: {
      rangemode: "tozero",
      zeroline: false,
      fixedrange: true,
      gridcolor: token("--rule"),
      tickfont: tick,
      ticksuffix: ySuffix,
    },
  };
}

/** A hatched column over each gap span, the height of the plot. */
function hatches(spans: GapSpan[], top: number): Plotly.Data {
  return {
    type: "bar",
    x: spans.map((s) => pacific((s.fromMs + s.toMs) / 2)),
    y: spans.map(() => top),
    width: spans.map((s) => s.toMs - s.fromMs),
    marker: {
      color: token("--gap-b"),
      line: { width: 0 },
      pattern: plotlyHatch(token("--gap-a"), token("--gap-b")),
    },
    hoverinfo: "skip",
  };
}

/** The in-place label over a gap span, when it is wide enough to hold one. */
function gapLabels(spans: GapSpan[], widthMs: number): Partial<Plotly.Annotation>[] {
  return spans
    .filter((s) => (s.toMs - s.fromMs) / widthMs > 0.12)
    .map((s) => ({
      // The live tail sits at the right edge: anchor its label there.
      x: pacific(s.liveTail ? s.toMs : (s.fromMs + s.toMs) / 2),
      xanchor: s.liveTail ? ("right" as const) : ("center" as const),
      y: 0,
      yanchor: "bottom",
      yshift: 6,
      text: s.liveTail ? "only the live <br>total is known" : "n/a: gap in record",
      showarrow: false,
      font: { size: 12, color: token("--ink-2") },
      bgcolor: token("--ground"),
    }));
}

/**
 * Draw an event's accumulation and rate charts into two hosts.
 *
 * @param hosts - The accumulation and rate chart elements.
 * @param curve - The event's curve.
 * @param row - The event, for its total and whether it is live.
 * @returns When both are drawn.
 */
export async function drawEventCharts(
  hosts: { accumulation: HTMLElement; rate: HTMLElement },
  curve: EventCurve,
  row: RainEventRow,
): Promise<void> {
  const Plotly = await loadPlotly();
  const spans = gapSpans(curve, row.live);
  const line = accumulationLine(curve, spans);
  const lastIn = curve.event_in[curve.event_in.length - 1] ?? row.total_in;
  const endMs = curve.t_ms[curve.t_ms.length - 1] ?? row.end_ms;
  const widthMs = Math.max(1, endMs - curve.t_ms[0]!);
  const topIn = Math.max(lastIn, ...line.y.filter((v): v is number => v !== null)) * 1.12 || 0.1;
  const rain = token("--rain-3");
  const dotColor = row.live ? token("--note") : rain;
  const config: Partial<Plotly.Config> = { displayModeBar: false, responsive: true };

  const tail = spans.find((s) => s.liveTail);
  const accTraces: Plotly.Data[] = [
    hatches(spans, topIn),
    {
      type: "scatter",
      mode: "lines",
      x: line.x,
      y: line.y,
      line: { color: rain, width: 3.5, shape: "hv" },
      fill: "tozeroy",
      fillcolor: token("--rain-0"),
      connectgaps: false,
      hovertemplate: "%{x|%b %-d, %-I:%M %p}: %{y:.2f} in<extra></extra>",
    },
  ];
  if (tail) {
    const before = curve.event_in[curve.t_ms.indexOf(tail.fromMs)] ?? null;
    accTraces.push({
      type: "scatter",
      mode: "lines",
      x: [pacific(tail.fromMs), pacific(tail.toMs)],
      y: [before, lastIn],
      line: { color: rain, width: 3, dash: "dot" },
      hoverinfo: "skip",
    });
  }
  accTraces.push({
    type: "scatter",
    mode: "markers",
    x: [pacific(endMs)],
    y: [lastIn],
    marker: { color: dotColor, size: 11, line: { color: token("--ground"), width: 2 } },
    hovertemplate: `${row.live ? "now: " : ""}%{y:.2f} in<extra></extra>`,
  });
  // Both charts share one time range, from the reading before the first.
  const xRange: [string, string] = [
    pacific(curve.t_ms[0]! - 300_000),
    pacific(endMs + 60_000),
  ];
  const accLayout = baseLayout(240, " in");
  accLayout.xaxis = { ...accLayout.xaxis, range: xRange };
  accLayout.yaxis = { ...accLayout.yaxis, range: [0, topIn] };
  accLayout.annotations = [
    ...gapLabels(spans, widthMs),
    {
      x: pacific(endMs),
      y: lastIn,
      xanchor: "right",
      yanchor: "bottom",
      yshift: 8,
      text: `${row.live ? "now " : ""}${inches(lastIn, { digits: 2 })}`,
      showarrow: false,
      font: { size: 13, color: row.live ? token("--note") : token("--ink") },
      bgcolor: token("--ground"),
    },
  ];
  await Plotly.react(hosts.accumulation, accTraces, accLayout, config);

  const rates = curve.rate_in_hr.map((r) => r ?? null);
  const topRate = Math.max(0.05, ...rates.filter((v): v is number => v !== null)) * 1.15;
  const rateTraces: Plotly.Data[] = [
    hatches(spans, topRate),
    {
      type: "bar",
      x: curve.t_ms.map((t) => pacific(t - 150_000)),
      y: rates,
      width: curve.t_ms.map(() => 300_000),
      marker: { color: token("--rain-2"), line: { width: 0 } },
      hovertemplate: "%{x|%b %-d, %-I:%M %p}: %{y:.2f} in/hr<extra></extra>",
    },
  ];
  const rateLayout = baseLayout(150, "");
  rateLayout.yaxis = { ...rateLayout.yaxis, range: [0, topRate], title: { text: "in/hr", font: { size: 12, color: token("--ink-2") } } };
  rateLayout.margin = { l: 52, r: 12, t: 4, b: 44 };
  rateLayout.xaxis = { ...rateLayout.xaxis, range: xRange };
  await Plotly.react(hosts.rate, rateTraces, rateLayout, config);
}
