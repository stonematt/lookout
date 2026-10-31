/**
 * The Overview's temperature context (#81), three blocks in the spec's order:
 *
 * 1. temperature across periods: one range bar per window (today, yesterday,
 *    7, 30, 90, 365 days), a whisker from the lowest to the highest reading,
 *    a bar over the middle half, and one "now" rule across every row;
 * 2. the ten-day high-low list, each day's low-to-high bar over the usual
 *    low-to-high for its date, in miniature;
 * 3. "now against normal" strips for temperature and for rain so far today.
 *
 * Every number is the snapshot's (`lookout/snapshot/temperature.py`): this
 * module words and draws, it does not compute a comparison. A window or day
 * the gap status marks invalid is hatched with its reason and has no value
 * under it; a prorated one shows `≈` (`gapValue`).
 */
import "./temperature.css";

import { gapLabel, gapSentence, gapValue, isComparable } from "../gaps.js";
import type {
  DashboardSnapshot,
  GapStatus,
  NormalStrip,
  TemperatureDay,
  TemperaturePeriod,
} from "../snapshot.js";
import { hatchPattern, niceTicks, scale, svg } from "./hero.js";
import type { OverviewSection } from "./index.js";

const SHORT_DATE = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
const DAY_LABEL = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  weekday: "short",
  month: "short",
  day: "numeric",
});

// ---------- words ----------

const deg = (v: number) => `${v.toFixed(1)}°`;
const degWhole = (v: number) => `${Math.round(v)}°`;
const inches = (v: number) => `${v.toFixed(2)} in`;

/** An ISO day as a UTC date, so the zone cannot move it. */
const dayDate = (day: string) => new Date(`${day}T12:00:00Z`);

/** "2025, 2024 and 2023": years, newest first. */
export function yearsText(years: readonly number[]): string {
  if (years.length < 2) return years.length === 1 ? String(years[0]) : "";
  return `${years.slice(0, -1).join(", ")} and ${years.at(-1)}`;
}

const NAMES: Record<string, string> = {
  today: "Today",
  yesterday: "Yesterday",
  "7d": "7 days",
  "30d": "30 days",
  "90d": "90 days",
  "365d": "365 days",
};

/** "Dec 31 to Jan 6": the window's days, for a row label. */
function spanText(status: GapStatus): string {
  const first = SHORT_DATE.format(dayDate(status.first_day));
  if (status.first_day === status.last_day) return first;
  return `${first} to ${SHORT_DATE.format(dayDate(status.last_day))}`;
}

/** How much of a window's readings `now` sits in, for the finding. */
function periodFinding(snapshot: DashboardSnapshot): string {
  const ctx = snapshot.temperature_context;
  const now = ctx.now;
  if (now === null) return "No current temperature to place in context";
  // The longest windows tell the most; the first that can be compared leads.
  for (const key of ["30d", "7d", "90d", "365d", "yesterday"]) {
    const p = ctx.periods.find((x) => x.status.key === key);
    if (!p || !isComparable(p.status) || p.q1 === null || p.q3 === null) continue;
    const name = key === "yesterday" ? "yesterday" : `the last ${NAMES[key]}`;
    const where = now > p.q3 ? "above" : now < p.q1 ? "below" : "inside";
    return `${deg(now)} now is ${where} the middle half of readings over ${name}`;
  }
  return `${deg(now)} now, with no window complete enough to compare`;
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
  const el = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "tc-svg" });
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
    g.append(svg("line", { class: "tc-grid", x1: x(v), x2: x(v), y1: 0, y2: h }));
    g.append(svg("text", { class: "tc-tick", x: x(v), y: h + 17, "text-anchor": "middle" }, label(v)));
  }
}

/** The hatch over a row whose window cannot be compared, with its reason. */
function hatchRow(
  g: SVGGElement,
  hatchId: string,
  status: GapStatus,
  x0: number,
  x1: number,
  cy: number,
  half: number,
): void {
  g.append(svg("rect", { class: "tc-hatch", x: x0, y: cy - half, width: Math.max(0, x1 - x0), height: half * 2, rx: 4, fill: `url(#${hatchId})` }));
  const label = svg("text", { class: "tc-reason", x: (x0 + x1) / 2, y: cy + 4, "text-anchor": "middle" }, gapLabel(status) ?? "");
  g.append(label);
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
  const name = NAMES[p.status.key] ?? p.status.key;
  if (!isComparable(p.status) || p.low === null || p.high === null || p.q1 === null || p.q3 === null) {
    return `${name}: ${gapSentence(p.status)}`;
  }
  const approx = p.status.status === "prorated" ? "about " : "";
  return (
    `${name} (${spanText(p.status)}): ${approx}low ${deg(p.low)}, middle half ${deg(p.q1)} to ${deg(p.q3)}, ` +
    `high ${deg(p.high)}.${p.status.status === "prorated" ? ` ${gapSentence(p.status)}` : ""}`
  );
}

function periodsChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const ctx = snapshot.temperature_context;
  const now = ctx.now;
  const narrow = width < 520;
  const rowH = 46;
  const left = narrow ? 82 : 118;
  const f = { l: left, r: 12, t: 34, b: 26 };
  const iw = width - f.l - f.r;
  const ih = rowH * ctx.periods.length;
  const height = f.t + ih + f.b;

  const numbers = ctx.periods.flatMap((p) => (isComparable(p.status) && p.low !== null && p.high !== null ? [p.low, p.high] : []));
  if (now !== null) numbers.push(now);
  if (numbers.length === 0 && ctx.periods.every((p) => isComparable(p.status) === false)) {
    numbers.push(40, 60);
  }
  const { lo, hi, ticks } = padded(numbers, 3);
  // Room at each end for the low and high values.
  const x = scale(lo, hi, 34, iw - 34);

  const el = root(width, height, periodFinding(snapshot));
  const hatchId = hatchPattern(el);
  const g = svg("g", { transform: `translate(${f.l},${f.t})` });
  const labels = svg("g", { transform: `translate(0,${f.t})` });
  el.append(g, labels);
  axis(g, ticks, x, ih, degWhole);

  if (now !== null) {
    const nx = x(now);
    g.append(svg("line", { class: "tc-now", x1: nx, x2: nx, y1: -8, y2: ih }));
    g.append(svg("circle", { class: "tc-now-dot", cx: nx, cy: -8, r: 4 }));
    const right = nx + 60 > iw + f.r;
    g.append(svg("text", { class: "tc-note", x: right ? nx - 8 : nx + 8, y: -14, "text-anchor": right ? "end" : "start" }, `now ${deg(now)}`));
  }
  ctx.periods.forEach((p, i) => {
    const cy = i * rowH + rowH / 2;
    labels.append(
      svg("text", { class: "tc-row", x: 2, y: cy - 2 }, NAMES[p.status.key] ?? p.status.key),
      svg("text", { class: "tc-sub", x: 2, y: cy + 13 }, spanText(p.status)),
    );
    if (p.low === null || p.high === null || p.q1 === null || p.q3 === null) {
      hatchRow(g, hatchId, p.status, 0, iw, cy, 15);
      return;
    }
    const lowText = gapValue(p.status, degWhole(p.low))!;
    const highText = gapValue(p.status, degWhole(p.high))!;
    g.append(svg("line", { class: "tc-whisker", x1: x(p.low), x2: x(p.high), y1: cy, y2: cy }));
    g.append(svg("rect", { class: "tc-iqr", x: x(p.q1), y: cy - 6, width: Math.max(2, x(p.q3) - x(p.q1)), height: 12, rx: 3 }));
    g.append(svg("circle", { class: "tc-end", cx: x(p.low), cy, r: 2.6 }), svg("circle", { class: "tc-end", cx: x(p.high), cy, r: 2.6 }));
    g.append(svg("text", { class: "tc-value", x: x(p.low) - 7, y: cy + 4, "text-anchor": "end" }, lowText));
    g.append(svg("text", { class: "tc-value", x: x(p.high) + 7, y: cy + 4, "text-anchor": "start" }, highText));
  });

  return el;
}

// ---------- 2. the ten-day high-low list ----------

/** The temperature ramp's colour at `t`, blended between the tokens. */
export function rampColour(t: number): string {
  const stops: [number, string][] = [
    [28, "--temp-cold"],
    [42, "--temp-cool"],
    [56, "--temp-mild"],
    [70, "--temp-warm"],
    [84, "--temp-hot"],
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
      ? ` Usual for the date: ${deg(d.usual_low)} to ${deg(d.usual_high)}, over ${d.usual_days} days in ${yearsText(d.usual_years)}.`
      : " No earlier year to set it against.";
  return `${when}: low ${deg(d.low)}, high ${deg(d.high)}.${usual}`;
}

function daysChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const ctx = snapshot.temperature_context;
  if (ctx.ten_days.length === 0) return null;
  // Newest first: today leads a phone's scroll.
  const days = [...ctx.ten_days].reverse();
  const narrow = width < 520;
  const rowH = narrow ? 30 : 32;
  const left = narrow ? 72 : 84;
  const f = { l: left, r: 12, t: 8, b: 26 };
  const iw = width - f.l - f.r;
  const ih = rowH * days.length;

  const numbers = days.flatMap((d) => [d.low, d.high, d.usual_low, d.usual_high].filter((v): v is number => v !== null));
  if (numbers.length === 0) numbers.push(40, 60);
  const { lo, hi, ticks } = padded(numbers, 3);
  const x = scale(lo, hi, 22, iw - 22);

  const el = root(width, f.t + ih + f.b, `The last ten days' lows and highs against the usual for each date in ${yearsText(ctx.ten_day_years)}`);
  const hatchId = hatchPattern(el);
  const g = svg("g", { transform: `translate(${f.l},${f.t})` });
  const labels = svg("g", { transform: `translate(0,${f.t})` });
  el.append(g, labels);
  axis(g, ticks, x, ih, degWhole);

  days.forEach((d, i) => {
    const cy = i * rowH + rowH / 2;
    const today = i === 0;
    labels.append(svg("text", { class: `tc-row${today ? " tc-row-today" : ""}`, x: 2, y: cy + 4 }, DAY_LABEL.format(dayDate(d.day))));
    if (d.usual_low !== null && d.usual_high !== null) {
      g.append(svg("rect", { class: "tc-usual", x: x(d.usual_low), y: cy - 10, width: Math.max(2, x(d.usual_high) - x(d.usual_low)), height: 20, rx: 3 }));
    }
    if (d.low === null || d.high === null) {
      hatchRow(g, hatchId, d.status, 0, iw, cy, 11);
      return;
    }
    const gradient = svg("linearGradient", { id: `tc-day-${i}-${d.day}`, x1: 0, x2: 1, y1: 0, y2: 0 });
    const stops = [d.low, d.high];
    for (let t = Math.ceil(d.low / 7) * 7; t < d.high; t += 7) stops.push(t);
    stops.sort((a, b) => a - b);
    for (const t of stops) {
      const stop = svg("stop", { offset: d.high === d.low ? 0 : (t - d.low) / (d.high - d.low) });
      stop.setAttribute("style", `stop-color:${rampColour(t)}`);
      gradient.append(stop);
    }
    g.append(gradient);
    g.append(svg("rect", { class: "tc-day", x: x(d.low), y: cy - 5, width: Math.max(4, x(d.high) - x(d.low)), height: 10, rx: 5, fill: `url(#${gradient.id})` }));
    g.append(svg("text", { class: "tc-value", x: x(d.low) - 6, y: cy + 4, "text-anchor": "end" }, degWhole(d.low)));
    g.append(svg("text", { class: "tc-value", x: x(d.high) + 6, y: cy + 4, "text-anchor": "start" }, degWhole(d.high)));
  });
  return el;
}

// ---------- 3. now against normal ----------

/** The claim line above a strip: the percentage and the sample. */
export function stripClaim(strip: NormalStrip): string {
  if (strip.missing === "no_reading") {
    return strip.kind === "rain" ? "No rain reading to place against normal." : "No temperature reading to place against normal.";
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
      : `Cooler than ${Math.max(0, 100 - below - equal)}% of ${sample}, at this hour.`;
  }
  if (strip.now === 0) {
    return `Dry so far today, as on ${equal}% of ${sample}, by this time of day.`;
  }
  return `More rain so far than ${below}% of ${sample}, by this time of day.`;
}

function nowLabel(strip: NormalStrip): string {
  const v = strip.now ?? 0;
  return strip.kind === "rain" ? `today ${inches(v)} so far` : `now ${deg(v)}`;
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
  const g = svg("g", { transform: `translate(${f.l},${f.t})` });
  el.append(g);
  g.append(svg("line", { class: "tc-base", x1: 0, x2: iw, y1: ih, y2: ih }));
  for (const t of ticks) {
    g.append(svg("line", { class: "tc-base", x1: at(t), x2: at(t), y1: ih, y2: ih + 4 }));
    g.append(svg("text", { class: "tc-tick", x: at(t), y: ih + 18, "text-anchor": t === ticks[0] ? "start" : "middle" }, rain ? (t === 0 ? "0" : t === ticks.at(-1) ? `${t} in` : String(t)) : degWhole(t)));
  }
  strip.values.forEach((v, i) => {
    g.append(svg("circle", { class: `tc-dot tc-dot-${strip.kind}`, cx: at(v), cy: 6 + jitter(i) * (ih - 14), r: 4 }));
  });
  const nx = at(strip.now);
  g.append(svg("line", { class: "tc-now", x1: nx, x2: nx, y1: -6, y2: ih }));
  const right = nx + 140 > iw + f.r;
  g.append(svg("text", { class: "tc-note", x: right ? nx - 6 : nx + 6, y: -10, "text-anchor": right ? "end" : "start" }, nowLabel(strip)));
  return el;
}

// ---------- the section ----------

function block(kind: string, headline: string, standfirst: string): HTMLElement {
  const el = document.createElement("div");
  el.className = "tc-block";
  el.dataset.block = kind;
  const h = document.createElement("h2");
  h.className = "tc-headline";
  h.dataset.part = "headline";
  h.textContent = headline;
  const p = document.createElement("p");
  p.className = "standfirst";
  p.dataset.part = "standfirst";
  p.textContent = standfirst;
  el.append(h, p);
  return el;
}

function source(text: string): HTMLElement {
  const p = document.createElement("p");
  p.className = "ov-source";
  p.dataset.part = "source";
  p.textContent = text;
  return p;
}

function spoken(sentences: string[]): HTMLElement {
  const list = document.createElement("ul");
  list.className = "ov-sr";
  for (const s of sentences) {
    const item = document.createElement("li");
    item.textContent = s;
    list.append(item);
  }
  return list;
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
  return years ? `The last ten days, against the usual for their dates in ${years}` : "The last ten days, with no earlier year to set them against";
}

function render(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const ctx = snapshot.temperature_context;

  // 1. across periods
  const periods = block(
    "periods",
    periodFinding(snapshot),
    "Each row runs from the lowest to the highest reading in the window; the bar covers the middle half of its readings.",
  );
  const periodChart = chartHost("periods-chart");
  periods.append(
    periodChart,
    spoken(ctx.periods.map(periodSentence)),
    source(
      "Five-minute readings, Pacific time, with the live reading as now. A window with a few days missing from the record is marked ≈; one with too many is hatched and not compared.",
    ),
  );

  // 2. ten days
  const days = block(
    "ten-days",
    daysHeadline(snapshot),
    "Newest first. Each bar runs from the day's low to its high; the pale band behind it is the usual low to high for that date.",
  );
  const dayChart = chartHost("days-chart");
  days.append(
    dayChart,
    spoken([...ctx.ten_days].reverse().map(daySentence)),
    source(
      "Usual is the median daily low and median daily high over the days within a week either side of the date, in each earlier year whose record there is complete. Today is the day so far.",
    ),
  );

  // 3. strips
  const strips = block(
    "strips",
    ctx.strips[0]?.missing === null && ctx.strips[0] ? stripHeadline(ctx.strips[0]) : "Now against normal",
    "Each dot is a past day within a week of this date; the rule is now.",
  );
  const stripHosts: [NormalStrip, HTMLElement][] = [];
  for (const strip of ctx.strips) {
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
    strips.append(figure);
  }
  strips.append(
    source(
      "Temperature dots are each past day's mean at this hour. Rain dots are each past day's rain by this time of day, on a square-root scale, so today's total so far is set among like totals.",
    ),
  );

  host.append(periods, days, strips);
  mountChart(periodChart, (w) => periodsChart(snapshot, w));
  mountChart(dayChart, (w) => daysChart(snapshot, w));
  for (const [strip, chart] of stripHosts) mountChart(chart, (w) => stripChart(strip, w));
}

function stripHeadline(strip: NormalStrip): string {
  const years = yearsText(strip.years);
  return `Now against normal for this date in ${years}`;
}

/** Temperature across periods, the ten-day list and the strips. */
export const temperatureContext: OverviewSection = {
  id: "temperature-context",
  label: "Temperature in context",
  render,
};
