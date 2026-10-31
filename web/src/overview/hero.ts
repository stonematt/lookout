/**
 * The Overview's hero lead (DESIGN.md, "The Adaptive Lead Rule"): the most
 * notable thing right now, chosen in Python (`lookout/snapshot/lead.py`),
 * and a "Why this leads" disclosure listing every rule with its live value.
 *
 * The section runs finding headline, standfirst, the hero facts, the chart,
 * the source line, then the disclosure. The chart follows "Bold now over a
 * soft usual range": the shaded band is the middle half of the comparison
 * set, the dashed line its median, the bold line now, ending in a dot with
 * its value. Where the live reading is ahead of the archive, that span is
 * hatched and the live point is joined by a dotted line.
 *
 * Every value is the snapshot's: this module words and draws, it does not
 * compute a comparison.
 */
import "./hero.css";

import { formatDuration, formatStamp, STATION_TIME_ZONE } from "../format.js";
import { gapSentence } from "../gaps.js";
import type { DashboardSnapshot, HeroLead, LeadKind, LeadRule } from "../snapshot.js";
import type { OverviewSection } from "./index.js";

const HOUR_MS = 3_600_000;
const DAY_MS = 24 * HOUR_MS;
const SVG_NS = "http://www.w3.org/2000/svg";
/** A live reading further ahead of the archive than this is hatched. */
const LAG_HATCH_MS = 15 * 60_000;

const MONTH = new Intl.DateTimeFormat("en-US", { timeZone: STATION_TIME_ZONE, month: "long" });
const HOUR = new Intl.DateTimeFormat("en-US", {
  timeZone: STATION_TIME_ZONE,
  hour: "numeric",
  hourCycle: "h23",
});

// ---------- words ----------

const f1 = (v: number) => v.toFixed(1);
const f2 = (v: number) => v.toFixed(2);
const watts = (v: number) => `${Math.round(v)} W/m²`;
const signed = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${f1(Math.abs(v))}°`;
const spellDays = (v: number) => `${Number.isInteger(v) ? v : f1(v)} ${v === 1 ? "day" : "days"}`;
const sinceText = (hours: number) => formatDuration(hours * HOUR_MS);
const daysText = (days: number) => formatDuration(days * DAY_MS);

/** "noon", "3 pm", "midnight" for a time's Pacific hour. */
function hourName(ms: number): string {
  const hour = Number(HOUR.format(ms));
  if (hour === 0) return "midnight";
  if (hour === 12) return "noon";
  return hour < 12 ? `${hour} am` : `${hour - 12} pm`;
}

function rule(lead: HeroLead, kind: LeadKind): LeadRule {
  return lead.rules.find((r) => r.kind === kind)!;
}

/** The compared months' years, newest first, as "2025 and 2024". */
function years(keys: string[]): string {
  const ys = keys.map((key) => key.slice(0, 4));
  return ys.length > 1 ? `${ys.slice(0, -1).join(", ")} and ${ys.at(-1)}` : (ys[0] ?? "");
}

function monthOf(snapshot: DashboardSnapshot): string {
  return MONTH.format(snapshot.observed_at_ms ?? snapshot.built_at_ms);
}

function hourOf(snapshot: DashboardSnapshot): string {
  return hourName(snapshot.observed_at_ms ?? snapshot.built_at_ms);
}

/**
 * The hero's finding: what leads, with its number.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Raining now: 0.30 in so far, against 0.42 in for a usual event".
 */
export function leadHeadline(snapshot: DashboardSnapshot): string {
  const lead = snapshot.hero_lead;
  const now = lead.now;
  switch (lead.kind) {
    case "rain": {
      const usual = lead.usual_median === null ? "" : `, against ${f2(lead.usual_median)} in for a usual event`;
      if (snapshot.active_rain_event) return `Raining now: ${f2(now ?? 0)} in so far${usual}`;
      const since = rule(lead, "rain").value ?? 0;
      return `Rain ended ${sinceText(since)} ago, after ${f2(now ?? 0)} in${usual}`;
    }
    case "temperature": {
      const t = rule(lead, "temperature");
      const dev = t.value ?? 0;
      return (
        `${f1(Math.abs(dev))}° ${dev > 0 ? "warmer" : "cooler"} than usual for ` +
        `${hourOf(snapshot)} in ${monthOf(snapshot)} (${years(t.compared)})`
      );
    }
    case "solar": {
      const usual =
        lead.usual_median === null
          ? ""
          : `, against a usual ${watts(lead.usual_median)} at ${hourOf(snapshot)}`;
      return `Strong sun: ${watts(now ?? 0)} now${usual}`;
    }
    case "calm": {
      const calm = rule(lead, "calm");
      if (calm.value === null) return "No rain on record to count a dry spell from";
      const usual = calm.reference === null ? "" : `, against a usual dry spell of ${spellDays(calm.reference)}`;
      return `${daysText(calm.value)} since it last rained${usual}`;
    }
  }
}

/**
 * The one line under the headline that says how to read the chart.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The standfirst.
 */
export function leadStandfirst(snapshot: DashboardSnapshot): string {
  const lead = snapshot.hero_lead;
  const band = lead.usual_median !== null;
  switch (lead.kind) {
    case "rain":
      return band
        ? `The bold line is this event's rain so far; the shaded band is the middle half of the station's ${lead.sample} other events, the dashed line their median.`
        : "The bold line is this event's rain so far. There are no other events on record to compare it with.";
    case "temperature":
    case "solar": {
      const what = lead.kind === "temperature" ? "temperature" : "sunshine";
      // Solar's band is over the same months as the temperature median.
      const compared = rule(lead, "temperature").compared;
      return band
        ? `The bold line is today's ${what}, hour by hour; the shaded band is the middle half of ${monthOf(snapshot)} readings at each hour in ${years(compared)}, the dashed line their median.`
        : `The bold line is today's ${what}, hour by hour. No earlier ${monthOf(snapshot)} is complete enough in the record to compare.`;
    }
    case "calm":
      return band
        ? `The bold bar is this dry spell so far; the shaded band is the middle half of the ${lead.sample} dry spells in the last 365 days, the dashed line their median.`
        : "The bold bar is this dry spell so far. The usual dry spell is not compared, as the last 365 days are not complete in the record.";
  }
}

/**
 * The hero facts: three values above their labels.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns [value, label] pairs.
 */
export function heroFacts(snapshot: DashboardSnapshot): [string, string][] {
  const lead = snapshot.hero_lead;
  const na = "n/a";
  switch (lead.kind) {
    case "rain": {
      const active = snapshot.active_rain_event;
      const since = rule(lead, "rain").value;
      return [
        [lead.now === null ? na : `${f2(lead.now)} in`, active ? "so far" : "event total"],
        [lead.usual_median === null ? na : `${f2(lead.usual_median)} in`, "usual event"],
        active
          ? [formatDuration(active.duration_minutes * 60_000), "running for"]
          : [since === null ? na : `${sinceText(since)} ago`, "ended"],
      ];
    }
    case "temperature": {
      const t = rule(lead, "temperature");
      return [
        [lead.now === null ? na : `${f1(lead.now)}°`, "now"],
        [t.reference === null ? na : `${f1(t.reference)}°`, `${monthOf(snapshot)} median at ${hourOf(snapshot)}`],
        [t.value === null ? na : signed(t.value), "from the median"],
      ];
    }
    case "solar": {
      const known = lead.trace?.values.filter((v): v is number => v !== null) ?? [];
      return [
        [lead.now === null ? na : watts(lead.now), "now"],
        [lead.usual_median === null ? na : watts(lead.usual_median), `usual at ${hourOf(snapshot)}`],
        [known.length ? watts(Math.max(...known)) : na, "brightest hour today"],
      ];
    }
    case "calm": {
      const calm = rule(lead, "calm");
      const since: [string, string] = [calm.value === null ? na : daysText(calm.value), "since rain"];
      if (calm.reference === null) {
        // No usual spell to compare: say when it last rained instead.
        const last = snapshot.current.last_rain_ms;
        return [since, [last === null ? na : formatStamp(last), "last rain"], ["not compared", "usual dry spell"]];
      }
      return [
        since,
        [spellDays(calm.reference), "usual dry spell"],
        [String(lead.sample), "dry spells compared"],
      ];
    }
  }
}

/**
 * One rule's line in "Why this leads": the rule, then its live value.
 *
 * @param r - The rule.
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Sun at least 400 W/m²: 322 W/m² now".
 */
export function ruleText(r: LeadRule, snapshot: DashboardSnapshot): string {
  const month = monthOf(snapshot);
  switch (r.kind) {
    case "rain": {
      const value = snapshot.active_rain_event
        ? "raining now"
        : r.value === null
          ? "no rain event on record"
          : `the last event ended ${sinceText(r.value)} ago`;
      return `Rain event live or ended in the last ${r.threshold ?? 12} h: ${value}`;
    }
    case "temperature": {
      const head = `Temperature at least ${r.threshold ?? 10}° from the ${month} median at ${hourOf(snapshot)}`;
      if (r.missing === "no_comparison") {
        return `${head}: not compared, as no earlier ${month} is complete in the record`;
      }
      if (r.value === null || r.reference === null) return `${head}: no reading`;
      return `${head}: ${signed(r.value)} now, against ${f1(r.reference)}° (${years(r.compared)})`;
    }
    case "solar":
      return `Sun at least ${watts(r.threshold ?? 400)}: ${r.value === null ? "no reading" : `${watts(r.value)} now`}`;
    case "calm": {
      if (r.value === null) return "Otherwise, calm: no last rain on record";
      const usual =
        r.reference === null
          ? "; the usual dry spell is not compared"
          : `, against a usual dry spell of ${spellDays(r.reference)}`;
      return `Otherwise, calm: ${daysText(r.value)} since it last rained${usual}`;
    }
  }
}

/**
 * The source line: where the numbers came from, gaps and the archive lag.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The sentence.
 */
export function heroSource(snapshot: DashboardSnapshot): string {
  const lead = snapshot.hero_lead;
  const lag = liveLagMs(snapshot) > LAG_HATCH_MS ? " The hatched span is not archived yet; only the live reading is known there." : "";
  switch (lead.kind) {
    case "rain":
      return `Events from the Event Catalog; rain from the station's event counter, in 15-minute slots, with the live total as now.${lag}`;
    case "temperature":
    case "solar":
      return (
        "Today is the archive's hourly means, Pacific time, with the live reading as now. " +
        `The usual range is every reading at that hour in ${monthOf(snapshot)} of earlier years; ` +
        `a month with too many missing days is not compared.${lag}`
      );
    case "calm": {
      const window = snapshot.gaps.windows.find((w) => w.key === "365d");
      const gap = rule(lead, "calm").reference === null && window ? ` ${gapSentence(window)}` : "";
      return (
        "Last rain from the live reading. A dry spell is a run of days under 0.01 in between two rain days; " +
        `a day missing from the record breaks a spell rather than counting as dry.${gap}`
      );
    }
  }
}

function liveLagMs(snapshot: DashboardSnapshot): number {
  const { archive_newest_ms: archived, observed_at_ms: observed } = snapshot;
  return archived === null || observed === null ? 0 : observed - archived;
}

// ---------- the chart ----------

interface Frame {
  w: number;
  h: number;
  l: number;
  r: number;
  t: number;
  b: number;
}

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

const scale = (d0: number, d1: number, r0: number, r1: number) => (v: number) =>
  d1 === d0 ? r0 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

/** Path segments through points, broken where a point is missing. */
function linePath(points: ([number, number] | null)[]): string {
  let d = "";
  let pen = false;
  for (const p of points) {
    if (p === null) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** A filled band between two lines, broken where either is missing. */
function bandPath(xs: number[], lo: (number | null)[], hi: (number | null)[]): string {
  let d = "";
  let run: number[] = [];
  const flush = () => {
    if (run.length > 1) {
      const top = run.map((i) => `${xs[i]!.toFixed(1)},${hi[i]!.toFixed(1)}`);
      const bottom = run.map((i) => `${xs[i]!.toFixed(1)},${lo[i]!.toFixed(1)}`).reverse();
      d += `M${top.join("L")}L${bottom.join("L")}Z`;
    }
    run = [];
  };
  xs.forEach((_, i) => {
    if (lo[i] === null || hi[i] === null) flush();
    else run.push(i);
  });
  flush();
  return d;
}

let hatchSeq = 0;

function hatchPattern(root: SVGSVGElement): string {
  const id = `hero-hatch-${++hatchSeq}`;
  const defs = svg("defs", {});
  const pattern = svg("pattern", {
    id,
    width: 8,
    height: 8,
    patternUnits: "userSpaceOnUse",
    patternTransform: "rotate(45)",
  });
  pattern.append(svg("rect", { width: 8, height: 8, class: "hero-hatch-b" }), svg("rect", { width: 3, height: 8, class: "hero-hatch-a" }));
  defs.append(pattern);
  root.append(defs);
  return id;
}

/** Up to five round ticks over [lo, hi]. */
function niceTicks(lo: number, hi: number): number[] {
  const span = hi - lo || 1;
  const raw = span / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= 5) ?? raw;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(Number(v.toPrecision(6)));
  return ticks;
}

function lineColour(snapshot: DashboardSnapshot): string {
  const lead = snapshot.hero_lead;
  if (lead.kind === "temperature") return (rule(lead, "temperature").value ?? 0) >= 0 ? "warm" : "cold";
  return lead.kind;
}

function nowLabel(lead: HeroLead): string {
  const v = lead.now ?? 0;
  switch (lead.unit) {
    case "in":
      return `now ${f2(v)} in`;
    case "f":
      return `now ${f1(v)}°`;
    case "wm2":
      return `now ${watts(v)}`;
    case "days":
      return `now ${daysText(v)}`;
  }
}

/** The ticks, band, median, bold line and now dot of a time chart. */
function timeChart(
  snapshot: DashboardSnapshot,
  width: number,
): SVGSVGElement | null {
  const lead = snapshot.hero_lead;
  const trace = lead.trace;
  if (!trace || lead.now === null || lead.now_ms === null) return null;
  const slotMs = trace.slot_minutes * 60_000;
  const hourly = lead.kind !== "rain";
  const nowMs = lead.now_ms;
  const t0 = trace.start_ms;
  const t1 = hourly ? t0 + trace.values.length * slotMs : nowMs;
  const narrow = width < 520;
  const f: Frame = { w: width, h: narrow ? 220 : 260, l: 40, r: 12, t: 24, b: 28 };
  const iw = f.w - f.l - f.r;
  const ih = f.h - f.t - f.b;

  // Each slot's point: hourly means at the slot's middle, rain at its end.
  const at = (i: number) => t0 + (hourly ? (i + 0.5) * slotMs : (i + 1) * slotMs);
  const archived = snapshot.archive_newest_ms ?? nowMs;
  const lagging = nowMs - archived > LAG_HATCH_MS;
  const shown = trace.values.map((v, i) =>
    v === null || at(i) > nowMs || (lagging && (hourly ? at(i) > archived : i === trace.values.length - 1))
      ? null
      : v,
  );

  const bandLo = hourly ? trace.usual_low : [];
  const bandHi = hourly ? trace.usual_high : [];
  const numbers = [
    lead.now,
    ...shown,
    ...bandLo,
    ...bandHi,
    lead.usual_low,
    lead.usual_high,
  ].filter((v): v is number => v !== null);
  let lo = lead.unit === "f" ? Math.min(...numbers) : 0;
  let hi = Math.max(...numbers);
  const pad = (hi - lo || 1) * 0.12;
  if (lead.unit === "f") lo -= pad;
  hi += pad;
  const ticks = niceTicks(lo, hi);
  lo = Math.min(lo, ticks[0] ?? lo);
  hi = Math.max(hi, ticks.at(-1) ?? hi);

  const x = scale(t0, t1, 0, iw);
  const y = scale(lo, hi, ih, 0);
  const root = svg("svg", { width: f.w, height: f.h, viewBox: `0 0 ${f.w} ${f.h}`, class: "hero-svg" });
  const hatchId = hatchPattern(root);
  const g = svg("g", { transform: `translate(${f.l},${f.t})` });
  root.append(g);

  const unit = lead.unit === "f" ? "°" : "";
  for (const v of ticks) {
    g.append(svg("line", { class: "hero-grid", x1: 0, x2: iw, y1: y(v), y2: y(v) }));
    g.append(svg("text", { class: "hero-tick", x: -6, y: y(v) + 4, "text-anchor": "end" }, `${lead.unit === "in" ? f2(v) : Math.round(v)}${unit}`));
  }
  // Time ticks: every six hours of the day, or the event's start and now.
  if (hourly) {
    for (let i = 0; i <= trace.values.length; i += 1) {
      const t = t0 + i * slotMs;
      const hour = Number(HOUR.format(t));
      if (hour % 6 || i === trace.values.length) continue;
      g.append(svg("line", { class: "hero-grid", x1: x(t), x2: x(t), y1: ih, y2: ih + 4 }));
      g.append(svg("text", { class: "hero-tick", x: x(t), y: ih + 18, "text-anchor": i === 0 ? "start" : "middle" }, hourName(t)));
    }
  } else {
    g.append(svg("text", { class: "hero-tick", x: 0, y: ih + 18 }, `started ${hourName(t0)}`));
  }

  // The usual range: per hour for temperature and solar, one band for rain.
  const band = `hero-band${lead.kind === "solar" ? " hero-band-sun" : ""}`;
  if (hourly) {
    const xs = trace.values.map((_, i) => x(at(i)));
    const yl = bandLo.map((v) => (v === null ? null : y(v)));
    const yh = bandHi.map((v) => (v === null ? null : y(v)));
    g.append(svg("path", { class: band, d: bandPath(xs, yl, yh) }));
    const median = trace.usual_median.map((v, i) => (v === null ? null : ([xs[i]!, y(v)] as [number, number])));
    g.append(svg("path", { class: "hero-median", d: linePath(median) }));
  } else if (lead.usual_low !== null && lead.usual_high !== null && lead.usual_median !== null) {
    g.append(svg("rect", { class: band, x: 0, width: iw, y: y(lead.usual_high), height: Math.max(1, y(lead.usual_low) - y(lead.usual_high)) }));
    g.append(svg("line", { class: "hero-median", x1: 0, x2: iw, y1: y(lead.usual_median), y2: y(lead.usual_median) }));
    g.append(svg("text", { class: "hero-label", x: 4, y: y(lead.usual_high) - 6 }, `usual event ${f2(lead.usual_median)} in`));
  }

  // The span only the live reading covers.
  if (lagging && archived >= t0) {
    g.append(svg("rect", { class: "hero-hatch", x: x(archived), width: Math.max(0, x(nowMs) - x(archived)), y: 0, height: ih, fill: `url(#${hatchId})` }));
    g.append(
      svg("text", { class: "hero-label", x: x(nowMs) - 4, y: ih - 8, "text-anchor": "end" }, hourly ? "only the live reading is known" : "only the live total is known"),
    );
  }

  const points = shown.map((v, i) => (v === null ? null : ([x(at(i)), y(v)] as [number, number])));
  g.append(svg("path", { class: `hero-line hero-line-${lineColour(snapshot)}`, d: linePath(points) }));
  const last = [...points].reverse().find((p) => p !== null);
  const dot: [number, number] = [x(nowMs), y(lead.now)];
  if (last) {
    const join = lagging ? "hero-join" : "hero-line";
    g.append(svg("path", { class: `${join} hero-line-${lineColour(snapshot)}`, d: `M${last[0].toFixed(1)},${last[1].toFixed(1)}L${dot[0].toFixed(1)},${dot[1].toFixed(1)}` }));
  }
  g.append(svg("circle", { class: "hero-dot", cx: dot[0], cy: dot[1], r: 5 }));
  const right = dot[0] + 8 + 110 > iw + f.r;
  g.append(
    svg(
      "text",
      { class: "hero-note", x: right ? dot[0] - 8 : dot[0] + 8, y: dot[1] - 8, "text-anchor": right ? "end" : "start" },
      nowLabel(lead),
    ),
  );
  return root;
}

/** The calm chart: this dry spell as a bold bar over the usual spells. */
function spellChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const lead = snapshot.hero_lead;
  if (lead.now === null) return null;
  const f: Frame = { w: width, h: 96, l: 4, r: 16, t: 24, b: 28 };
  const iw = f.w - f.l - f.r;
  const max = Math.max(lead.now, (lead.usual_high ?? 0) * 1.25, 7) * 1.1;
  const x = scale(0, max, 0, iw);
  const root = svg("svg", { width: f.w, height: f.h, viewBox: `0 0 ${f.w} ${f.h}`, class: "hero-svg" });
  const g = svg("g", { transform: `translate(${f.l},${f.t})` });
  root.append(g);
  if (lead.usual_low !== null && lead.usual_high !== null && lead.usual_median !== null) {
    g.append(svg("rect", { class: "hero-band", x: x(lead.usual_low), width: Math.max(3, x(lead.usual_high) - x(lead.usual_low)), y: 0, height: 34, rx: 3 }));
    g.append(svg("line", { class: "hero-median", x1: x(lead.usual_median), x2: x(lead.usual_median), y1: -2, y2: 36 }));
    g.append(svg("text", { class: "hero-label", x: x(lead.usual_median) + 6, y: 52 }, `usual ${spellDays(lead.usual_median)}`));
  }
  g.append(svg("line", { class: "hero-line hero-line-calm", x1: 0, x2: x(lead.now), y1: 17, y2: 17 }));
  g.append(svg("circle", { class: "hero-dot", cx: x(lead.now), cy: 17, r: 5 }));
  const right = x(lead.now) + 110 > iw;
  g.append(svg("text", { class: "hero-note", x: right ? x(lead.now) - 10 : x(lead.now) + 10, y: 12, "text-anchor": right ? "end" : "start" }, nowLabel(lead)));
  for (let d = 0; d <= max; d += max > 30 ? 10 : max > 14 ? 5 : 2) {
    g.append(svg("text", { class: "hero-tick", x: x(d), y: 68, "text-anchor": d === 0 ? "start" : "middle" }, d === 0 ? "0 days" : String(d)));
  }
  return root;
}

/**
 * The chart's accessible name: what it shows, in words.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The sentence.
 */
export function chartLabel(snapshot: DashboardSnapshot): string {
  return `${leadHeadline(snapshot)}. ${leadStandfirst(snapshot)}`;
}

function drawChart(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const width = Math.floor(host.clientWidth) || 640;
  const chart = snapshot.hero_lead.kind === "calm" ? spellChart(snapshot, width) : timeChart(snapshot, width);
  host.replaceChildren();
  if (!chart) return;
  chart.setAttribute("role", "img");
  chart.setAttribute("aria-label", chartLabel(snapshot));
  host.append(chart);
}

// ---------- the section ----------

function whyThisLeads(snapshot: DashboardSnapshot): HTMLElement {
  const lead = snapshot.hero_lead;
  const details = document.createElement("details");
  details.className = "hero-why";
  details.dataset.part = "why";
  const summary = document.createElement("summary");
  summary.textContent = "Why this leads";
  const list = document.createElement("ol");
  for (const r of lead.rules) {
    const item = document.createElement("li");
    item.dataset.rule = r.kind;
    item.dataset.met = String(r.met);
    if (r.kind === lead.kind) item.dataset.lead = "true";
    const ring = document.createElement("span");
    ring.className = "hero-ring";
    ring.setAttribute("aria-hidden", "true");
    const words = document.createElement("span");
    const state = r.kind === lead.kind ? ", met: this leads" : r.met ? ", met" : ", not met";
    words.textContent = `${ruleText(r, snapshot)}${state}`;
    item.append(ring, words);
    list.append(item);
  }
  details.append(summary, list);
  return details;
}

function render(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const lead = snapshot.hero_lead;
  host.dataset.lead = lead.kind;

  const headline = document.createElement("h2");
  headline.className = "hero-headline";
  headline.dataset.part = "headline";
  headline.textContent = leadHeadline(snapshot);

  const standfirst = document.createElement("p");
  standfirst.className = "standfirst";
  standfirst.dataset.part = "standfirst";
  standfirst.textContent = leadStandfirst(snapshot);

  const facts = document.createElement("dl");
  facts.className = "hero-facts";
  facts.dataset.part = "facts";
  for (const [value, label] of heroFacts(snapshot)) {
    const item = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    item.append(dt, dd);
    facts.append(item);
  }

  const chart = document.createElement("div");
  chart.className = "hero-chart";
  chart.dataset.part = "chart";

  const source = document.createElement("p");
  source.className = "ov-source";
  source.dataset.part = "source";
  source.textContent = heroSource(snapshot);

  host.append(headline, standfirst, facts, chart, source, whyThisLeads(snapshot));
  drawChart(chart, snapshot);
  if (typeof ResizeObserver !== "undefined") {
    let width = chart.clientWidth;
    new ResizeObserver(() => {
      if (Math.abs(chart.clientWidth - width) < 2) return;
      width = chart.clientWidth;
      drawChart(chart, snapshot);
    }).observe(chart);
  }
}

/** The hero lead and "Why this leads", as an Overview section. */
export const heroLead: OverviewSection = {
  id: "hero-lead",
  label: "What leads right now",
  render,
};
