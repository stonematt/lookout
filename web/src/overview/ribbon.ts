/**
 * The 48-hour condition ribbon (DESIGN.md, "48-hour condition ribbon"): a
 * 52px band of 15-minute slots, each night or day, overprinted by the rain
 * ramp in four steps, with the temperature traced over it and a dot at the
 * right edge for the live value. Slots the archive cannot speak for (a gap,
 * or the span only the live reading covers) are hatched; the live point is
 * joined to the archived trace by a dotted line across that span.
 *
 * The band is drawn in a 1000 x 100 viewBox stretched to the band's width,
 * so it needs no measuring; ticks, hatches and the dot are positioned in %.
 */
import { STATION_TIME_ZONE } from "../format.js";
import type { ConditionRibbon, DashboardSnapshot } from "../snapshot.js";

const HOUR_MS = 3_600_000;
const SVG_NS = "http://www.w3.org/2000/svg";
const VIEW_W = 1000;
const VIEW_H = 100;

/** Rain-rate ceilings (in/hr) of the first three steps; above is step 4. */
const RAIN_STEPS = [0.05, 0.1, 0.3] as const;
const RAIN_STEP_NAMES = ["light", "moderate", "steady", "heavy"] as const;

/**
 * Ticks are dropped this close to either edge. Past `TICK_LATE_PCT` a tick
 * would run into the "now" label on a phone, so it is marked late and shown
 * only on wider screens.
 */
const TICK_MIN_PCT = 4;
const TICK_LATE_PCT = 74;
const TICK_MAX_PCT = 90;

/**
 * The rain-ramp step for one slot's rain, by its rate.
 *
 * @param rainIn - The rain that fell in the slot.
 * @param slotMinutes - The slot's length.
 * @returns 1 (up to 0.05 in/hr) to 4 (over 0.3 in/hr), or null for no rain.
 */
export function rainStep(rainIn: number | null, slotMinutes = 15): 1 | 2 | 3 | 4 | null {
  if (rainIn === null || rainIn <= 0) return null;
  const rate = (rainIn * 60) / slotMinutes;
  const step = RAIN_STEPS.findIndex((ceiling) => rate <= ceiling + 1e-9);
  return (step === -1 ? 4 : step + 1) as 1 | 2 | 3 | 4;
}

/** Hours as `9 h`, `47 h 30 m` or `45 m` (never days: the span is 48 h). */
function hoursText(hours: number): string {
  const minutes = Math.round(hours * 60);
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  // No-break spaces: a duration never wraps across lines.
  if (!h) return `${m}\u00a0m`;
  return m ? `${h}\u00a0h\u00a0${m}\u00a0m` : `${h}\u00a0h`;
}

/**
 * The one-line claim above the band: the range and the hours of rain.
 *
 * @param ribbon - The snapshot's ribbon.
 * @returns e.g. "40° to 60° over the last 48 hours, with rain in 9 h of the
 *   47 h 30 m on record."
 */
export function ribbonClaim(ribbon: ConditionRibbon): string {
  const { temperature_low_f: lo, temperature_high_f: hi } = ribbon;
  const range =
    lo === null || hi === null
      ? "The last 48 hours"
      : `${Math.round(lo)}° to ${Math.round(hi)}° over the last 48 hours`;
  const whole = ribbon.known_hours >= 48;
  if (ribbon.rain_hours <= 0) {
    return whole
      ? `${range}, with no rain.`
      : `${range}, with no rain in the ${hoursText(ribbon.known_hours)} on record.`;
  }
  const of = whole ? "the 48 h" : `the ${hoursText(ribbon.known_hours)} on record`;
  return `${range}, with rain in ${hoursText(ribbon.rain_hours)} of ${of}.`;
}

/** One tick under the band. */
export interface RibbonTick {
  /** Position across the band, 0-100. */
  pct: number;
  /** "6 am", "noon", "6 pm", or the weekday at midnight. */
  label: string;
  /** Close to the "now" label: shown only where the band is wide. */
  late: boolean;
}

const HOUR = new Intl.DateTimeFormat("en-US", {
  timeZone: STATION_TIME_ZONE,
  hour: "numeric",
  hourCycle: "h23",
});
const WEEKDAY = new Intl.DateTimeFormat("en-US", {
  timeZone: STATION_TIME_ZONE,
  weekday: "short",
});

/**
 * Ticks every six hours of Pacific time, clear of the edges.
 *
 * @param ribbon - The snapshot's ribbon.
 * @returns The ticks, left to right.
 */
export function ribbonTicks(ribbon: ConditionRibbon): RibbonTick[] {
  const span = spanMs(ribbon);
  const ticks: RibbonTick[] = [];
  for (let t = Math.ceil(ribbon.start_ms / HOUR_MS) * HOUR_MS; t < ribbon.start_ms + span; t += HOUR_MS) {
    const hour = Number(HOUR.format(t));
    if (hour % 6) continue;
    const pct = ((t - ribbon.start_ms) / span) * 100;
    if (pct < TICK_MIN_PCT || pct > TICK_MAX_PCT) continue;
    const label =
      hour === 0 ? WEEKDAY.format(t) : hour === 12 ? "noon" : hour < 12 ? `${hour} am` : `${hour - 12} pm`;
    ticks.push({ pct, label, late: pct > TICK_LATE_PCT });
  }
  return ticks;
}

function spanMs(ribbon: ConditionRibbon): number {
  return ribbon.temperature_f.length * ribbon.slot_minutes * 60_000;
}

/** A run of slots `[from, to)` sharing one value. */
interface Run<T> {
  from: number;
  to: number;
  value: T;
}

function runs<T>(values: T[]): Run<T>[] {
  const out: Run<T>[] = [];
  values.forEach((value, i) => {
    const last = out.at(-1);
    if (last && last.value === value) last.to = i + 1;
    else out.push({ from: i, to: i + 1, value });
  });
  return out;
}

function svg(tag: string, attrs: Record<string, string | number>): SVGElement {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  return el;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Draw the ribbon for a snapshot.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The ribbon block (claim, band, ticks, key), or null without one.
 */
export function renderRibbon(snapshot: DashboardSnapshot): HTMLElement | null {
  const ribbon = snapshot.condition_ribbon;
  if (!ribbon || !ribbon.temperature_f.length) return null;
  const n = ribbon.temperature_f.length;
  const slotW = VIEW_W / n;
  const x = (i: number) => +(i * slotW).toFixed(2);
  const pct = (i: number) => (i / n) * 100;
  const now = snapshot.current.temperature_f ?? ribbon.temperature_f.at(-1) ?? null;

  const block = el("div", "ribbon");
  block.dataset.part = "ribbon";
  const claim = ribbonClaim(ribbon);
  block.append(el("p", "ribbon-claim", claim));

  const band = el("div", "ribbon-band");
  band.setAttribute("role", "img");
  band.setAttribute(
    "aria-label",
    `48-hour condition ribbon: ${claim.replace(/^The last 48 hours, /, "")} ` +
      "Day and night, rain intensity and the temperature trace, oldest on the left.",
  );

  // Night and day, then rain over them.
  const fills = svg("svg", { viewBox: `0 0 ${VIEW_W} ${VIEW_H}`, preserveAspectRatio: "none" });
  fills.setAttribute("aria-hidden", "true");
  for (const run of runs(ribbon.daylight.map((d) => (d ? "day" : "night")))) {
    const rect = svg("rect", { x: x(run.from), y: 0, width: x(run.to) - x(run.from) + 0.4, height: VIEW_H });
    rect.dataset.light = run.value;
    fills.append(rect);
  }
  for (const run of runs(ribbon.rain_in.map((r) => rainStep(r, ribbon.slot_minutes)))) {
    if (run.value === null) continue;
    const rect = svg("rect", { x: x(run.from), y: 0, width: x(run.to) - x(run.from) + 0.4, height: VIEW_H });
    rect.dataset.rain = String(run.value);
    fills.append(rect);
  }
  band.append(fills);

  // Hatch what the archive cannot speak for (lane I's shared hatching
  // component replaces this local one once it lands).
  const archiveEnd = snapshot.archive_newest_ms;
  for (const run of runs(ribbon.rain_in.map((r) => r === null))) {
    if (!run.value) continue;
    const startMs = ribbon.start_ms + run.from * ribbon.slot_minutes * 60_000;
    const hatch = el("div", "ribbon-hatch gap-hatch");
    hatch.dataset.reason =
      archiveEnd === null || startMs > archiveEnd ? "not archived yet" : "gap in record";
    // A span at the right edge is anchored there, so its minimum width
    // grows leftward and stays inside the band.
    if (run.to === n) hatch.style.right = "0";
    else hatch.style.left = `${pct(run.from)}%`;
    hatch.style.width = `${pct(run.to) - pct(run.from)}%`;
    band.append(hatch);
  }

  // The temperature trace, broken where there is no reading.
  const temps = ribbon.temperature_f;
  const known = temps.filter((t): t is number => t !== null).concat(now === null ? [] : [now]);
  if (known.length) {
    const lo = Math.min(...known) - 2;
    const hi = Math.max(...known) + 2;
    const y = (v: number) => +(12 + (1 - (v - lo) / (hi - lo)) * 76).toFixed(2);
    const trace = svg("svg", { viewBox: `0 0 ${VIEW_W} ${VIEW_H}`, preserveAspectRatio: "none" });
    trace.setAttribute("aria-hidden", "true");
    trace.classList.add("ribbon-trace");
    const point = (i: number) => `${x(i + 0.5)},${y(temps[i]!)}`;
    const segments = runs(temps.map((t) => t !== null)).filter((r) => r.value);
    for (const seg of segments) {
      const pts = Array.from({ length: seg.to - seg.from }, (_, k) => point(seg.from + k));
      // A lone point still needs a length to show.
      if (pts.length === 1) pts.push(pts[0]!);
      for (const cls of ["ribbon-halo", "ribbon-line"]) {
        trace.append(svg("polyline", { points: pts.join(" "), class: cls }));
      }
    }
    // The live point, alone after the unarchived span: join it dotted.
    const last = segments.at(-1);
    const before = segments.at(-2);
    if (last && before && last.to === n && last.to - last.from === 1) {
      trace.append(
        svg("polyline", {
          points: `${point(before.to - 1)} ${point(n - 1)}`,
          class: "ribbon-live",
        }),
      );
    }
    band.append(trace);
    if (now !== null) {
      const dot = el("div", "ribbon-dot");
      dot.style.top = `${y(now)}%`;
      band.append(dot);
    }
  }
  block.append(band);

  const ticks = el("div", "ribbon-ticks");
  ticks.setAttribute("aria-hidden", "true");
  for (const tick of ribbonTicks(ribbon)) {
    const node = el("span", tick.late ? "ribbon-tick ribbon-tick-late" : "ribbon-tick", tick.label);
    node.style.left = `${tick.pct}%`;
    ticks.append(node);
  }
  if (now !== null) ticks.append(el("span", "ribbon-now", `now ${now.toFixed(1)}°`));
  block.append(ticks);

  block.append(ribbonKey());
  return block;
}

function ribbonKey(): HTMLElement {
  const key = el("p", "ribbon-key");
  const swatch = (cls: string, label: string) => {
    const item = el("span", "ribbon-key-item");
    item.append(el("i", cls), label);
    return item;
  };
  const ramp = el("span", "ribbon-key-item");
  const steps = el("span", "ribbon-key-ramp");
  RAIN_STEP_NAMES.forEach((_, i) => steps.append(el("i", `ribbon-key-rain-${i + 1}`)));
  ramp.append("rain, light", steps, "heavy");
  key.append(
    swatch("ribbon-key-night", "night"),
    swatch("ribbon-key-day", "day"),
    ramp,
    swatch("gap-hatch", "no archived reading"),
  );
  return key;
}
