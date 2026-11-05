/**
 * The 48-hour condition ribbon (DESIGN.md, "48-hour condition ribbon"): a
 * band of 15-minute slots, each night or day, with the temperature traced
 * over it and a dot at the right edge for the live value. Under the band,
 * its own track of rain bars: one rain colour, four steps told apart by
 * height, never overprinted on the band (no blue passes 3:1 against both
 * night and day and still tells light rain from heavy). Slots the archive
 * cannot speak for (a gap, or the span only the live reading covers) are
 * hatched in the band and the track; the live point is joined to the
 * archived trace by a dotted line across that span.
 *
 * The band and the track are drawn in a 1000 x 100 viewBox stretched to
 * their width, so they need no measuring; ticks, hatches and the dot are
 * positioned in %.
 */
import { HOUR_MS, stationFormat, degrees, formatHours, hourText, localHour } from "../format.js";
import { make, svgEl } from "../dom.js";
import { hatchSpan } from "../gaps.js";
import { NARROW } from "../charts/svg.js";
import type { ConditionRibbon, DashboardSnapshot } from "../snapshot.js";

const VIEW_W = 1000;
const VIEW_H = 100;

/** Rain-rate ceilings (in/hr) of the first three steps; above is step 4. */
const RAIN_STEPS = [0.05, 0.1, 0.3] as const;
const RAIN_STEP_NAMES = ["light", "moderate", "steady", "heavy"] as const;

/** A rain bar's height in the track's viewBox: a quarter more per step. */
const rainBarHeight = (step: 1 | 2 | 3 | 4) => (step / RAIN_STEP_NAMES.length) * VIEW_H;

/** Ticks are dropped this close to either edge. */
const TICK_MIN_PCT = 4;
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

/** Hours as `9 h`, `47 h 30 m` or `45 m`, on one line (the span is 48 h). */
const hoursText = (hours: number) =>
  formatHours(Math.round(hours * 60) * 60_000, { noBreak: true });

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
      : `${degrees(lo, 0)} to ${degrees(hi, 0)} over the last 48 hours`;
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
  /** A midnight, named by its weekday: the only ticks a phone-width ribbon shows. */
  day: boolean;
}

/** The weekday formatter in the station's zone. */
const weekdayOf = () => stationFormat("en-US", { weekday: "short" });

/**
 * Ticks every six hours of station time, clear of the edges.
 *
 * @param ribbon - The snapshot's ribbon.
 * @returns The ticks, left to right.
 */
export function ribbonTicks(ribbon: ConditionRibbon): RibbonTick[] {
  const span = spanMs(ribbon);
  const ticks: RibbonTick[] = [];
  for (
    let t = Math.ceil(ribbon.start_ms / HOUR_MS) * HOUR_MS;
    t < ribbon.start_ms + span;
    t += HOUR_MS
  ) {
    const hour = localHour(t);
    if (hour % 6) continue;
    const pct = ((t - ribbon.start_ms) / span) * 100;
    if (pct < TICK_MIN_PCT || pct > TICK_MAX_PCT) continue;
    const label = hour === 0 ? weekdayOf().format(t) : hourText(hour);
    ticks.push({ pct, label, day: hour === 0 });
  }
  return ticks;
}

/**
 * Give the ribbon its layout for its width (DESIGN.md, "The Chart Width
 * Rule"): under 520px its ticks name only the days; wider, every six hours.
 *
 * @param block - The ribbon block.
 * @param width - The band's width in px.
 */
export function fitRibbon(block: HTMLElement, width: number): void {
  block.dataset.narrow = String(width > 0 && width < NARROW);
}

/** The room a tick label keeps from the now label. */
const NOW_LABEL_GAP_PX = 6;

/**
 * Hide every tick label that would run into the "now" label (or come
 * within 6px of it), and show the rest again. Ticks are placed by the time
 * of day across a band of any width, so this reads the laid-out row; a row
 * not laid out yet (a hidden tab) is left alone.
 *
 * @param row - The ribbon's `.ribbon-ticks` row.
 */
export function clearNowLabel(row: HTMLElement): void {
  const now = row.querySelector<HTMLElement>(".ribbon-now")?.getBoundingClientRect();
  if (!now || now.width === 0) return;
  for (const tick of row.querySelectorAll<HTMLElement>(".ribbon-tick")) {
    tick.classList.remove("ribbon-tick-hidden");
    const box = tick.getBoundingClientRect();
    const clear =
      box.width === 0 ||
      box.right + NOW_LABEL_GAP_PX <= now.left ||
      box.left >= now.right + NOW_LABEL_GAP_PX;
    tick.classList.toggle("ribbon-tick-hidden", !clear);
  }
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
  // A run of slots as one rect standing on the floor (full height by
  // default); 0.4 overlaps the next, no seam.
  const runRect = (run: Run<unknown>, height: number = VIEW_H) =>
    svgEl("rect", {
      x: x(run.from),
      y: VIEW_H - height,
      width: x(run.to) - x(run.from) + 0.4,
      height,
    });
  const now = snapshot.current.temperature_f ?? ribbon.temperature_f.at(-1) ?? null;

  const block = make("div", "ribbon");
  block.dataset.part = "ribbon";
  const claim = ribbonClaim(ribbon);
  block.append(make("p", "ribbon-claim", claim));

  const band = make("div", "ribbon-band");
  band.setAttribute("role", "img");
  band.setAttribute(
    "aria-label",
    `48-hour condition ribbon: ${claim.replace(/^The last 48 hours, /, "")} ` +
      "Day and night, rain intensity and the temperature trace, oldest on the left.",
  );

  // Night and day; rain has its own track under the band.
  const fills = svgEl("svg", { viewBox: `0 0 ${VIEW_W} ${VIEW_H}`, preserveAspectRatio: "none" });
  fills.setAttribute("aria-hidden", "true");
  for (const run of runs(ribbon.daylight.map((d) => (d ? "day" : "night")))) {
    const rect = runRect(run);
    rect.dataset.light = run.value;
    fills.append(rect);
  }
  band.append(fills);

  // The rain track: a bar per run of one step, standing on the track's floor.
  const track = make("div", "ribbon-rain");
  track.setAttribute("aria-hidden", "true");
  const bars = svgEl("svg", { viewBox: `0 0 ${VIEW_W} ${VIEW_H}`, preserveAspectRatio: "none" });
  for (const run of runs(ribbon.rain_in.map((r) => rainStep(r, ribbon.slot_minutes)))) {
    if (run.value === null) continue;
    const rect = runRect(run, rainBarHeight(run.value));
    rect.dataset.rain = String(run.value);
    // how many slots the bar covers, so a reader of the markup can count them
    rect.dataset.slots = String(run.to - run.from);
    bars.append(rect);
  }
  track.append(bars);

  // Hatch what the archive cannot speak for, in the band (with its reason)
  // and in the track under it, where no bar would otherwise read as dry.
  const archiveEnd = snapshot.archive_newest_ms;
  const place = (hatch: HTMLElement, run: Run<unknown>) => {
    // A span at the right edge is anchored there, so its minimum width
    // grows leftward and stays inside the band.
    if (run.to === n) hatch.style.right = "0";
    else hatch.style.left = `${pct(run.from)}%`;
    hatch.style.width = `${pct(run.to) - pct(run.from)}%`;
  };
  for (const run of runs(ribbon.rain_in.map((r) => r === null))) {
    if (!run.value) continue;
    const startMs = ribbon.start_ms + run.from * ribbon.slot_minutes * 60_000;
    const hatch = hatchSpan(
      archiveEnd === null || startMs > archiveEnd ? "not archived yet" : "gap in record",
    );
    hatch.classList.add("ribbon-hatch");
    place(hatch, run);
    band.append(hatch);
    const trackHatch = make("div", "gap-hatch ribbon-rain-hatch");
    place(trackHatch, run);
    track.append(trackHatch);
  }

  // The temperature trace, broken where there is no reading.
  const temps = ribbon.temperature_f;
  const known = temps.filter((t): t is number => t !== null).concat(now === null ? [] : [now]);
  if (known.length) {
    const lo = Math.min(...known) - 2;
    const hi = Math.max(...known) + 2;
    const y = (v: number) => +(12 + (1 - (v - lo) / (hi - lo)) * 76).toFixed(2);
    const trace = svgEl("svg", { viewBox: `0 0 ${VIEW_W} ${VIEW_H}`, preserveAspectRatio: "none" });
    trace.setAttribute("aria-hidden", "true");
    trace.classList.add("ribbon-trace");
    const point = (i: number) => `${x(i + 0.5)},${y(temps[i]!)}`;
    const segments = runs(temps.map((t) => t !== null)).filter((r) => r.value);
    for (const seg of segments) {
      const pts = Array.from({ length: seg.to - seg.from }, (_, k) => point(seg.from + k));
      // A lone point still needs a length to show.
      if (pts.length === 1) pts.push(pts[0]!);
      for (const cls of ["ribbon-halo", "ribbon-line"]) {
        trace.append(svgEl("polyline", { points: pts.join(" "), class: cls }));
      }
    }
    // The live point, alone after the unarchived span: join it dotted.
    const last = segments.at(-1);
    const before = segments.at(-2);
    if (last && before && last.to === n && last.to - last.from === 1) {
      trace.append(
        svgEl("polyline", {
          points: `${point(before.to - 1)} ${point(n - 1)}`,
          class: "ribbon-live",
        }),
      );
    }
    band.append(trace);
    if (now !== null) {
      const dot = make("div", "ribbon-dot");
      dot.style.top = `${y(now)}%`;
      band.append(dot);
    }
  }
  block.append(band, track);

  const ticks = make("div", "ribbon-ticks");
  ticks.setAttribute("aria-hidden", "true");
  for (const tick of ribbonTicks(ribbon)) {
    const node = make("span", tick.day ? "ribbon-tick ribbon-tick-day" : "ribbon-tick", tick.label);
    node.style.left = `${tick.pct}%`;
    ticks.append(node);
  }
  if (now !== null) ticks.append(make("span", "ribbon-now", `now ${degrees(now)}`));
  block.append(ticks);
  // Which ticks show depends on the band's width, and where a tick falls on
  // the time of day, so only the laid-out row can say which run into the
  // now label.
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      fitRibbon(block, ticks.clientWidth);
      if (now !== null) clearNowLabel(ticks);
    }).observe(ticks);
    // The labels change width once the web font arrives; the row does not.
    void document.fonts?.ready.then(() => clearNowLabel(ticks));
  }

  block.append(ribbonKey());
  return block;
}

function ribbonKey(): HTMLElement {
  const key = make("p", "ribbon-key");
  const swatch = (cls: string, label: string) => {
    const item = make("span", "ribbon-key-item");
    item.append(make("i", cls), label);
    return item;
  };
  const ramp = make("span", "ribbon-key-item");
  const steps = make("span", "ribbon-key-ramp");
  // each step as tall in the key as its bar in the track, by the same rule
  for (const step of [1, 2, 3, 4] as const) {
    const bar = make("i", `ribbon-key-rain-${step}`);
    bar.style.height = `${(rainBarHeight(step) / VIEW_H) * 100}%`;
    steps.append(bar);
  }
  ramp.append("rain, light ", steps, " heavy");
  key.append(
    swatch("ribbon-key-night", "night"),
    swatch("ribbon-key-day", "day"),
    ramp,
    swatch("gap-hatch", "no archived reading"),
  );
  return key;
}
