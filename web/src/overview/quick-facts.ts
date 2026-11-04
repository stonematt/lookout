/**
 * The Overview's quick-fact tiles (DESIGN.md "Quick-fact tiles", #340):
 * wind, pressure and air. Each is a kicker, one large value in Fact type and
 * one sentence, read from the snapshot's live-stitched `current`, the same
 * values the now tile shows, in the same formats, so the two always agree.
 *
 * A missing value reads as "n/a" and its sentence says it is not known,
 * never zero. The pressure sentence repeats the snapshot's 3-hour trend word
 * (`barometer_trend`, decided in Python); the site never recomputes it.
 *
 * The sun and moon tile (#341) draws the snapshot's SunMoon block, every
 * value decided in Python, its times in the station's zone. A station with
 * no coordinates has no block: the tile keeps its kicker, shows no value and
 * says the station reports no location.
 *
 * Offers `windFact`, `pressureFact`, `airFact` and `sunMoonFact`
 * (`OverviewSection`s with `fact: true`, which keeps them open tiles on a
 * phone, two to a row), and `renderFact`, the value-and-sentence body any
 * quick-fact tile renders.
 */
import "./quick-facts.css";

import {
  compassPoint,
  degrees,
  formatHours,
  formatTime,
  inHg,
  known,
  mph,
  percent,
} from "../format.js";
import type { DashboardSnapshot, SnapshotConditions, SunMoon } from "../snapshot.js";
import type { OverviewSection } from "./index.js";

/**
 * A quick-fact tile's body: its one large value, then its one sentence.
 *
 * @param host - The tile, holding its kicker.
 * @param value - The value ("n/a" when not known), as text or as nodes
 *   that already carry their units, or null for a tile that has no value to
 *   show: the sentence then sits under the kicker alone.
 * @param sentence - The sentence under it.
 * @param unit - A word unit after the value ("mph"), set smaller and kept
 *   on the value's line, so a half-width phone tile never breaks the number
 *   from its unit.
 */
export function renderFact(
  host: HTMLElement,
  value: string | Node | null,
  sentence: string,
  unit?: string,
): void {
  const line = document.createElement("p");
  line.className = "qf-sentence";
  line.dataset.part = "fact-sentence";
  line.textContent = sentence;
  if (value === null) {
    host.append(line);
    return;
  }
  const big = document.createElement("p");
  big.className = "qf-value";
  big.dataset.part = "fact";
  big.append(value);
  if (unit) big.append("\u00a0", unitSpan(unit));
  host.append(big, line);
}

/**
 * A word set as a quick fact's unit: Stat type, on the value's line.
 *
 * @param text - The word, e.g. "mph" or "pm".
 * @returns The span.
 */
function unitSpan(text: string): HTMLSpanElement {
  const small = document.createElement("span");
  small.className = "qf-unit";
  small.textContent = text;
  return small;
}

/**
 * The wind sentence: where it blows from and its gust.
 *
 * @param c - The current conditions.
 * @returns E.g. "From the SE, gusts to 2.5 mph."
 */
function windSentence(c: SnapshotConditions): string {
  const from = c.wind_dir_deg === null ? null : `From the ${compassPoint(c.wind_dir_deg)}`;
  const gust = c.wind_gust_mph === null ? null : `gusts to ${mph(c.wind_gust_mph)}`;
  let rest: string;
  if (from && gust) rest = `${from}, ${gust}.`;
  else if (from) rest = `${from}; gust not known.`;
  else if (gust) rest = `Direction not known; ${gust}.`;
  else rest = "Direction and gust not known.";
  return c.wind_speed_mph === null ? `Wind speed not known. ${rest}` : rest;
}

/**
 * The pressure sentence: the unit, then the snapshot's 3-hour trend word.
 *
 * @param c - The current conditions.
 * @returns E.g. "inHg, rising over the last 3 hours."
 */
function pressureSentence(c: SnapshotConditions): string {
  const trend = c.barometer_trend;
  if (c.barometer_inhg === null) {
    return trend === null
      ? "Reading and 3-hour trend not known."
      : `Reading not known; ${trend} over the last 3 hours.`;
  }
  return trend === null
    ? "inHg; the 3-hour trend is not known."
    : `inHg, ${trend} over the last 3 hours.`;
}

/**
 * The air sentence: what the value is, then the dew point.
 *
 * @param c - The current conditions.
 * @returns E.g. "Humidity; dew point 39.1°."
 */
function airSentence(c: SnapshotConditions): string {
  const dew = c.dew_point_f === null ? "not known" : degrees(c.dew_point_f);
  if (c.humidity_pct === null) {
    return c.dew_point_f === null
      ? "Humidity and dew point not known."
      : `Humidity not known; dew point ${dew}.`;
  }
  return `Humidity; dew point ${dew}.`;
}

/**
 * Sunrise to sunset as a quick fact's value: each time's am or pm, and the
 * "to" between them, set as units. "to" holds to the sunset time, so a
 * narrow tile breaks the value once, between the two times, and never
 * inside one.
 *
 * @param block - The snapshot's sun and moon.
 * @returns The value's nodes, e.g. "7:15 am to 4:38 pm" in station time.
 */
function sunTimes(block: SunMoon): DocumentFragment {
  const value = document.createDocumentFragment();
  const time = (ms: number): void => {
    const [clock, period] = formatTime(ms).split(" ");
    value.append(clock!, "\u00a0", unitSpan(period!));
  };
  time(block.sunrise_ms);
  value.append(" ", unitSpan("to"), "\u00a0");
  time(block.sunset_ms);
  return value;
}

/**
 * The daylight clause: its length, then its change from yesterday.
 *
 * @param block - The snapshot's sun and moon.
 * @returns E.g. "9 h 23 m of daylight, 2 m less than yesterday".
 */
function daylightClause(block: SunMoon): string {
  const length = formatHours(block.daylight_minutes * 60_000);
  const change = block.daylight_change_minutes;
  if (change === 0) return `${length} of daylight, the same as yesterday`;
  const by = formatHours(Math.abs(change) * 60_000);
  return `${length} of daylight, ${by} ${change > 0 ? "more" : "less"} than yesterday`;
}

/**
 * The moon clause: its phase, its lit share and the days to full.
 *
 * @param block - The snapshot's sun and moon.
 * @returns E.g. "waning gibbous moon, 68% lit, full in 24 days".
 */
function moonClause(block: SunMoon): string {
  const phase = block.moon_phase;
  const lit = `${phase} moon, ${percent(block.moon_illumination_pct)} lit`;
  const days = block.days_to_full_moon;
  if (phase === "full") return lit;
  if (days === 0) return `${lit}, full today`;
  return `${lit}, full in ${days} ${days === 1 ? "day" : "days"}`;
}

/** Wind: speed, compass direction and gust. */
export const windFact: OverviewSection = {
  id: "wind",
  label: "Wind",
  kicker: "Wind",
  fact: true,
  render(host: HTMLElement, snapshot: DashboardSnapshot): void {
    const c = snapshot.current;
    const speed = c.wind_speed_mph;
    if (speed === null) renderFact(host, "n/a", windSentence(c));
    else renderFact(host, mph(speed, { unit: false }), windSentence(c), "mph");
  },
};

/** Pressure: the barometer reading and its 3-hour trend. */
export const pressureFact: OverviewSection = {
  id: "pressure",
  label: "Pressure",
  kicker: "Pressure",
  fact: true,
  render(host: HTMLElement, snapshot: DashboardSnapshot): void {
    const c = snapshot.current;
    renderFact(host, known(c.barometer_inhg, inHg), pressureSentence(c));
  },
};

/** Air: humidity and dew point. */
export const airFact: OverviewSection = {
  id: "air",
  label: "Air",
  kicker: "Air",
  fact: true,
  render(host: HTMLElement, snapshot: DashboardSnapshot): void {
    const c = snapshot.current;
    renderFact(host, known(c.humidity_pct, percent), airSentence(c));
  },
};

/** Sun and moon: sunrise to sunset, daylight and its change, the moon. */
export const sunMoonFact: OverviewSection = {
  id: "sun-moon",
  label: "Sun and moon",
  kicker: "Sun and moon",
  fact: true,
  render(host: HTMLElement, snapshot: DashboardSnapshot): void {
    const block = snapshot.sun_moon;
    if (block === null) {
      renderFact(host, null, "Not known: the station reports no location.");
      return;
    }
    renderFact(host, sunTimes(block), `${daylightClause(block)}; ${moonClause(block)}.`);
  },
};
