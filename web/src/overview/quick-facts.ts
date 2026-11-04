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
 * Offers `windFact`, `pressureFact` and `airFact` (`OverviewSection`s with
 * `fact: true`, which keeps them open tiles on a phone, two to a row), and
 * `renderFact`, the value-and-sentence body any quick-fact tile renders.
 */
import "./quick-facts.css";

import { compassPoint, degrees, inHg, known, mph, percent } from "../format.js";
import type { DashboardSnapshot, SnapshotConditions } from "../snapshot.js";
import type { OverviewSection } from "./index.js";

/**
 * A quick-fact tile's body: its one large value, then its one sentence.
 *
 * @param host - The tile, holding its kicker.
 * @param value - The value ("n/a" when not known).
 * @param sentence - The sentence under it.
 * @param unit - A word unit after the value ("mph"), set smaller and kept
 *   on the value's line, so a half-width phone tile never breaks the number
 *   from its unit.
 */
export function renderFact(
  host: HTMLElement,
  value: string,
  sentence: string,
  unit?: string,
): void {
  const big = document.createElement("p");
  big.className = "qf-value";
  big.dataset.part = "fact";
  big.append(value);
  if (unit) {
    const small = document.createElement("span");
    small.className = "qf-unit";
    small.textContent = unit;
    big.append("\u00a0", small);
  }
  const line = document.createElement("p");
  line.className = "qf-sentence";
  line.dataset.part = "fact-sentence";
  line.textContent = sentence;
  host.append(big, line);
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
