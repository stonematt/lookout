/**
 * The almanac grid's and the rolling-window table's words and scales: pure
 * functions of the snapshot's `almanac` (built by
 * `lookout/snapshot/almanac.py`), no DOM, so each is tested against the
 * golden snapshots and worked examples.
 *
 * DESIGN.md rules held here: the Finding First Rule (a headline states what
 * the data shows, with the number in it), the Name the Years Rule (years
 * compared are named, newest first), the One Gap Status Rule (an invalid
 * month's or window's value is never shown, a prorated one carries ≈), and
 * the almanac's square-root rain ramp.
 */
import { gapSentence, gapValue } from "../gaps.js";
import { NAMED_YEARS, WINDOW_NAMES, dayText, inches, listText } from "../rain/text.js";
import type {
  Almanac,
  AlmanacMonth,
  AlmanacYear,
  GapStatus,
  RollingWindow,
  RollingYear,
} from "../snapshot.js";

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/** The grid's columns: a water year runs Oct through Sep. */
export const WATER_MONTHS = [
  "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
];

/** How many steps the rain ramp has: `rain-0` to `rain-4`. */
export const RAMP_STEPS = 5;

/** The rolling windows' names, as the table's rows read them. */
export const ROLLING_NAMES: Record<string, string> = {
  ...WINDOW_NAMES,
  water_year: "Since Oct 1",
};

/** What a cell shows: one of the grid's six kinds of month. */
export type CellKind = "valid" | "prorated" | "invalid" | "before" | "current" | "future";

/**
 * Which kind of cell a month is.
 *
 * @param month - The month, or null for one not begun.
 * @returns Its kind: "current" while it runs, "before" when it starts
 *   before the archive, else its gap state.
 */
export function cellKind(month: AlmanacMonth | null): CellKind {
  if (!month) return "future";
  if (month.to_date) return "current";
  if (month.status.reason === "before_archive") return "before";
  return month.status.status;
}

/**
 * "October 2025" from a month key.
 *
 * @param key - "YYYY-MM".
 * @returns The month in words.
 */
export function monthName(key: string): string {
  return `${MONTH_NAMES[Number(key.slice(5, 7)) - 1]} ${key.slice(0, 4)}`;
}

/**
 * Inches as a 44px cell has room for: "0.04", "4.2", "12".
 *
 * @param value - Inches.
 * @returns At most four characters.
 */
export function cellText(value: number): string {
  if (value < 1) return value.toFixed(2);
  if (value < 10) return value.toFixed(1);
  return value.toFixed(0);
}

/**
 * A month's place on the rain ramp, on a square-root scale so a dry month
 * still differs from a drier one.
 *
 * @param value - The month's rain.
 * @param max - The wettest comparable month in the grid.
 * @returns 0 (`rain-0`) to 4 (`rain-4`).
 */
export function rampStep(value: number, max: number): number {
  if (max <= 0 || value <= 0) return 0;
  return Math.min(RAMP_STEPS - 1, Math.floor(Math.sqrt(value / max) * RAMP_STEPS));
}

/**
 * Where each ramp step starts, for the key.
 *
 * @param max - The wettest comparable month in the grid.
 * @returns Five lower bounds in inches, the first 0.
 */
export function rampBounds(max: number): number[] {
  return Array.from({ length: RAMP_STEPS }, (_, i) => max * (i / RAMP_STEPS) ** 2);
}

/**
 * The wettest comparable month in the grid: the ramp's top.
 *
 * @param almanac - The snapshot's almanac.
 * @returns Inches; 0 when no month can be compared.
 */
export function wettestMonth(almanac: Almanac): number {
  let max = 0;
  for (const year of almanac.years) {
    for (const month of year.months) {
      if (month?.rain_in != null) max = Math.max(max, month.rain_in);
    }
  }
  return max;
}

/**
 * A month's full sentence, for the live readout and screen readers.
 *
 * @param month - A month that has begun.
 * @returns E.g. "December 2024: 1.00 in of rain. Dec 1 to Dec 31, 2024:
 *   every day is in the record."
 */
export function monthSentence(month: AlmanacMonth): string {
  const name = monthName(month.status.key);
  if (month.to_date) {
    const span = month.to_date;
    const value = month.to_date_in === null ? null : gapValue(span, inches(month.to_date_in));
    const soFar = value
      ? `${name} so far: ${value} of rain through ${dayText(span.last_day)}.`
      : `${name} so far: ${gapSentence(span)}`;
    return `${soFar} ${gapSentence(month.status)}`;
  }
  const value = month.rain_in === null ? null : gapValue(month.status, inches(month.rain_in));
  return value ? `${name}: ${value} of rain. ${gapSentence(month.status)}` : `${name}: ${gapSentence(month.status)}`;
}

/**
 * A water year's season total in a sentence.
 *
 * @param year - The row.
 * @param soFar - The rolling table's season so far, for the running year.
 * @returns E.g. "Water year 2025, Oct 1, 2024 to Sep 30, 2025: 1.95 in of
 *   rain."
 */
export function seasonSentence(year: AlmanacYear, soFar: RollingYear | null): string {
  const name = `Water year ${year.water_year}`;
  if (year.season.reason === "in_progress" && soFar) {
    const value = soFar.rain_in === null ? null : gapValue(soFar.status, inches(soFar.rain_in));
    return value
      ? `${name} so far: ${value} of rain since Oct 1.`
      : `${name} so far: not compared. ${gapSentence(soFar.status)}`;
  }
  const value = year.season_in === null ? null : gapValue(year.season, inches(year.season_in));
  return value
    ? `${name}, Oct 1 to Sep 30: ${value} of rain.`
    : `${name}: not compared. ${gapSentence(year.season)}`;
}

/**
 * The rolling table's season-so-far window ending today, if it has one.
 *
 * @param almanac - The snapshot's almanac.
 * @returns This year's "water_year" entry.
 */
export function seasonSoFar(almanac: Almanac): RollingYear | null {
  return almanac.rolling.find((w) => w.key === "water_year")?.years[0] ?? null;
}

/**
 * The grid's finding: the newest month that can be compared, against the
 * same month in earlier years, newest first.
 *
 * @param almanac - The snapshot's almanac.
 * @returns E.g. "September 2025 had 0.70 in of rain, against 0.50 in
 *   (2024)".
 */
export function gridHeadline(almanac: Almanac): string {
  const fallback = "Rain by month, each water year from Oct 1";
  for (const [row, year] of almanac.years.entries()) {
    for (let column = 11; column >= 0; column--) {
      const month = year.months[column];
      if (!month || month.rain_in === null) continue;
      const amount = month.rain_in === 0 ? "no rain" : `${shown(month.status, month.rain_in)} of rain`;
      const lead = `${monthName(month.status.key)} had ${amount}`;
      const earlier = almanac.years
        .slice(row + 1)
        .map((other) => other.months[column])
        .filter((m): m is AlmanacMonth => m != null && m.rain_in !== null)
        .slice(0, NAMED_YEARS)
        .map((m) => `${m.rain_in === 0 ? "no rain" : shown(m.status, m.rain_in!)} (${m.status.key.slice(0, 4)})`);
      return earlier.length ? `${lead}, against ${listText(earlier)}` : lead;
    }
  }
  return fallback;
}

/**
 * The rolling table's finding: the last 30 days (else 7, else the season)
 * against the same days in earlier years, newest first.
 *
 * @param almanac - The snapshot's almanac.
 * @returns E.g. "1.00 in of rain in the last 30 days, against 0.75 in
 *   (2024) over the same days".
 */
export function rollingHeadline(almanac: Almanac): string {
  for (const key of ["30d", "7d", "water_year"]) {
    const window = almanac.rolling.find((w) => w.key === key);
    const now = window?.years[0];
    if (!window || !now || now.rain_in === null) continue;
    const span = key === "water_year" ? "since Oct 1" : `in the last ${ROLLING_NAMES[key]}`;
    const lead = `${shown(now.status, now.rain_in)} of rain ${span}`;
    const earlier = shownYears(window).slice(1, NAMED_YEARS + 1);
    const named = earlier.map((y) => `${y.text} (${y.year})`);
    return named.length ? `${lead}, against ${listText(named)} over the same days` : lead;
  }
  return "Rain over the last week, month and season";
}

/**
 * A table cell's number: inches without the unit, ≈ when prorated.
 *
 * @param status - The window's or season's gap status.
 * @param rain - Its rain, or null.
 * @returns E.g. "0.30", "≈ 1.95"; null when it cannot be compared.
 */
export function tableValue(status: GapStatus, rain: number | null): string | null {
  if (rain === null) return null;
  return gapValue(status, inches(rain).replace(/ in$/, ""));
}

/**
 * The years a table row shows: this year's window and the earlier years
 * with a value, newest first.
 */
function shownYears(window: RollingWindow): { year: number; text: string }[] {
  return window.years
    .filter((y) => y.rain_in !== null)
    .map((y) => ({ year: y.year, text: shown(y.status, y.rain_in!) }));
}

/**
 * A value as its gap status shows it, for a headline.
 *
 * @param status - The month's or window's gap status.
 * @param value - Its rain.
 * @returns "0.70 in" or "≈ 0.70 in".
 */
function shown(status: GapStatus, value: number): string {
  return gapValue(status, inches(value)) ?? inches(value);
}

