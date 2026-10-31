/**
 * The One Format Rule (DESIGN.md): times as `9:36 am`, dates with times as
 * `Oct 27, 11:00 pm`, durations as `1 d 10 h`, `5 h 12 m` or `45 m`, all in
 * station-local Pacific time, and imperial units. Every view writes its
 * times, durations, month names and units through this module.
 */

/** The station's time zone. */
export const STATION_TIME_ZONE = "America/Los_Angeles";

/** The data age turns stale past this. */
export const STALE_AFTER_MS = 15 * 60_000;

export const HOUR_MS = 3_600_000;
export const DAY_MS = 24 * HOUR_MS;

/** The calendar's months, short, January first. */
export const MONTHS: readonly string[] = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** The water year's months, short: it runs Oct through Sep. */
export const WATER_MONTHS: readonly string[] = [...MONTHS.slice(9), ...MONTHS.slice(0, 9)];

const LONG_MONTH = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "long" });

/**
 * A month's full name, e.g. `October`.
 *
 * @param month - 1 (January) to 12.
 * @returns The name.
 */
export function monthLongName(month: number): string {
  return LONG_MONTH.format(Date.UTC(2000, month - 1, 15));
}

const STAMP = new Intl.DateTimeFormat("en-US", {
  timeZone: STATION_TIME_ZONE,
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  hour12: true,
});

/**
 * A date with a time, e.g. `Jan 8, 4:45 pm`.
 *
 * @param ms - Epoch milliseconds UTC.
 * @returns The stamp in Pacific time.
 */
export function formatStamp(ms: number): string {
  const parts = Object.fromEntries(
    STAMP.formatToParts(ms).map((part) => [part.type, part.value]),
  );
  const period = (parts.dayPeriod ?? "").toLowerCase();
  return `${parts.month} ${parts.day}, ${parts.hour}:${parts.minute} ${period}`;
}

/**
 * A duration, e.g. `45 m`, `5 h 12 m`, `1 d 10 h`; `under 1 m` below a minute.
 *
 * @param ms - The duration in milliseconds.
 * @returns The duration in words.
 */
export function formatDuration(ms: number): string {
  const minutes = Math.floor(Math.max(0, ms) / 60_000);
  if (minutes < 1) return "under 1 m";
  if (minutes < 60) return `${minutes} m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return hoursAndMinutes(minutes);
  return `${Math.floor(hours / 24)} d ${hours % 24} h`;
}

/**
 * A duration in hours that never goes to days, e.g. `30 h`, `47 h 30 m`,
 * `45 m`: for a span bounded by a day or two.
 *
 * @param ms - The duration in milliseconds.
 * @param options - `noBreak` joins the words with no-break spaces, so the
 *   duration never wraps across lines.
 * @returns The duration in words.
 */
export function formatHours(ms: number, options: { noBreak?: boolean } = {}): string {
  const minutes = Math.floor(Math.max(0, ms) / 60_000);
  const text =
    minutes < 1 ? "under 1 m" : minutes < 60 ? `${minutes} m` : hoursAndMinutes(minutes);
  return options.noBreak ? text.replaceAll(" ", "\u00a0") : text;
}

/** `3 h` or `5 h 12 m` from whole minutes, an hour or more. */
function hoursAndMinutes(minutes: number): string {
  const m = minutes % 60;
  const h = Math.floor(minutes / 60);
  return m ? `${h} h ${m} m` : `${h} h`;
}

/**
 * A duration given in days, e.g. `1 d 22 h` since it last rained.
 *
 * @param days - Fractional days.
 * @returns The duration in words.
 */
export function formatDays(days: number): string {
  return formatDuration(days * DAY_MS);
}

/**
 * A count of days, as a usual dry spell reads: `1 day`, `2 days`, `1.5 days`
 * (a median can fall between two whole days).
 *
 * @param days - The count.
 * @returns The count in words.
 */
export function formatDayCount(days: number): string {
  return `${Number.isInteger(days) ? days : days.toFixed(1)} ${days === 1 ? "day" : "days"}`;
}

const HOUR = new Intl.DateTimeFormat("en-US", {
  timeZone: STATION_TIME_ZONE,
  hour: "numeric",
  hourCycle: "h23",
});

/**
 * The Pacific hour of a time.
 *
 * @param ms - Epoch milliseconds UTC.
 * @returns 0 to 23.
 */
export function localHour(ms: number): number {
  return Number(HOUR.format(ms));
}

/**
 * An hour of the day, e.g. `midnight`, `3 am`, `noon`, `9 pm`.
 *
 * @param hour - 0 to 23.
 * @returns The hour in words.
 */
export function hourText(hour: number): string {
  if (hour === 0) return "midnight";
  if (hour === 12) return "noon";
  return hour < 12 ? `${hour} am` : `${hour - 12} pm`;
}

/**
 * A time's Pacific hour in words, e.g. `noon`, `3 pm`.
 *
 * @param ms - Epoch milliseconds UTC.
 * @returns The hour in words.
 */
export function hourName(ms: number): string {
  return hourText(localHour(ms));
}

/**
 * A station-local day, e.g. `Jan 8`, or `Jan 8, 2026` with the year.
 *
 * @param iso - "YYYY-MM-DD".
 * @param withYear - Add the year.
 * @returns The day in words.
 */
export function dayText(iso: string, withYear = false): string {
  const text = `${MONTHS[Number(iso.slice(5, 7)) - 1]} ${Number(iso.slice(8, 10))}`;
  return withYear ? `${text}, ${iso.slice(0, 4)}` : text;
}

// ---------- units ----------

/**
 * Degrees Fahrenheit, e.g. `52.3°`.
 *
 * @param value - Degrees.
 * @param digits - Decimals; 0 rounds to whole degrees.
 * @returns The temperature.
 */
export function degrees(value: number, digits = 1): string {
  return `${digits ? value.toFixed(digits) : Math.round(value)}°`;
}

/** Options for `inches`. */
export interface InchOptions {
  /** Fixed decimals; by default two, and one from 10 in up. */
  digits?: number;
  /** Write " in" after the number (the default). */
  unit?: boolean;
}

/**
 * Rain in inches, e.g. `0.30 in`, `12.4 in`.
 *
 * @param value - Inches.
 * @param options - Fixed `digits`; `unit: false` for the bare number.
 * @returns The amount.
 */
export function inches(value: number, options: InchOptions = {}): string {
  const digits = options.digits ?? (value >= 10 ? 1 : 2);
  const text = value.toFixed(digits);
  return options.unit === false ? text : `${text} in`;
}

/** A rain rate, e.g. `0.12 in/hr`. */
export function inchesPerHour(value: number): string {
  return `${value.toFixed(2)} in/hr`;
}

/** Wind speed, e.g. `4.3 mph`. */
export function mph(value: number): string {
  return `${value.toFixed(1)} mph`;
}

/** Solar radiation, e.g. `412 W/m²`. */
export function wattsPerM2(value: number): string {
  return `${Math.round(value)} W/m²`;
}

/**
 * Solar energy per m²: two decimals, one from 10 up, none from 100 up.
 *
 * @param value - kWh per m².
 * @param options - `unit: false` for the bare number.
 * @returns E.g. `2.41 kWh/m²`.
 */
export function kwhPerM2(value: number, options: { unit?: boolean } = {}): string {
  const text = value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2);
  return options.unit === false ? text : `${text} kWh/m²`;
}

/** How old the newest observation is, as the header shows it. */
export interface DataAge {
  /** Milliseconds since the observation, or null with none. */
  ageMs: number | null;
  /** True past 15 minutes, or with no observation. */
  stale: boolean;
  /** The words the header shows. */
  text: string;
}

/**
 * The data age, computed in the browser from the newest observation time so
 * that it keeps counting when the snapshot itself is old.
 *
 * @param observedAtMs - The newest observation, or null.
 * @param nowMs - The browser's clock.
 * @returns The age, whether it is stale, and its words.
 */
export function dataAge(observedAtMs: number | null, nowMs: number): DataAge {
  if (observedAtMs === null) {
    return { ageMs: null, stale: true, text: "No reading yet" };
  }
  const ageMs = Math.max(0, nowMs - observedAtMs);
  const stale = ageMs > STALE_AFTER_MS;
  const duration = formatDuration(ageMs);
  return { ageMs, stale, text: stale ? `Stale: ${duration} old` : `${duration} ago` };
}
