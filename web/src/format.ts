/**
 * The One Format Rule (DESIGN.md): times as `9:36 am`, dates with times as
 * `Oct 27, 11:00 pm`, durations as `1 d 10 h`, `5 h 12 m` or `45 m`, all in
 * station-local Pacific time.
 */

/** The station's time zone. */
export const STATION_TIME_ZONE = "America/Los_Angeles";

/** The data age turns stale past this. */
export const STALE_AFTER_MS = 15 * 60_000;

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
  if (hours < 24) return `${hours} h ${minutes % 60} m`;
  return `${Math.floor(hours / 24)} d ${hours % 24} h`;
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
