/**
 * The snapshot contract, TypeScript side.
 *
 * The Python refresh run publishes two documents per station on Storj
 * (`lookout/snapshot/model.py` holds the rules; `tests/golden/snapshots/`
 * holds the goldens both sides test against):
 *
 * - `snapshot/<MAC>.dashboard.json`, served by `/dashboard`.
 * - `snapshot/<MAC>.context.json`, served by `/current` next to the live
 *   Ambient `lastData`.
 *
 * Shape rules: every document starts with `schema_version` and `kind`; times
 * are integer epoch milliseconds UTC in fields ending `_ms`; units are named
 * in the field; an unknown value is `null`, never missing.
 *
 * This module is shared by the browser and the read functions, so it uses no
 * DOM and no Node APIs.
 */

/** The schema version this site reads. A breaking change bumps both sides. */
export const SCHEMA_VERSION = 1;

/** The station a snapshot describes. */
export interface Station {
  /** Its MAC address, the key of every file it has on Storj. */
  mac: string;
  /** Its human-readable name in Ambient, or null when unnamed. */
  name: string | null;
}

/** "rising", "falling" or "steady". */
export type Trend = "rising" | "falling" | "steady";

/** The newest reading, live where the live reading reports it. */
export interface SnapshotConditions {
  temperature_f: number | null;
  feels_like_f: number | null;
  dew_point_f: number | null;
  humidity_pct: number | null;
  barometer_inhg: number | null;
  wind_speed_mph: number | null;
  wind_gust_mph: number | null;
  wind_dir_deg: number | null;
  rain_rate_in_hr: number | null;
  rain_today_in: number | null;
  rain_event_in: number | null;
  solar_wm2: number | null;
  uv_index: number | null;
  last_rain_ms: number | null;
  temp_trend: Trend | null;
  barometer_trend: Trend | null;
}

/** The rain event still under way. */
export interface ActiveRainEvent {
  event_id: string;
  start_ms: number;
  duration_minutes: number;
  total_in: number;
  avg_rate_in_hr: number | null;
}

/** A month's or window's gap status: valid, prorated (≈) or invalid. */
export type GapState = "valid" | "prorated" | "invalid";

/** Why a month or window is not valid. */
export type GapReason =
  | "gap"
  | "low_coverage"
  | "not_archived"
  | "before_archive"
  | "in_progress";

/**
 * One month's or one window's gap status (`lookout/snapshot/gaps.py` holds
 * the rules). An invalid one has no value anywhere in the snapshot.
 */
export interface GapStatus {
  /** "YYYY-MM" for a month, else the window's name ("7d", "water_year"). */
  key: string;
  /** Its first station-local day, "YYYY-MM-DD". */
  first_day: string;
  /** Its last station-local day, inclusive. */
  last_day: string;
  days: number;
  /** Days that cannot be counted, whatever the reason. */
  missing_days: number;
  status: GapState;
  /** Null when valid; for a prorated one, why days are missing. */
  reason: GapReason | null;
}

/** The gap status of every archive month and every standard window. */
export interface Gaps {
  /** Oldest first, from the archive's first month through the current one. */
  months: GapStatus[];
  /** today, yesterday, 7d, 30d, 90d, 365d, water_year, in that order. */
  windows: GapStatus[];
}

/** The fields both documents share. */
interface SnapshotHead {
  schema_version: number;
  station: Station;
  /** When the refresh run built the snapshot. */
  built_at_ms: number;
  /** The newest observation, live or archived, whichever is newer. */
  observed_at_ms: number | null;
  current: SnapshotConditions;
  active_rain_event: ActiveRainEvent | null;
}

/** `snapshot/<MAC>.dashboard.json`: every field of the snapshot. */
export interface DashboardSnapshot extends SnapshotHead {
  kind: "dashboard";
  /** The archive's newest row, so the site can name the archive lag. */
  archive_newest_ms: number | null;
  /** The gap status of each month and standard window. */
  gaps: Gaps;
}

/** `snapshot/<MAC>.context.json`: what `/current` serves with `lastData`. */
export interface ContextBlock extends SnapshotHead {
  kind: "context";
}

/**
 * Ambient's live `lastData` for one device, passed through as Ambient sends
 * it. Only `dateutc` (epoch ms UTC) is relied on here.
 */
export interface LastData {
  dateutc: number;
  [field: string]: unknown;
}

/** The body `/current` returns. */
export interface CurrentReading {
  schema_version: number;
  kind: "current";
  station: Station;
  /** The newest observation: the live reading, or the context's if newer. */
  observed_at_ms: number | null;
  /** When the snapshot behind `context` was built, or null without one. */
  built_at_ms: number | null;
  /** Ambient's live reading, or null when Ambient could not be read. */
  last_data: LastData | null;
  /** The context block, or null when Storj could not be read. */
  context: ContextBlock | null;
}

/** Thrown when a document does not match the contract this site reads. */
export class SnapshotContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SnapshotContractError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isTime(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

function checkHead(doc: unknown, kind: string): Record<string, unknown> {
  if (!isRecord(doc)) {
    throw new SnapshotContractError(`${kind}: not a JSON object`);
  }
  if (doc.schema_version !== SCHEMA_VERSION) {
    throw new SnapshotContractError(
      `${kind}: schema_version ${String(doc.schema_version)}, ` +
        `this site reads ${SCHEMA_VERSION}`,
    );
  }
  if (doc.kind !== kind) {
    throw new SnapshotContractError(
      `expected kind "${kind}", got ${JSON.stringify(doc.kind)}`,
    );
  }
  const station = doc.station;
  if (!isRecord(station) || typeof station.mac !== "string") {
    throw new SnapshotContractError(`${kind}: station.mac missing`);
  }
  if (station.name !== null && typeof station.name !== "string") {
    throw new SnapshotContractError(`${kind}: station.name not a string`);
  }
  if (!isTime(doc.built_at_ms)) {
    throw new SnapshotContractError(`${kind}: built_at_ms not an integer`);
  }
  if (doc.observed_at_ms !== null && !isTime(doc.observed_at_ms)) {
    throw new SnapshotContractError(`${kind}: observed_at_ms not an integer`);
  }
  if (!isRecord(doc.current)) {
    throw new SnapshotContractError(`${kind}: current missing`);
  }
  if (doc.active_rain_event !== null && !isRecord(doc.active_rain_event)) {
    throw new SnapshotContractError(`${kind}: active_rain_event malformed`);
  }
  return doc;
}

/**
 * Check a parsed `dashboard.json` against the contract and type it.
 *
 * Checks the head (version, kind, station, times) rather than every field,
 * so an added field never breaks the site.
 *
 * @param doc - The parsed JSON.
 * @returns The same object, typed.
 * @throws SnapshotContractError when the version, kind or head is wrong.
 */
export function parseDashboard(doc: unknown): DashboardSnapshot {
  const head = checkHead(doc, "dashboard");
  if (head.archive_newest_ms !== null && !isTime(head.archive_newest_ms)) {
    throw new SnapshotContractError(
      "dashboard: archive_newest_ms not an integer",
    );
  }
  return head as unknown as DashboardSnapshot;
}

/**
 * Check a parsed `context.json` against the contract and type it.
 *
 * @param doc - The parsed JSON.
 * @returns The same object, typed.
 * @throws SnapshotContractError when the version, kind or head is wrong.
 */
export function parseContext(doc: unknown): ContextBlock {
  return checkHead(doc, "context") as unknown as ContextBlock;
}

/**
 * Check a parsed `/current` body and type it.
 *
 * @param doc - The parsed JSON.
 * @returns The same object, typed.
 * @throws SnapshotContractError when the version, kind or station is wrong.
 */
export function parseCurrent(doc: unknown): CurrentReading {
  if (!isRecord(doc) || doc.kind !== "current") {
    throw new SnapshotContractError('expected kind "current"');
  }
  if (doc.schema_version !== SCHEMA_VERSION) {
    throw new SnapshotContractError(
      `current: schema_version ${String(doc.schema_version)}, ` +
        `this site reads ${SCHEMA_VERSION}`,
    );
  }
  if (!isRecord(doc.station) || typeof doc.station.mac !== "string") {
    throw new SnapshotContractError("current: station.mac missing");
  }
  if (doc.observed_at_ms !== null && !isTime(doc.observed_at_ms)) {
    throw new SnapshotContractError("current: observed_at_ms not an integer");
  }
  if (doc.context !== null) parseContext(doc.context);
  return doc as unknown as CurrentReading;
}

/**
 * The newer of two observation times; null only when both are.
 *
 * @param a - One observation time, or null.
 * @param b - Another, or null.
 * @returns The later one.
 */
export function newestObservation(
  a: number | null | undefined,
  b: number | null | undefined,
): number | null {
  const times = [a, b].filter((t): t is number => typeof t === "number");
  return times.length ? Math.max(...times) : null;
}
