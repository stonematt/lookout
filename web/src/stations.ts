/**
 * The station index (#272): the list of stations the site can show, as the
 * refresh run publishes it at `snapshot/stations.json`
 * (`lookout/snapshot/station_index.py`) and `/stations` serves it.
 *
 * It is its own document with its own `schema_version`, outside the
 * dashboard contract: `SCHEMA_VERSION` in `snapshot.ts` does not cover it. A
 * shape change bumps `STATION_INDEX_SCHEMA_VERSION` here and in Python
 * together.
 *
 * Stations are in station inventory order; the first is the default
 * station, the one every read function serves when no `?station=` is given.
 */
import { SnapshotContractError } from "./snapshot.js";

/** The station index's own contract version. */
export const STATION_INDEX_SCHEMA_VERSION = 1;

/** One station the site can show. */
export interface StationEntry {
  /** Its MAC address, the key of its snapshot documents. */
  mac: string;
  /** Its Ambient name; null when unnamed. */
  name: string | null;
  /** Its IANA time zone. */
  tz: string;
  /** Its snapshot's newest reading, epoch ms; null when unknown. */
  newest_reading_ms: number | null;
  /** Its archive's first reading, epoch ms; null when unknown. */
  archive_start_ms: number | null;
}

/** The station index document. */
export interface StationIndex {
  schema_version: typeof STATION_INDEX_SCHEMA_VERSION;
  kind: "station_index";
  /** When the refresh run published it, epoch ms. */
  built_at_ms: number;
  /** In inventory order; never empty. The first is the default station. */
  stations: StationEntry[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isTimeOrNull(value: unknown): boolean {
  return value === null || (typeof value === "number" && Number.isInteger(value));
}

function assertEntry(entry: unknown): void {
  if (
    !isRecord(entry) ||
    typeof entry.mac !== "string" ||
    entry.mac === "" ||
    typeof entry.tz !== "string" ||
    !(entry.name === null || typeof entry.name === "string") ||
    !isTimeOrNull(entry.newest_reading_ms) ||
    !isTimeOrNull(entry.archive_start_ms)
  ) {
    throw new SnapshotContractError("station_index: a station entry is malformed");
  }
}

/**
 * Check a station index document and type it.
 *
 * @param doc - The parsed JSON.
 * @returns The typed index.
 * @throws SnapshotContractError for another schema version or kind, an
 *   empty or malformed station list, or a station listed twice.
 */
export function parseStationIndex(doc: unknown): StationIndex {
  if (!isRecord(doc)) throw new SnapshotContractError("station_index: not a JSON object");
  if (doc.schema_version !== STATION_INDEX_SCHEMA_VERSION) {
    throw new SnapshotContractError(
      `station_index: schema_version ${String(doc.schema_version)}, ` +
        `this site reads ${STATION_INDEX_SCHEMA_VERSION}`,
    );
  }
  if (doc.kind !== "station_index") {
    throw new SnapshotContractError(`station_index: kind ${String(doc.kind)}`);
  }
  if (typeof doc.built_at_ms !== "number" || !Number.isInteger(doc.built_at_ms)) {
    throw new SnapshotContractError("station_index: built_at_ms missing");
  }
  if (!Array.isArray(doc.stations) || doc.stations.length === 0) {
    throw new SnapshotContractError("station_index: no stations");
  }
  doc.stations.forEach(assertEntry);
  const macs = (doc.stations as StationEntry[]).map((s) => s.mac.toUpperCase());
  if (new Set(macs).size !== macs.length) {
    throw new SnapshotContractError("station_index: a station is listed twice");
  }
  return doc as unknown as StationIndex;
}

/**
 * A station's entry, by MAC in any case.
 *
 * @param index - The station index.
 * @param mac - The MAC asked for.
 * @returns Its entry, or undefined when the index does not list it.
 */
export function findStation(index: StationIndex, mac: string): StationEntry | undefined {
  const wanted = mac.toUpperCase();
  return index.stations.find((s) => s.mac.toUpperCase() === wanted);
}
