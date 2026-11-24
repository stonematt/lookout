/**
 * Load the golden snapshot documents the Python suite writes, in place, from
 * `tests/golden/snapshots/` at the repo root. They are never copied into
 * `web/`: a schema change in Python then fails these tests too.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { StationIndex } from "../src/stations.js";
import { parseStationIndex } from "../src/stations.js";

/**
 * The golden scenarios `tests/test_snapshot.py` writes: `dry` (a young
 * station, no earlier year), `raining` (mid event) and `seasoned` (two
 * earlier years around its date, #461).
 */
export const SCENARIOS = ["dry", "raining", "seasoned"] as const;

/** One golden scenario. */
export type Scenario = (typeof SCENARIOS)[number];

const CONDITIONS = ["wind", "pressure", "humidity", "sun", "daylight", "inside"];

/**
 * Each golden's mosaic, in the order its wet mode sets (#658, Mosaic
 * decision 24): `dry` and `raining` are wet mode by the calendar (too few
 * countable hours, in November and January), Wet hours beside Temperature;
 * `seasoned` is dry mode by the weather (a dry May), Dry spell beside
 * Temperature and Rain since Oct 1 after Last 10 days. Rain leads on
 * `raining` (Overview C, #661): Wet hours and Rain since Oct 1 leave the
 * mosaic for the rain tile's season foot.
 */
export const MOSAIC_ORDER: Record<Scenario, readonly string[]> = {
  dry: ["temperature", "wet-hours", "rain", "last-10-days", ...CONDITIONS],
  raining: ["temperature", "last-10-days", ...CONDITIONS],
  seasoned: ["temperature", "dry-spell", "last-10-days", "rain", ...CONDITIONS],
};

/**
 * Each golden's tile under the now tile: the feature, or the rain tile
 * (section `rain`) while rain leads (#661).
 */
export const LEAD_TILE: Record<Scenario, string> = {
  dry: "feature",
  raining: "rain",
  seasoned: "feature",
};

/**
 * The documents a refresh run publishes per station: the snapshot's two, and
 * its status document (`tests/test_status.py` writes those goldens).
 */
export type DocumentKind = "dashboard" | "context" | "status";

/** The directory the goldens live in. */
export const GOLDEN_DIR = `${resolve(import.meta.dirname, "../../tests/golden/snapshots")}/`;

/** The directory the golden status documents live in. */
export const STATUS_DIR = `${resolve(import.meta.dirname, "../../tests/golden/status")}/`;

/**
 * The raw text of one golden document.
 *
 * @param scenario - One of `SCENARIOS`.
 * @param kind - "dashboard", "context" or "status".
 * @returns The file's text, as Storj would serve it.
 */
export function goldenText(scenario: Scenario, kind: DocumentKind): string {
  const dir = kind === "status" ? STATUS_DIR : GOLDEN_DIR;
  return readFileSync(`${dir}${scenario}.${kind}.json`, "utf8");
}

/**
 * One golden document, parsed.
 *
 * @param scenario - One of `SCENARIOS`.
 * @param kind - "dashboard", "context" or "status".
 * @returns The parsed JSON, untyped: the code under test types it.
 */
export function golden(scenario: Scenario, kind: DocumentKind): unknown {
  return JSON.parse(goldenText(scenario, kind));
}

/**
 * The raw text of one golden event curve file, as Storj would serve it.
 *
 * @param scenario - One of `SCENARIOS`.
 * @param startMs - The event's start.
 * @returns The file's text.
 */
export function goldenCurveText(scenario: Scenario, startMs: number): string {
  return readFileSync(`${GOLDEN_DIR}events/${scenario}.${startMs}.json`, "utf8");
}

/**
 * One golden event curve file, parsed.
 *
 * @param scenario - One of `SCENARIOS`.
 * @param startMs - The event's start.
 * @returns The parsed JSON, untyped.
 */
export function goldenCurve(scenario: Scenario, startMs: number): unknown {
  return JSON.parse(goldenCurveText(scenario, startMs));
}

/** The golden station index's two stations (`tests/golden/stations/`). */
export const FIXTURE_MAC = "00:11:22:33:44:55";
export const SALMON_MAC = "00:11:22:33:44:66";

/**
 * The golden station index, parsed: Fixture Station (the default), then
 * Salmon Run, which has no snapshot yet.
 *
 * @returns The typed index.
 */
export function twoStations(): StationIndex {
  const path = resolve(import.meta.dirname, "../../tests/golden/stations/raining.stations.json");
  return parseStationIndex(JSON.parse(readFileSync(path, "utf8")));
}

/**
 * The golden station index cut to its first station, as the live inventory
 * stands today.
 *
 * @returns The typed index.
 */
export function oneStation(): StationIndex {
  const index = twoStations();
  return { ...index, stations: index.stations.slice(0, 1) };
}

/**
 * The schema version after the goldens' version, which the site reads before
 * the refresh run writes it (ADR-0008, `SNAPSHOT_SCHEMA_VERSIONS`, #667): a
 * v5 document counts `days_since_rain` in whole dry days and drops the Rain
 * tab's prorated `recent.wet_hours`. Once the refresh run writes 5 (#668) the
 * goldens are v5 and this builder goes.
 */
export const NEXT_SCHEMA_VERSION = 5;

/**
 * One golden document as the refresh run will write it at
 * `NEXT_SCHEMA_VERSION`: the golden with `schema_version` set forward and, on
 * a dashboard, `days_since_rain` in whole days (`wholeDays`, or the golden's
 * own value cut to whole days) and no prorated `recent.wet_hours`. Built
 * here, never by editing a golden file (#351).
 *
 * @param scenario - One of `SCENARIOS`.
 * @param kind - "dashboard" or "context".
 * @param wholeDays - The whole dry days to carry; the golden's, floored, when
 *   omitted.
 * @returns The parsed JSON, untyped.
 */
export function goldenNext(
  scenario: Scenario,
  kind: "dashboard" | "context",
  wholeDays?: number | null,
): unknown {
  const doc = structuredClone(golden(scenario, kind)) as Record<string, unknown>;
  doc.schema_version = NEXT_SCHEMA_VERSION;
  if (kind === "context") return doc;
  const dry = doc.dry_spell as { days_since_rain: number | null; usual_days: number | null };
  const since = dry.days_since_rain;
  dry.days_since_rain =
    wholeDays !== undefined ? wholeDays : since === null ? null : Math.floor(since);
  const tab = doc.rain_tab as { recent: Record<string, unknown> };
  delete tab.recent.wet_hours;
  return doc;
}
