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
 * The schema version the refresh run wrote before the goldens' version, which
 * the site still reads (ADR-0008, `SNAPSHOT_SCHEMA_VERSIONS`): a v3 document
 * carries `hero_lead`, which v4 removed (#480). When the site stops reading 3
 * this builder goes.
 */
export const PREVIOUS_SCHEMA_VERSION = 3;

/**
 * One golden document as the refresh run wrote it at
 * `PREVIOUS_SCHEMA_VERSION`: the golden with `schema_version` set back and,
 * on a dashboard, a v3 `hero_lead` whose calm rule restates `dry_spell`, as
 * the refresh run wrote it. Built here, never by editing a golden file (#351).
 *
 * @param scenario - One of `SCENARIOS`.
 * @param kind - "dashboard" or "context".
 * @returns The parsed JSON, untyped.
 */
export function goldenPrevious(scenario: Scenario, kind: "dashboard" | "context"): unknown {
  const doc = golden(scenario, kind) as Record<string, unknown>;
  if (kind === "context") return { ...doc, schema_version: PREVIOUS_SCHEMA_VERSION };
  const dry = doc.dry_spell as { days_since_rain: number | null; usual_days: number | null };
  const calm = {
    kind: "calm",
    met: true,
    value: dry.days_since_rain,
    unit: "days",
    threshold: null,
    reference: dry.usual_days,
    compared: [],
    missing: null,
  };
  const heroLead = {
    kind: "calm",
    rules: [calm],
    unit: "days",
    now: dry.days_since_rain,
    now_ms: doc.observed_at_ms,
    usual_low: null,
    usual_median: dry.usual_days,
    usual_high: null,
    sample: 0,
    trace: null,
  };
  return { ...doc, schema_version: PREVIOUS_SCHEMA_VERSION, hero_lead: heroLead };
}
