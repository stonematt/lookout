/**
 * Load the golden snapshot documents the Python suite writes, in place, from
 * `tests/golden/snapshots/` at the repo root. They are never copied into
 * `web/`: a schema change in Python then fails these tests too.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/** The golden scenarios `tests/test_snapshot.py` writes. */
export type Scenario = "dry" | "raining";

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
 * @param scenario - "dry" or "raining".
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
 * @param scenario - "dry" or "raining".
 * @param kind - "dashboard", "context" or "status".
 * @returns The parsed JSON, untyped: the code under test types it.
 */
export function golden(scenario: Scenario, kind: DocumentKind): unknown {
  return JSON.parse(goldenText(scenario, kind));
}

/**
 * The raw text of one golden event curve file, as Storj would serve it.
 *
 * @param scenario - "dry" or "raining".
 * @param startMs - The event's start.
 * @returns The file's text.
 */
export function goldenCurveText(scenario: Scenario, startMs: number): string {
  return readFileSync(`${GOLDEN_DIR}events/${scenario}.${startMs}.json`, "utf8");
}

/**
 * One golden event curve file, parsed.
 *
 * @param scenario - "dry" or "raining".
 * @param startMs - The event's start.
 * @returns The parsed JSON, untyped.
 */
export function goldenCurve(scenario: Scenario, startMs: number): unknown {
  return JSON.parse(goldenCurveText(scenario, startMs));
}
