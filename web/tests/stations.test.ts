/**
 * The station index contract (#272), read from the goldens the Python suite
 * writes (`tests/test_station_index.py`), so a shape change in Python fails
 * here too; and the client's `/stations` read. No network.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it, vi } from "vitest";

import { fetchStations, readUrl } from "../src/load.js";
import { SnapshotContractError } from "../src/snapshot.js";
import { STATION_INDEX_SCHEMA_VERSION, findStation, parseStationIndex } from "../src/stations.js";
import type { Scenario } from "./golden.js";
import { SCENARIOS } from "./golden.js";

const STATIONS_DIR = resolve(import.meta.dirname, "../../tests/golden/stations");

function goldenIndex(scenario: Scenario): Record<string, unknown> {
  return JSON.parse(readFileSync(`${STATIONS_DIR}/${scenario}.stations.json`, "utf8")) as Record<
    string,
    unknown
  >;
}

/** Each golden index's first station: the scenario's own. */
const GOLDEN_FIRST: Record<Scenario, string> = {
  dry: "Fixture Station",
  raining: "Fixture Station",
  seasoned: "Seasoned Station",
};

describe("the station index contract", () => {
  it.each(SCENARIOS)("reads the %s golden", (scenario) => {
    const index = parseStationIndex(goldenIndex(scenario));
    expect(index.schema_version).toBe(STATION_INDEX_SCHEMA_VERSION);
    expect(index.stations.map((s) => s.name)).toEqual([GOLDEN_FIRST[scenario], "Salmon Run"]);
    expect(index.stations[1]!.newest_reading_ms).toBeNull();
  });

  it.each([
    ["another schema version", { schema_version: 2 }],
    ["another kind", { kind: "dashboard" }],
    ["no stations", { stations: [] }],
    ["no built_at_ms", { built_at_ms: null }],
    ["an entry without a tz", { stations: [{ mac: "00:11:22:33:44:55", name: null }] }],
    ["a time that is not epoch ms", { stations: [{ ...entry(), archive_start_ms: "2022" }] }],
    ["a station listed twice", { stations: [entry(), { ...entry(), mac: "00:11:22:33:44:55" }] }],
  ])("refuses %s", (_name, change) => {
    expect(() => parseStationIndex({ ...goldenIndex("dry"), ...change })).toThrow(
      SnapshotContractError,
    );
  });

  it("finds a station by MAC in any case", () => {
    const index = parseStationIndex(goldenIndex("dry"));
    expect(findStation(index, "00:11:22:33:44:66")?.name).toBe("Salmon Run");
    expect(findStation(index, "aa:bb:cc:dd:ee:ff")).toBeUndefined();
  });
});

function entry() {
  return {
    mac: "00:11:22:33:44:55",
    name: "Fixture Station",
    tz: "America/Los_Angeles",
    newest_reading_ms: 1,
    archive_start_ms: 1,
  };
}

describe("the client's /stations read", () => {
  it("never passes the page's station on, which /stations refuses", () => {
    expect(readUrl("/stations", "?station=00:11:22:33:44:66", {}, false)).toBe("/stations");
    expect(readUrl("/stations", "?station=x&fixture=raining&age=3", {}, true)).toBe(
      "/stations?fixture=raining&age=3",
    );
  });

  it("fetches and checks the index", async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(Response.json(goldenIndex("raining"))));
    const index = await fetchStations("?station=x", fetchImpl as unknown as typeof fetch);
    expect(index.stations).toHaveLength(2);
    expect(fetchImpl).toHaveBeenCalledWith("/stations", expect.anything());
  });
});
