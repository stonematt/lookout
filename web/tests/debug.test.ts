// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";

import { debugSearch, renderStationDump } from "../src/debug.js";
import { SCHEMA_VERSION, parseCurrent } from "../src/snapshot.js";
import { FIXTURE_MAC, SALMON_MAC, twoStations } from "./golden.js";

const station = twoStations().stations[0]!;

describe("debugSearch", () => {
  it("sets the station and keeps the lane server's parameters", () => {
    const query = new URLSearchParams(debugSearch("?fixture=dry&station=x", SALMON_MAC));
    expect(query.get("station")).toBe(SALMON_MAC);
    expect(query.get("fixture")).toBe("dry");
  });
});

describe("renderStationDump", () => {
  it("dumps last_data as text, never as markup", () => {
    const reading = parseCurrent({
      schema_version: SCHEMA_VERSION,
      kind: "current",
      station: { mac: FIXTURE_MAC, name: "Fixture Station" },
      observed_at_ms: 1767918300000,
      built_at_ms: null,
      last_data: { dateutc: 1767918300000, note: "<img src=x>" },
      context: null,
    });
    const section = renderStationDump(station, reading);
    const pre = section.querySelector("pre")!;
    expect(JSON.parse(pre.textContent!)).toMatchObject({ note: "<img src=x>" });
    expect(section.querySelector("img")).toBeNull();
    expect(section.querySelector("h2")!.textContent).toContain(FIXTURE_MAC);
  });

  it("shows a failed read instead of the dump", () => {
    const section = renderStationDump(station, new Error("/current answered 502"));
    expect(section.querySelector("pre")).toBeNull();
    expect(section.textContent).toContain("502");
  });
});
