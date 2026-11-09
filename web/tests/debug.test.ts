// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";

import { parseLastData, renderDump, renderStationDump } from "../src/debug.js";
import { FIXTURE_MAC, SALMON_MAC } from "./golden.js";

const READ_AT = 1767918300000;

describe("parseLastData", () => {
  it("accepts a /last-data document", () => {
    const doc = {
      kind: "last_data",
      read_at_ms: READ_AT,
      stations: [{ mac: FIXTURE_MAC, name: null, last_data: null }],
    };
    expect(parseLastData(doc)).toEqual(doc);
  });

  it.each([{}, { kind: "current", read_at_ms: 1, stations: [] }, null])("refuses %j", (doc) => {
    expect(() => parseLastData(doc)).toThrow();
  });
});

describe("renderStationDump", () => {
  it("dumps lastData as text, never as markup", () => {
    const section = renderStationDump({
      mac: FIXTURE_MAC,
      name: "Fixture Station",
      last_data: { dateutc: READ_AT, note: "<img src=x>" },
    });
    const pre = section.querySelector("pre")!;
    expect(JSON.parse(pre.textContent!)).toMatchObject({ note: "<img src=x>" });
    expect(section.querySelector("img")).toBeNull();
    expect(section.querySelector("h2")!.textContent).toContain(FIXTURE_MAC);
  });

  it("says so when Ambient has no lastData for the station", () => {
    const section = renderStationDump({ mac: SALMON_MAC, name: null, last_data: null });
    expect(section.textContent).toContain("no lastData");
  });
});

describe("renderDump", () => {
  it("shows the read time, then one section per station in order", () => {
    const parts = renderDump({
      kind: "last_data",
      read_at_ms: READ_AT,
      stations: [
        { mac: FIXTURE_MAC, name: null, last_data: null },
        { mac: SALMON_MAC, name: null, last_data: null },
      ],
    });
    expect(parts[0]!.textContent).toContain(new Date(READ_AT).toISOString());
    expect(parts.slice(1).map((p) => p.querySelector("h2")!.textContent)).toEqual([
      `(unnamed) · ${FIXTURE_MAC}`,
      `(unnamed) · ${SALMON_MAC}`,
    ]);
  });
});
