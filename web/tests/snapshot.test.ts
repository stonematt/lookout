import { describe, expect, it } from "vitest";

import {
  SCHEMA_VERSION,
  SnapshotContractError,
  newestObservation,
  parseContext,
  parseDashboard,
} from "../src/snapshot.js";
import { golden } from "./golden.js";

describe("the contract version", () => {
  it("is the version the Python builder writes (read from the goldens)", () => {
    for (const scenario of ["dry", "raining"] as const) {
      for (const kind of ["dashboard", "context"] as const) {
        const doc = golden(scenario, kind) as { schema_version: number };
        expect(doc.schema_version).toBe(SCHEMA_VERSION);
      }
    }
  });
});

describe("parseDashboard", () => {
  it.each([
    "gaps",
    "rain_events",
    "hero_lead",
    "rain_tab",
    "temperature_context",
    "almanac",
  ])("refuses a dashboard missing %s", (section) => {
    const doc = { ...(golden("dry", "dashboard") as Record<string, unknown>) };
    delete doc[section];
    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });

  it("refuses a water year of events with no gap status (#156)", () => {
    const doc = structuredClone(golden("raining", "dashboard")) as {
      rain_events: { seasons: { status: unknown }[] };
    };
    doc.rain_events.seasons[0]!.status = null;
    expect(() => parseDashboard(doc)).toThrow("rain_events.seasons");
  });

  it("refuses recent rain that does not say which hours are to come (#156)", () => {
    const doc = structuredClone(golden("raining", "dashboard")) as {
      rain_tab: { recent: Record<string, unknown> };
    };
    delete doc.rain_tab.recent.first_hour_to_come;
    expect(() => parseDashboard(doc)).toThrow("first_hour_to_come");
  });

  it("reads the raining golden with its live event", () => {
    const snap = parseDashboard(golden("raining", "dashboard"));
    expect(snap.station).toEqual({
      mac: "00:11:22:33:44:55",
      name: "Fixture Station",
    });
    expect(snap.observed_at_ms).toBe(1767918300000);
    expect(snap.archive_newest_ms).toBe(1767916500000);
    expect(snap.active_rain_event?.total_in).toBe(0.3);
  });

  it("reads the dry golden with no event", () => {
    const snap = parseDashboard(golden("dry", "dashboard"));
    expect(snap.active_rain_event).toBeNull();
    expect(snap.current.temperature_f).toBe(39.4);
  });

  it("refuses a context block", () => {
    expect(() => parseDashboard(golden("dry", "context"))).toThrow(
      SnapshotContractError,
    );
  });

  it("refuses another schema version", () => {
    const doc = { ...(golden("dry", "dashboard") as object), schema_version: SCHEMA_VERSION + 1 };
    expect(() => parseDashboard(doc)).toThrow(`schema_version ${SCHEMA_VERSION + 1}`);
  });

  it("keeps fields it does not know", () => {
    const doc = { ...(golden("dry", "dashboard") as object), added_later: 1 };
    expect(parseDashboard(doc)).toHaveProperty("added_later", 1);
  });
});

describe("parseContext", () => {
  it("reads both context goldens", () => {
    expect(parseContext(golden("raining", "context")).kind).toBe("context");
    expect(parseContext(golden("dry", "context")).built_at_ms).toBe(
      1763539200000,
    );
  });

  it("refuses a dashboard document", () => {
    expect(() => parseContext(golden("dry", "dashboard"))).toThrow(
      /expected kind "context"/,
    );
  });
});

describe("newestObservation", () => {
  it("takes the later time and skips nulls", () => {
    expect(newestObservation(5, 9)).toBe(9);
    expect(newestObservation(null, 9)).toBe(9);
    expect(newestObservation(null, undefined)).toBeNull();
  });
});
