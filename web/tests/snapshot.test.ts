import { describe, expect, it } from "vitest";

import {
  SCHEMA_VERSION,
  SNAPSHOT_SCHEMA_VERSIONS,
  SnapshotContractError,
  newestObservation,
  parseContext,
  parseCurrent,
  parseDashboard,
} from "../src/snapshot.js";
import { fetchCurrent } from "../src/load.js";
import { PREVIOUS_SCHEMA_VERSION, SCENARIOS, golden, goldenPrevious } from "./golden.js";

describe("the contract version", () => {
  it("is the version the Python builder writes (read from the goldens)", () => {
    for (const scenario of SCENARIOS) {
      for (const kind of ["dashboard", "context"] as const) {
        const doc = golden(scenario, kind) as { schema_version: number };
        expect(doc.schema_version).toBe(SCHEMA_VERSION);
      }
    }
  });
});

describe("parseDashboard", () => {
  it.each(["gaps", "rain_events", "hero_lead", "rain_tab", "temperature_context", "almanac"])(
    "refuses a dashboard missing %s",
    (section) => {
      const doc = { ...(golden("dry", "dashboard") as Record<string, unknown>) };
      delete doc[section];
      expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
    },
  );

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

  it.each([
    ["no zone", undefined],
    ["an empty zone", ""],
    ["a zone no runtime knows", "Mars/Olympus_Mons"],
  ])("refuses a station with %s (#197)", (_case, tz) => {
    const doc = structuredClone(golden("raining", "dashboard")) as {
      station: Record<string, unknown>;
    };
    if (tz === undefined) delete doc.station.tz;
    else doc.station.tz = tz;
    expect(() => parseDashboard(doc)).toThrow("dashboard: station.tz");
  });

  it("reads the raining golden with its live event", () => {
    const snap = parseDashboard(golden("raining", "dashboard"));
    expect(snap.station).toEqual({
      mac: "00:11:22:33:44:55",
      name: "Fixture Station",
      tz: "America/Los_Angeles",
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

  it("reads the sun and moon (#344)", () => {
    const sunMoon = parseDashboard(golden("dry", "dashboard")).sun_moon;
    expect(sunMoon).not.toBeNull();
    // 19 November 2025 in the station's zone; new moon that night, 22:47 PST.
    const day = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles" });
    expect(day.format(sunMoon!.sunrise_ms)).toBe("2025-11-19");
    expect(day.format(sunMoon!.sunset_ms)).toBe("2025-11-19");
    expect(sunMoon!.daylight_change_minutes).toBeLessThan(0);
    expect(sunMoon!.moon_phase).toBe("new");
  });

  it("reads a station with no coordinates as no sun and moon", () => {
    const doc = { ...(golden("dry", "dashboard") as object), sun_moon: null };
    expect(parseDashboard(doc).sun_moon).toBeNull();
  });

  it.each<[string, (block: Record<string, unknown>) => void]>([
    ["an unknown moon phase", (b) => (b.moon_phase = "blue")],
    ["illumination over 100", (b) => (b.moon_illumination_pct = 101)],
    ["a sunset before sunrise", (b) => (b.sunset_ms = (b.sunrise_ms as number) - 1)],
    ["a fractional day count", (b) => (b.days_to_full_moon = 1.5)],
  ])("refuses sun and moon with %s", (_case, spoil) => {
    const doc = structuredClone(golden("dry", "dashboard")) as {
      sun_moon: Record<string, unknown>;
    };
    spoil(doc.sun_moon);
    expect(() => parseDashboard(doc)).toThrow("dashboard: sun_moon malformed");
  });

  it("refuses a dashboard missing sun_moon", () => {
    const doc = { ...(golden("dry", "dashboard") as Record<string, unknown>) };
    delete doc.sun_moon;
    expect(() => parseDashboard(doc)).toThrow("dashboard: sun_moon malformed");
  });

  it.each<[string, (s: Record<string, unknown>) => void]>([
    ["no key", (s) => delete s.key],
    ["a text humidity", (s) => (s.humidity_pct = "45")],
    ["no humidity field", (s) => delete s.humidity_pct],
    ["no read time", (s) => delete s.observed_at_ms],
    ["a fractional read time", (s) => (s.observed_at_ms = 1.5)],
  ])("refuses an indoor sensor with %s", (_case, spoil) => {
    const doc = structuredClone(golden("seasoned", "dashboard")) as {
      indoor: Record<string, unknown>[];
    };
    expect(doc.indoor.length).toBeGreaterThan(0);
    spoil(doc.indoor[0]!);
    expect(() => parseDashboard(doc)).toThrow("dashboard: indoor malformed");
  });

  // Release A3's fields (#539, #540, #554): read when present, null or absent.
  type Raw = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  it.each<[string, (d: Raw) => void, string]>([
    [
      "a usual to date with a text percent",
      (d) => (d.rain_tab.usual_to_date = { usual_in: 1, years: 1, percent: "74" }),
      "dashboard: rain_tab.usual_to_date malformed",
    ],
    [
      "ten-day highs without a delta",
      (d) => (d.temperature_context.ten_day_highs = { days: 10 }),
      "dashboard: temperature_context.ten_day_highs malformed",
    ],
    [
      "an outdoor trace with a text reading",
      (d) => (d.outdoor_trace.humidity_pct[0] = "65"),
      "dashboard: outdoor_trace malformed",
    ],
    [
      "an outdoor trace whose series differ in length",
      (d) => d.outdoor_trace.pressure_inhg.pop(),
      "dashboard: outdoor_trace malformed",
    ],
  ])("refuses %s", (_case, spoil, message) => {
    const doc = structuredClone(golden("seasoned", "dashboard")) as Raw;
    spoil(doc);
    expect(() => parseDashboard(doc)).toThrow(message);
  });

  it("reads Release A3's fields, and a snapshot without them (N−1)", () => {
    const doc = structuredClone(golden("seasoned", "dashboard")) as Raw;
    const read = parseDashboard(doc);
    expect(read.rain_tab.usual_to_date).toEqual({ usual_in: 0.304, years: 1, percent: 145 });
    expect(read.temperature_context.ten_day_highs).toEqual({ delta_f: 2, days: 10 });
    expect(read.outdoor_trace?.pressure_inhg).toHaveLength(24);

    delete doc.rain_tab.usual_to_date;
    delete doc.temperature_context.ten_day_highs;
    delete doc.outdoor_trace;
    delete doc.feature.event_start_ms;
    const older = parseDashboard(doc);
    expect(older.rain_tab.usual_to_date).toBeUndefined();
    expect(older.temperature_context.ten_day_highs).toBeUndefined();
    expect(older.outdoor_trace).toBeUndefined();
  });

  it("reads an indoor sensor with no humidity", () => {
    const doc = structuredClone(golden("seasoned", "dashboard")) as {
      indoor: Record<string, unknown>[];
    };
    doc.indoor[0]!.humidity_pct = null;
    expect(parseDashboard(doc).indoor?.[0]?.humidity_pct).toBeNull();
  });

  it("refuses a context block", () => {
    expect(() => parseDashboard(golden("dry", "context"))).toThrow(SnapshotContractError);
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
    expect(parseContext(golden("dry", "context")).built_at_ms).toBe(1763539200000);
  });

  it("names the station's zone", () => {
    expect(parseContext(golden("dry", "context")).station.tz).toBe("America/Los_Angeles");
  });

  it("refuses a station with no zone (#197)", () => {
    const doc = structuredClone(golden("dry", "context")) as {
      station: Record<string, unknown>;
    };
    delete doc.station.tz;
    expect(() => parseContext(doc)).toThrow("context: station.tz");
  });

  it("refuses a dashboard document", () => {
    expect(() => parseContext(golden("dry", "dashboard"))).toThrow(/expected kind "context"/);
  });
});

describe("parseCurrent", () => {
  const LIVE = 1767918300000;
  const current = (context: unknown) => ({
    schema_version: SCHEMA_VERSION,
    kind: "current",
    station: { mac: "00:11:22:33:44:55", name: "Fixture Station" },
    observed_at_ms: LIVE,
    built_at_ms: 1767916800000,
    last_data: { dateutc: LIVE, tempf: 59.5 },
    context,
  });

  it("reads a body with its context", () => {
    const body = parseCurrent(current(golden("raining", "context")));
    expect(body.context?.kind).toBe("context");
  });

  it("keeps the live reading when the context is another schema version", () => {
    // A refresh run that bumps the schema writes context.json before this
    // site is redeployed: the context no longer fits, the live reading does.
    const skewed = { ...(golden("raining", "context") as object), schema_version: 99 };

    const body = parseCurrent(current(skewed));

    expect(body.context).toBeNull();
    expect(body.last_data).toEqual({ dateutc: LIVE, tempf: 59.5 });
    expect(body.observed_at_ms).toBe(LIVE);
  });

  it("lets the page's /current read keep the live reading over a skewed context", async () => {
    const skewed = { ...(golden("raining", "context") as object), schema_version: 99 };
    const fetchImpl = (async () => Response.json(current(skewed))) as typeof fetch;

    const body = await fetchCurrent("", fetchImpl);

    expect(body.last_data?.dateutc).toBe(LIVE);
    expect(body.observed_at_ms).toBe(LIVE);
    expect(body.context).toBeNull();
  });

  it("keeps a v2 context: the envelope is this site's, the context is the refresh run's", () => {
    const body = parseCurrent(current(goldenPrevious("raining", "context")));
    expect(body.context?.schema_version).toBe(PREVIOUS_SCHEMA_VERSION);
  });

  it("refuses a body at schema_version 2: this site's own server writes 3", () => {
    expect(() => parseCurrent({ ...current(null), schema_version: 2 })).toThrow(
      "current: schema_version 2, this site reads 3",
    );
  });

  it("still refuses a body of another schema version", () => {
    expect(() => parseCurrent({ ...current(null), schema_version: 99 })).toThrow(
      SnapshotContractError,
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

describe("the schema version before (N-1, ADR-0008)", () => {
  // The site reads N and N-1 so the site and the refresh run need not switch
  // at the same moment (#351). PREVIOUS_SCHEMA_VERSION is the one constant the
  // next bump changes; these tests keep guarding the rule.
  it("is the version just before SCHEMA_VERSION, and one the site reads", () => {
    expect(PREVIOUS_SCHEMA_VERSION).toBe(SCHEMA_VERSION - 1);
    expect(SNAPSHOT_SCHEMA_VERSIONS).toContain(PREVIOUS_SCHEMA_VERSION);
  });

  it("reads an N-1 dashboard, its sun_moon null", () => {
    for (const scenario of SCENARIOS) {
      const snap = parseDashboard(goldenPrevious(scenario, "dashboard"));
      expect(snap.schema_version).toBe(PREVIOUS_SCHEMA_VERSION);
      expect(snap.sun_moon).toBeNull();
    }
  });

  it("reads an N-1 context", () => {
    const block = parseContext(goldenPrevious("raining", "context"));
    expect(block.schema_version).toBe(PREVIOUS_SCHEMA_VERSION);
    expect(block.kind).toBe("context");
  });

  for (const kind of ["dashboard", "context"] as const) {
    const parse = kind === "dashboard" ? parseDashboard : parseContext;
    for (const version of [1, 4, undefined]) {
      it(`refuses a ${kind} at schema_version ${String(version)}`, () => {
        const doc = { ...(golden("dry", kind) as Record<string, unknown>) };
        if (version === undefined) delete doc.schema_version;
        else doc.schema_version = version;
        expect(() => parse(doc)).toThrow(SnapshotContractError);
      });
    }

    it(`names the versions it reads when it refuses a ${kind}`, () => {
      const doc = { ...(golden("dry", kind) as object), schema_version: 4 };
      expect(() => parse(doc)).toThrow(`${kind}: schema_version 4, this site reads 2 or 3`);
    });
  }

  it("still refuses a v3 dashboard without sun_moon", () => {
    const doc = { ...(golden("raining", "dashboard") as Record<string, unknown>) };
    delete doc.sun_moon;
    expect(doc.schema_version).toBe(3);
    expect(() => parseDashboard(doc)).toThrow("dashboard: sun_moon malformed");
  });
});
