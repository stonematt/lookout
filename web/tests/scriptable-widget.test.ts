/**
 * The Scriptable widget's logic (`widgets/scriptable/Lookout.js`), on the
 * golden snapshots: which station a widget shows, and what each of its three
 * views draws. Drawing itself needs Scriptable and is checked on the phone.
 */
import { createRequire } from "node:module";
import { describe, expect, it } from "vitest";
import { FIXTURE_MAC, SALMON_MAC, golden } from "./golden.js";

interface Station {
  mac: string;
  name: string | null;
}
interface Row {
  key: string;
  label: string;
  low: number | null;
  q1: number | null;
  q3: number | null;
  high: number | null;
}
interface Model {
  view: "rain_event" | "dry_spell" | "temperature_ranges";
  name: string;
  sentence: string;
  now: number | null;
  rows?: Row[];
  totalIn?: number;
  medianIn?: number | null;
  curve?: { t: number; inches: number }[];
  daysDry?: number;
  usualDays?: number;
  longestDays?: number | null;
}
interface Widget {
  matchStation(
    stations: Station[],
    parameter: string | null,
  ): { all: true } | { station: Station } | { error: string };
  leadingStation<T extends { view: string }>(entries: T[]): T;
  leadingDashboard(results: PromiseSettledResult<unknown>[]): unknown;
  widgetModel(dashboard: unknown): Model;
  formatDuration(minutes: number): string;
}

const widget = createRequire(import.meta.url)("../../widgets/scriptable/Lookout.js") as Widget;

const STATIONS: Station[] = [
  { mac: FIXTURE_MAC, name: "Lookout Post" },
  { mac: SALMON_MAC, name: "Salmon Run" },
];

/** A golden dashboard, deep-copied so a test can change it. */
function dashboard(scenario: "dry" | "raining"): Record<string, unknown> {
  return structuredClone(golden(scenario, "dashboard")) as Record<string, unknown>;
}

describe("matchStation", () => {
  it("finds a station by name, ignoring case and spaces", () => {
    expect(widget.matchStation(STATIONS, "  salmon   RUN ")).toEqual({ station: STATIONS[1] });
  });

  it("reads an empty parameter as All stations", () => {
    expect(widget.matchStation(STATIONS, "")).toEqual({ all: true });
    expect(widget.matchStation(STATIONS, null)).toEqual({ all: true });
  });

  it("names the real stations when the name matches none", () => {
    const result = widget.matchStation(STATIONS, "Salmon");
    expect(result).toEqual({ error: 'No station named "Salmon". Try Lookout Post or Salmon Run.' });
  });

  it("matches an unnamed station by its MAC", () => {
    const unnamed = [{ mac: SALMON_MAC, name: null }];
    expect(widget.matchStation(unnamed, SALMON_MAC.toLowerCase())).toEqual({
      station: unnamed[0],
    });
  });
});

describe("leadingStation", () => {
  it("puts a rain event before a dry spell before temperature ranges", () => {
    const entries = [
      { mac: "a", view: "temperature_ranges" },
      { mac: "b", view: "dry_spell" },
      { mac: "c", view: "rain_event" },
    ];
    expect(widget.leadingStation(entries).mac).toBe("c");
  });

  it("gives a tie to the station first in the index, the default station", () => {
    const entries = [
      { mac: "a", view: "dry_spell" },
      { mac: "b", view: "dry_spell" },
    ];
    expect(widget.leadingStation(entries).mac).toBe("a");
  });
});

describe("leadingDashboard", () => {
  it("leads with a station that loaded when another station's fetch failed", () => {
    const raining = dashboard("raining");
    const results: PromiseSettledResult<unknown>[] = [
      { status: "rejected", reason: new Error("timeout") },
      { status: "fulfilled", value: dashboard("dry") },
      { status: "fulfilled", value: raining },
    ];

    expect(widget.leadingDashboard(results)).toBe(raining);
  });

  it("fails only when no station loaded", () => {
    const results: PromiseSettledResult<unknown>[] = [
      { status: "rejected", reason: new Error("timeout") },
    ];

    expect(() => widget.leadingDashboard(results)).toThrow();
  });
});

describe("widgetModel", () => {
  it("draws the live event for rain_event", () => {
    const doc = dashboard("raining");
    const active = doc.active_rain_event as { total_in: number };
    const events = doc.rain_events as { all_time: { median_in: number | null } };

    const model = widget.widgetModel(doc);

    expect(model.view).toBe("rain_event");
    expect(model.totalIn).toBe(active.total_in);
    expect(model.medianIn).toBe(events.all_time.median_in);
    expect(model.curve?.length).toBeGreaterThan(1);
    expect(model.sentence).toMatch(/^\d\.\d\d in over .+, still raining$/);
  });

  it("draws today, 7, 30 and 365 days for temperature_ranges", () => {
    const doc = dashboard("raining");
    doc.widget_view = "temperature_ranges";

    const model = widget.widgetModel(doc);

    expect(model.view).toBe("temperature_ranges");
    expect(model.rows?.map((r) => r.key)).toEqual(["today", "7d", "30d", "365d"]);
    expect(model.rows?.[0]?.low).not.toBeNull();
    // The 7-day window is invalid in this golden: its row draws nothing.
    expect(model.rows?.[1]?.low).toBeNull();
    expect(model.sentence).toBe("Above the middle half of yesterday");
  });

  it("says so when no window is complete enough to compare", () => {
    const model = widget.widgetModel(dashboard("dry"));

    expect(model.view).toBe("temperature_ranges");
    expect(model.rows?.every((r) => r.low === null)).toBe(true);
    expect(model.sentence).toBe("No window complete enough to compare");
  });

  it("draws the spell against the usual and the longest for dry_spell", () => {
    const doc = dashboard("dry");
    doc.widget_view = "dry_spell";
    const lead = doc.hero_lead as { rules: { kind: string; value: number; reference: number }[] };
    const calm = lead.rules.find((r) => r.kind === "calm")!;
    calm.value = 9.4;
    calm.reference = 4;

    const model = widget.widgetModel(doc);

    expect(model.view).toBe("dry_spell");
    expect(model.daysDry).toBe(9);
    expect(model.usualDays).toBe(4);
    expect(model.sentence).toBe("Dry for 9 days, longer than the usual 4");
  });

  it("falls back to temperature_ranges for a snapshot without widget_view", () => {
    const doc = dashboard("raining");
    delete doc.widget_view;

    expect(widget.widgetModel(doc).view).toBe("temperature_ranges");
  });
});

describe("formatDuration", () => {
  it("keeps the site's exact short forms", () => {
    expect(widget.formatDuration(45)).toBe("45 m");
    expect(widget.formatDuration(334)).toBe("5 h 34 m");
    expect(widget.formatDuration(1500)).toBe("1 d 1 h");
  });
});
