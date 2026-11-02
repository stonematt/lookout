// @vitest-environment happy-dom
/**
 * A catalog the refresh run could not update (#118): the contract reads its
 * mark, and the Rain Events and Solar tabs show their figures as old.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import { mountEvents } from "../src/events/tab.js";
import { formatStamp } from "../src/format.js";
import { renderSolar } from "../src/solar/index.js";
import type { DashboardSnapshot, StaleCatalog } from "../src/snapshot.js";
import {
  SnapshotContractError,
  parseDashboard,
  parseEventCurve,
  staleCatalog,
} from "../src/snapshot.js";
import { staleCatalogNote } from "../src/stale.js";
import { golden, goldenCurve } from "./golden.js";

const AS_OF = Date.UTC(2026, 0, 8, 0, 45);

function withStale(
  scenario: "dry" | "raining",
  stale: unknown,
): DashboardSnapshot {
  const doc = structuredClone(golden(scenario, "dashboard")) as Record<string, unknown>;
  doc.stale_catalogs = stale;
  return parseDashboard(doc);
}

let panel: HTMLElement;

beforeEach(() => {
  history.replaceState(null, "", "/?fixture=raining#events");
  document.body.innerHTML = "<section id=panel></section>";
  panel = document.getElementById("panel")!;
});

const text = (el: Element | null | undefined) =>
  el?.textContent?.replace(/\s+/g, " ").trim();

describe("the stale_catalogs contract", () => {
  it("is empty in a normal run's goldens", () => {
    for (const scenario of ["dry", "raining"] as const) {
      expect(parseDashboard(golden(scenario, "dashboard")).stale_catalogs).toEqual([]);
    }
  });

  it("reads a document published before the field as none stale", () => {
    const doc = structuredClone(golden("dry", "dashboard")) as Record<string, unknown>;
    delete doc.stale_catalogs;
    expect(parseDashboard(doc).stale_catalogs).toEqual([]);
  });

  it("reads each mark, dated or not", () => {
    const marks: StaleCatalog[] = [
      { catalog: "events", as_of_ms: AS_OF },
      { catalog: "energy", as_of_ms: null },
    ];
    const snap = withStale("dry", marks);
    expect(staleCatalog(snap, "events")).toEqual(marks[0]);
    expect(staleCatalog(snap, "energy")).toEqual(marks[1]);
  });

  it.each([
    ["not a list", { catalog: "events" }],
    ["an unknown catalog", [{ catalog: "curves", as_of_ms: null }]],
    ["a time that is not an integer", [{ catalog: "energy", as_of_ms: "soon" }]],
  ])("refuses %s", (_, stale) => {
    expect(() => withStale("dry", stale)).toThrow(SnapshotContractError);
  });
});

describe("the note", () => {
  it("dates the figures by the stored catalog", () => {
    const snap = withStale("dry", [{ catalog: "energy", as_of_ms: AS_OF }]);
    expect(staleCatalogNote(snap, "energy")).toBe(
      "The energy catalog could not be updated on the latest refresh, so the " +
        `solar energy figures are as of ${formatStamp(AS_OF)}.`,
    );
    expect(staleCatalogNote(snap, "events")).toBeNull();
  });

  it("says out of date when no catalog was stored", () => {
    const snap = withStale("dry", [{ catalog: "events", as_of_ms: null }]);
    expect(staleCatalogNote(snap, "events")).toBe(
      "The Event Catalog could not be updated on the latest refresh, so these " +
        "events may be out of date.",
    );
  });
});

describe("the Rain Events tab", () => {
  const deps = () => ({
    fetchCurve: vi.fn(async (start: number) =>
      parseEventCurve(goldenCurve("raining", start)),
    ),
    draw: vi.fn(async () => {}),
    nowMs: () => Date.UTC(2026, 0, 9, 0, 30),
  });

  it("shows its catalog as old when it is stale", () => {
    const snap = withStale("raining", [{ catalog: "events", as_of_ms: AS_OF }]);
    mountEvents(panel, snap, deps());

    const note = panel.querySelector<HTMLElement>(".ev-stale")!;
    expect(note.hidden).toBe(false);
    expect(text(note)).toBe(staleCatalogNote(snap, "events"));
  });

  it("shows the note with an empty catalog too", () => {
    const snap = withStale("raining", [{ catalog: "events", as_of_ms: null }]);
    snap.rain_events.events = [];
    mountEvents(panel, snap, deps());

    expect(panel.querySelector<HTMLElement>(".ev-stale")!.hidden).toBe(false);
  });

  it("shows no note on a normal run or when only energy is stale", () => {
    for (const stale of [[], [{ catalog: "energy", as_of_ms: AS_OF }]]) {
      mountEvents(panel, withStale("raining", stale), deps());
      expect(panel.querySelector<HTMLElement>(".ev-stale")!.hidden).toBe(true);
    }
  });
});

describe("the Solar tab", () => {
  const render = (snap: DashboardSnapshot) =>
    renderSolar(panel, snap, () => "", () => {});

  it("shows its catalog as old when it is stale", () => {
    const snap = withStale("dry", [{ catalog: "energy", as_of_ms: AS_OF }]);
    render(snap);

    expect(text(panel.querySelector(".solar-stale"))).toBe(
      staleCatalogNote(snap, "energy"),
    );
  });

  it("shows the note when there is no solar record either", () => {
    const snap = withStale("dry", [{ catalog: "energy", as_of_ms: null }]);
    snap.solar_tab = null;
    render(snap);

    expect(panel.querySelector(".solar-stale")).not.toBeNull();
    expect(text(panel)).toContain("The solar record is not in this snapshot yet.");
  });

  it("shows no note on a normal run or when only events are stale", () => {
    for (const stale of [[], [{ catalog: "events", as_of_ms: AS_OF }]]) {
      render(withStale("dry", stale));
      expect(panel.querySelector(".solar-stale")).toBeNull();
    }
  });
});
