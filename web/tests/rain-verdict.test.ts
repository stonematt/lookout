// @vitest-environment happy-dom
/**
 * The Rain tab's headline as the water year's verdict (#640): the golden
 * snapshots, each verdict word, the fallback with no verdict, and the
 * window strip's contract.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { renderRain } from "../src/rain/index.js";
import { verdictHeadline, yearsHeadline } from "../src/rain/text.js";
import type { DashboardSnapshot, UsualToDate } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const seasoned = parseDashboard(golden("seasoned", "dashboard"));
const raining = parseDashboard(golden("raining", "dashboard"));

let panel: HTMLElement;

beforeEach(() => {
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const finding = () => text(panel.querySelector("[data-section=years] [data-part=finding]"));

/** The seasoned snapshot with its usual to date changed. */
function withUsual(usual: UsualToDate | null | undefined): DashboardSnapshot {
  const tab = { ...seasoned.rain_tab, usual_to_date: usual };
  if (usual === undefined) delete tab.usual_to_date;
  return { ...seasoned, rain_tab: tab };
}

const SEASONED_FALLBACK =
  "Water year 2026 has 0.44 in of rain since Oct 1, against ≈ 0.30 in (2025) by this date";

describe("the Rain tab's headline", () => {
  it("leads with the verdict over the water-year curves", () => {
    renderRain(panel, seasoned);

    expect(finding()).toBe(
      "Water year 2026 is wetter than usual: 0.44 in since Oct 1, " +
        "against a usual 0.30 in by this date (2025)",
    );
    expect(panel.querySelector("[data-section=years] .rain-chart")).not.toBeNull();
  });

  it.each([
    ["drier", "drier than usual"],
    ["about the same", "about the same as usual"],
  ] as const)("prints the snapshot's word: %s", (verdict, words) => {
    const usual = {
      usual_in: 0.5,
      years: 3,
      percent: 88,
      verdict,
      water_years: [2023, 2024, 2025],
    };

    expect(verdictHeadline(withUsual(usual).rain_tab)).toBe(
      `Water year 2026 is ${words}: 0.44 in since Oct 1, ` +
        "against a usual 0.50 in by this date (2025, 2024 and 2023)",
    );
  });

  it("counts the years past three, and without their names", () => {
    const four = { usual_in: 0.5, years: 4, percent: 140, verdict: "wetter" as const };

    for (const water_years of [[2022, 2023, 2024, 2025], undefined]) {
      expect(verdictHeadline(withUsual({ ...four, water_years }).rain_tab)).toMatch(
        /by this date \(4 earlier years\)$/,
      );
    }
  });

  it("falls back to naming the years with no verdict", () => {
    const old = { usual_in: 0.304, years: 1, percent: 145 };
    const nullVerdict = { ...old, verdict: null };

    for (const snapshot of [withUsual(old), withUsual(nullVerdict), withUsual(undefined)]) {
      expect(verdictHeadline(snapshot.rain_tab)).toBeNull();
      expect(yearsHeadline(snapshot.rain_tab)).toBe(SEASONED_FALLBACK);
    }
  });

  it("gives a young station no verdict", () => {
    expect(raining.rain_tab.usual_to_date).toBeNull();
    expect(verdictHeadline(raining.rain_tab)).toBeNull();
  });
});

describe("the window strip's contract", () => {
  it("reads each window against its usual range, in the windows' order", () => {
    const strip = seasoned.rain_tab.window_strip!;

    expect(strip.map((w) => w.key)).toEqual(seasoned.rain_tab.windows.map((w) => w.key));
    expect(strip.find((w) => w.key === "water_year")).toMatchObject({
      rain_in: 0.44,
      years: 1,
      low_in: 0.304,
      high_in: 0.304,
      mark: "above usual range",
    });
    expect(strip.find((w) => w.key === "365d")).toMatchObject({ years: 0, low_in: null });
  });

  it("reads a snapshot without it (N−1)", () => {
    const doc = structuredClone(golden("seasoned", "dashboard")) as {
      rain_tab: Record<string, unknown>;
    };
    delete doc.rain_tab.window_strip;

    expect(parseDashboard(doc).rain_tab.window_strip).toBeUndefined();
  });

  it.each([
    ["a verdict word as a mark", (w: Record<string, unknown>) => (w.mark = "wetter")],
    ["fewer earlier years than it rests on", (w: Record<string, unknown>) => (w.years = 5)],
    ["a range that is not a number", (w: Record<string, unknown>) => (w.low_in = "0.3")],
    ["an unknown band", (w: Record<string, unknown>) => (w.band = "quartile")],
    [
      "an earlier year with no rain",
      (w: Record<string, unknown>) => ((w.earlier as Record<string, unknown>[])[0]!.rain_in = null),
    ],
  ])("refuses %s", (_case, spoil) => {
    const doc = structuredClone(golden("seasoned", "dashboard")) as {
      rain_tab: { window_strip: Record<string, unknown>[] };
    };
    spoil(doc.rain_tab.window_strip[0]!);

    expect(() => parseDashboard(doc)).toThrow("dashboard: rain_tab.window_strip malformed");
  });

  it("refuses a verdict outside its three words", () => {
    const doc = structuredClone(golden("seasoned", "dashboard")) as {
      rain_tab: { usual_to_date: Record<string, unknown> };
    };
    doc.rain_tab.usual_to_date.verdict = "wet";

    expect(() => parseDashboard(doc)).toThrow("dashboard: rain_tab.usual_to_date malformed");
  });
});
