// @vitest-environment happy-dom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import { renderOverview, SECTIONS } from "../src/overview/index.js";
import { stripClaim, yearsText } from "../src/overview/temperature.js";
import type { DashboardSnapshot, TemperatureContext } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const dry = parseDashboard(golden("dry", "dashboard"));
const raining = parseDashboard(golden("raining", "dashboard"));

/** The `year` section `tests/test_snapshot_temperature.py` writes. */
const year = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../../tests/golden/temperature/year.json"), "utf8"),
) as TemperatureContext;

const withYear = (ctx: TemperatureContext = year): DashboardSnapshot => ({ ...dry, temperature_context: ctx });

let host: HTMLElement;

function show(snapshot: DashboardSnapshot): void {
  document.body.innerHTML = "<section id=host></section>";
  host = document.getElementById("host")!;
  SECTIONS.find((s) => s.id === "temperature-context")!.render(host, snapshot);
}

const text = (selector: string) => host.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim();
const all = (selector: string) => [...host.querySelectorAll(selector)];

beforeEach(() => show(withYear()));

describe("the temperature context's place", () => {
  it("follows the hero lead", () => {
    expect(SECTIONS.map((s) => s.id).slice(0, 3)).toEqual(["first-screen", "hero-lead", "temperature-context"]);
  });

  it("renders in the Overview for both golden scenarios", () => {
    for (const snapshot of [dry, raining]) {
      document.body.innerHTML = "<div id=panel></div>";
      const panel = document.getElementById("panel")!;
      renderOverview(panel, snapshot);
      expect(panel.querySelector("[data-section=temperature-context]")).not.toBeNull();
    }
  });

  it("parses the section and rejects a snapshot without it", () => {
    const { temperature_context: _gone, ...rest } = golden("dry", "dashboard") as Record<string, unknown>;
    expect(() => parseDashboard(rest)).toThrow(/temperature_context/);
  });
});

describe("temperature across periods", () => {
  it("draws one range bar per window, in order, with a now rule", () => {
    const rows = all("[data-part=periods-chart] .tc-row").map((el) => el.textContent);
    expect(rows).toEqual(["Today", "Yesterday", "7 days", "30 days", "90 days", "365 days"]);
    expect(all("[data-part=periods-chart] .tc-whisker")).toHaveLength(6);
    expect(all("[data-part=periods-chart] .tc-iqr")).toHaveLength(6);
    expect(all("[data-part=periods-chart] .tc-now")).toHaveLength(1);
    expect(text("[data-part=periods-chart] .tc-note")).toBe("now 55.0°");
    expect(all("[data-part=periods-chart] .tc-hatch")).toHaveLength(0);
  });

  it("names the start year of a window that spans a new year", () => {
    // "Jan 7 to Jan 6" read backwards (#147); the start year sets it right.
    const spans = all("[data-part=periods-chart] .tc-sub").map((el) => el.textContent);
    expect(spans).toEqual([
      "Jan 6",
      "Jan 5",
      "Dec 31, 2025 to Jan 6",
      "Dec 8, 2025 to Jan 6",
      "Oct 9, 2025 to Jan 6",
      "Jan 7, 2025 to Jan 6",
    ]);
  });

  it("words the finding from the 30-day window", () => {
    expect(text("[data-block=periods] [data-part=headline]")).toBe(
      "55.0° now is inside the middle half of readings over the last 30 days",
    );
  });

  it("hatches an invalid window with its reason and draws no value under it", () => {
    const ctx = structuredClone(year);
    const week = ctx.periods[2]!;
    week.status = { ...week.status, status: "invalid", reason: "gap", missing_days: 5 };
    Object.assign(week, { low: null, q1: null, q3: null, high: null, readings: 0 });
    show(withYear(ctx));

    expect(all("[data-part=periods-chart] .tc-hatch")).toHaveLength(1);
    expect(all("[data-part=periods-chart] .tc-reason").map((el) => el.textContent)).toEqual(["n/a: gap in record"]);
    expect(all("[data-part=periods-chart] .tc-whisker")).toHaveLength(5);
    expect(text("[data-block=periods] .ov-sr")).toContain("7 days: Dec 31, 2025 to Jan 6, 2026: 5 of 7 days");
  });

  it("marks a prorated window with ≈ on its values", () => {
    const ctx = structuredClone(year);
    const month = ctx.periods[3]!;
    month.status = { ...month.status, status: "prorated", reason: "gap", missing_days: 2 };
    show(withYear(ctx));

    const values = all("[data-part=periods-chart] .tc-value").map((el) => el.textContent!);
    expect(values.filter((v) => v.startsWith("≈ "))).toHaveLength(2);
  });

  it("hatches every row when the archive lags, as in the dry golden", () => {
    show(dry);

    expect(all("[data-part=periods-chart] .tc-hatch")).toHaveLength(6);
    expect(all("[data-part=periods-chart] .tc-reason").every((el) => el.textContent === "not archived yet")).toBe(true);
    expect(all("[data-part=periods-chart] .tc-whisker")).toHaveLength(0);
  });
});

describe("the ten-day list", () => {
  it("names the compared years, newest first", () => {
    expect(text("[data-block=ten-days] [data-part=headline]")).toBe(
      "The last ten days, against the usual for their dates in 2025, 2024 and 2023",
    );
    expect(yearsText([2025, 2024])).toBe("2025 and 2024");
  });

  it("lists ten days, today first, each with a usual band", () => {
    const labels = all("[data-part=days-chart] .tc-row").map((el) => el.textContent);
    expect(labels).toHaveLength(10);
    expect(labels[0]).toBe("Tue, Jan 6");
    expect(labels.at(-1)).toBe("Sun, Dec 28");
    expect(all("[data-part=days-chart] .tc-day")).toHaveLength(10);
    expect(all("[data-part=days-chart] .tc-usual")).toHaveLength(10);
  });

  it("hatches a day the archive cannot count and says so in words", () => {
    const ctx = structuredClone(year);
    const day = ctx.ten_days[5]!;
    day.status = { ...day.status, status: "invalid", reason: "gap", missing_days: 1 };
    Object.assign(day, { low: null, high: null });
    show(withYear(ctx));

    expect(all("[data-part=days-chart] .tc-hatch")).toHaveLength(1);
    expect(all("[data-part=days-chart] .tc-day")).toHaveLength(9);
    expect(text("[data-block=ten-days] .ov-sr")).toContain("the day is missing from the record");
  });

  it("has nothing to set the days against in a young archive", () => {
    const ctx = structuredClone(year);
    ctx.ten_day_years = [];
    for (const d of ctx.ten_days) Object.assign(d, { usual_low: null, usual_high: null, usual_years: [], usual_days: 0 });
    show(withYear(ctx));

    expect(text("[data-block=ten-days] [data-part=headline]")).toBe("The last ten days, with no earlier year to set them against");
    expect(all("[data-part=days-chart] .tc-usual")).toHaveLength(0);
  });
});

describe("now against normal", () => {
  it("states the temperature claim with the percentage and the sample", () => {
    expect(text("[data-strip=temperature] [data-part=claim]")).toBe(
      "Cooler than 80% of 30 past days within a week of this date in 2025 and 2024, at this hour.",
    );
    expect(all("[data-strip=temperature] .tc-dot")).toHaveLength(30);
    expect(text("[data-strip=temperature] .tc-note")).toBe("now 55.0°");
  });

  it("states the rain claim and draws today so far", () => {
    expect(text("[data-strip=rain] [data-part=claim]")).toMatch(/^Dry so far today, as on 90% of 30 past days/);
    expect(text("[data-strip=rain] .tc-note")).toBe("today 0.00 in so far");
    expect(all("[data-strip=rain] .tc-dot")).toHaveLength(30);
  });

  it("names the years in the heading, newest first", () => {
    expect(text("[data-block=strips] [data-part=headline]")).toBe("Now against normal for this date in 2025 and 2024");
  });

  it("says why a strip cannot be drawn rather than drawing one", () => {
    const strip = { ...year.strips[0]!, missing: "no_comparison" as const, values: [], below_pct: null, equal_pct: null, above_pct: null, years: [] };
    expect(stripClaim(strip)).toBe("No earlier year is complete enough to set this against.");
    const ctx = structuredClone(year);
    ctx.strips = [strip, ctx.strips[1]!];
    show(withYear(ctx));

    expect(all("[data-strip=temperature] .tc-dot")).toHaveLength(0);
    expect(text("[data-strip=temperature] [data-part=claim]")).toBe(stripClaim(strip));
  });

  it("says the no-comparison sentence once when both strips lack one", () => {
    const none = { missing: "no_comparison" as const, values: [], below_pct: null, equal_pct: null, above_pct: null, years: [] };
    const ctx = structuredClone(year);
    ctx.strips = [{ ...year.strips[0]!, ...none }, { ...year.strips[1]!, ...none }];
    show(withYear(ctx));

    const claims = all("[data-block=strips] [data-part=claim]").map((el) => el.textContent);
    expect(claims).toEqual(["No earlier year is complete enough to set this against."]);
  });
});
