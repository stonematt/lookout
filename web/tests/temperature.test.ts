// @vitest-environment happy-dom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import { renderOverview } from "../src/overview/index.js";
import {
  againstNormal,
  stripClaim,
  temperaturePeriods,
  tenDays,
  yearsText,
} from "../src/overview/temperature.js";
import type { DashboardSnapshot, TemperatureContext } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const dry = parseDashboard(golden("dry", "dashboard"));
const raining = parseDashboard(golden("raining", "dashboard"));

/** The `year` section `tests/test_snapshot_temperature.py` writes. */
const year = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../../tests/golden/temperature/year.json"), "utf8"),
) as TemperatureContext;

const withYear = (ctx: TemperatureContext = year): DashboardSnapshot => ({
  ...dry,
  temperature_context: ctx,
});

let host: HTMLElement;

/**
 * The three temperature sections, which left the Overview with the
 * accordion (#472) and stay as the temperature zoom's content, rendered in
 * a tile grid of their own.
 */
function show(snapshot: DashboardSnapshot): void {
  document.body.innerHTML = "<div id=host></div>";
  host = document.getElementById("host")!;
  renderOverview(host, snapshot, [temperaturePeriods, againstNormal, tenDays]);
}

const text = (selector: string) =>
  host.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim();
const all = (selector: string) => [...host.querySelectorAll(selector)];

beforeEach(() => show(withYear()));

describe("the temperature context", () => {
  it("renders for both golden scenarios", () => {
    for (const snapshot of [dry, raining]) {
      show(snapshot);
      for (const id of ["temperature-periods", "against-normal", "ten-days"]) {
        expect(host.querySelector(`[data-section=${id}]`)).not.toBeNull();
      }
    }
  });

  it("parses the section and rejects a snapshot without it", () => {
    const { temperature_context: _gone, ...rest } = golden("dry", "dashboard") as Record<
      string,
      unknown
    >;
    expect(() => parseDashboard(rest)).toThrow(/temperature_context/);
  });
});

describe("temperature across periods", () => {
  it("draws one range bar per window, in order, then Full history, with a now rule", () => {
    const rows = all("[data-part=periods-chart] .tc-row").map((el) => el.textContent);
    expect(rows).toEqual([
      "Today",
      "Yesterday",
      "7 days",
      "30 days",
      "90 days",
      "365 days",
      "Full history",
    ]);
    expect(all("[data-part=periods-chart] .tc-whisker")).toHaveLength(7);
    expect(all("[data-part=periods-chart] .tc-iqr")).toHaveLength(7);
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
      "Dec 1, 2023 to Jan 6",
    ]);
  });

  it("words the finding from the 30-day window", () => {
    expect(text("[data-block=periods] [data-part=finding]")).toBe(
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
    // A gap inside a shown window stays visible, worded without "n/a" (#470).
    expect(all("[data-part=periods-chart] .tc-reason").map((el) => el.textContent)).toEqual([
      "gap in record",
    ]);
    expect(all("[data-part=periods-chart] .tc-whisker")).toHaveLength(6);
    expect(text("[data-block=periods] .ov-sr")).toContain(
      "7 days: Dec 31, 2025 to Jan 6, 2026: 5 of 7 days",
    );
  });

  it("marks a prorated window with ≈ on its values", () => {
    const ctx = structuredClone(year);
    const month = ctx.periods[3]!;
    month.status = { ...month.status, status: "prorated", reason: "gap", missing_days: 2 };
    show(withYear(ctx));

    const values = all("[data-part=periods-chart] .tc-value").map((el) => el.textContent!);
    expect(values.filter((v) => v.startsWith("≈\u00a0"))).toHaveLength(2);
  });

  it("writes a below-zero low with the true minus, in the chart and its words", () => {
    const ctx = structuredClone(year);
    const month = ctx.periods[3]!;
    Object.assign(month, { low: -12.4, q1: -3 });
    month.status = { ...month.status, status: "prorated", reason: "gap", missing_days: 2 };
    show(withYear(ctx));

    const values = all("[data-part=periods-chart] .tc-value").map((el) => el.textContent!);
    expect(values).toContain("≈\u00a0−12°");
    const spoken = text("[data-block=periods] .ov-sr")!;
    expect(spoken).toContain("low −12.4°");
    expect(spoken).toContain("middle half −3.0° to");
    expect(`${values.join(" ")} ${spoken}`).not.toMatch(/-\d/);
  });

  it("draws no chart when the archive holds none of its rows, as in the dry golden (#606)", () => {
    show(dry);

    // Today, yesterday and Full history are all not archived yet: no chart,
    // not three hatched rows. The finding and the spoken rows still say so.
    expect(all("[data-part=periods-chart]")).toHaveLength(0);
    expect(all("[data-block=periods] .tc-hatch")).toHaveLength(0);
    expect(text("[data-block=periods] [data-part=finding]")).toContain("no window complete");
    expect(text("[data-block=periods] .ov-sr")).toContain(
      "the archive does not reach this far yet",
    );
  });

  it("keeps a not-archived row hatched in a chart that shows other rows", () => {
    const ctx = structuredClone(year);
    const today = ctx.periods[0]!;
    today.status = { ...today.status, status: "invalid", reason: "not_archived", missing_days: 1 };
    Object.assign(today, { low: null, q1: null, q3: null, high: null, readings: 0 });
    show(withYear(ctx));

    expect(all("[data-part=periods-chart] .tc-hatch")).toHaveLength(1);
    expect(all("[data-part=periods-chart] .tc-reason").map((el) => el.textContent)).toEqual([
      "not archived yet",
    ]);
    expect(all("[data-part=periods-chart] .tc-whisker")).toHaveLength(6);
  });
});

describe("the ten-day list", () => {
  it("names the compared years, newest first", () => {
    expect(text("[data-block=ten-days] [data-part=finding]")).toBe(
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

  it("puts a day's label above a full-width bar on a phone", () => {
    const labelX = () => all("[data-part=days-chart] .tc-row").map((el) => el.getAttribute("x"));
    expect(labelX().every((x) => x === "2")).toBe(true);

    const width = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
      configurable: true,
      get: () => 390,
    });
    try {
      show(withYear());
      expect(labelX().every((x) => x === "0")).toBe(true);
      const bar = host.querySelector("[data-part=days-chart] .tc-day")!;
      const label = host.querySelector("[data-part=days-chart] .tc-row")!;
      const barTop = Number(bar.getAttribute("y"));
      expect(Number(label.getAttribute("y"))).toBeLessThan(barTop);
      // the plot starts at the left edge: no label column
      expect(host.querySelector("[data-part=days-chart] g")?.getAttribute("transform")).toBe(
        "translate(0,8)",
      );
    } finally {
      if (width) Object.defineProperty(HTMLElement.prototype, "clientWidth", width);
      else delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientWidth;
    }
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
    for (const d of ctx.ten_days)
      Object.assign(d, { usual_low: null, usual_high: null, usual_years: [], usual_days: 0 });
    show(withYear(ctx));

    expect(text("[data-block=ten-days] [data-part=finding]")).toBe(
      "The last ten days, with no earlier year to set them against",
    );
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
    expect(text("[data-strip=rain] [data-part=claim]")).toMatch(
      /^Dry so far today, as on 90% of 30 past days/,
    );
    expect(text("[data-strip=rain] .tc-note")).toBe("today 0.00 in so far");
    expect(all("[data-strip=rain] .tc-dot")).toHaveLength(30);
  });

  it("names the years in the heading, newest first", () => {
    expect(text("[data-block=strips] [data-part=finding]")).toBe(
      "Now against usual for this date in 2025 and 2024",
    );
  });

  it("says why a strip cannot be drawn rather than drawing one", () => {
    const strip = {
      ...year.strips[0]!,
      missing: "no_comparison" as const,
      values: [],
      below_pct: null,
      equal_pct: null,
      above_pct: null,
      years: [],
    };
    expect(stripClaim(strip)).toBe("No earlier year is complete enough to set this against.");
    const ctx = structuredClone(year);
    ctx.strips = [strip, ctx.strips[1]!];
    show(withYear(ctx));

    expect(all("[data-strip=temperature] .tc-dot")).toHaveLength(0);
    expect(text("[data-strip=temperature] [data-part=claim]")).toBe(stripClaim(strip));
  });

  it("says the no-comparison sentence once when both strips lack one", () => {
    const none = {
      missing: "no_comparison" as const,
      values: [],
      below_pct: null,
      equal_pct: null,
      above_pct: null,
      years: [],
    };
    const ctx = structuredClone(year);
    ctx.strips = [
      { ...year.strips[0]!, ...none },
      { ...year.strips[1]!, ...none },
    ];
    show(withYear(ctx));

    const claims = all("[data-block=strips] [data-part=claim]").map((el) => el.textContent);
    expect(claims).toEqual(["No earlier year is complete enough to set this against."]);
  });
});
