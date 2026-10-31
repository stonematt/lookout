// @vitest-environment happy-dom
/**
 * The Rain tab (#83) against the golden snapshots (spec #66 seam 2), and its
 * headlines against worked examples.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { dayText, hourText, inches } from "../src/format.js";
import { zeroTicks as niceTicks } from "../src/charts/svg.js";
import { renderOverview } from "../src/overview/index.js";
import { renderRain } from "../src/rain/index.js";
import {
  dailyHeadline,
  dryHeadline,
  hoursHeadline,
  windowsNote,
  yearLeftOut,
  yearsHeadline,
} from "../src/rain/text.js";
import type { DashboardSnapshot, RainTab, WaterYearCurve } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { golden } from "./golden.js";

const raining = parseDashboard(golden("raining", "dashboard"));
const dry = parseDashboard(golden("dry", "dashboard"));

let panel: HTMLElement;

beforeEach(() => {
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const section = (id: string) => panel.querySelector(`[data-section=${id}]`)!;

/** A copy of a snapshot with its Rain tab changed. */
function withTab(snapshot: DashboardSnapshot, change: Partial<RainTab>): DashboardSnapshot {
  const copy = structuredClone(snapshot);
  copy.rain_tab = { ...copy.rain_tab, ...change };
  return copy;
}

/** A water year whose value at `place` is `value`. */
function year(water_year: number, place: number, value: number | null, prorated = false): WaterYearCurve {
  const cumulative = Array.from({ length: place + 1 }, () => value);
  return {
    water_year,
    first_day: `${water_year - 1}-10-01`,
    cumulative_in: cumulative,
    prorated: cumulative.map(() => prorated),
    status: {
      key: `water_year_${water_year}`,
      first_day: `${water_year - 1}-10-01`,
      last_day: `${water_year}-01-08`,
      days: place + 1,
      missing_days: 0,
      status: value === null ? "invalid" : prorated ? "prorated" : "valid",
      reason: value === null ? "before_archive" : prorated ? "gap" : null,
    },
    total_in: value,
    to_date_in: value,
  };
}

describe("the Rain tab's words", () => {
  it("writes inches, days and hours in the One Format Rule", () => {
    expect(inches(0.3)).toBe("0.30 in");
    expect(inches(12.44)).toBe("12.4 in");
    expect(dayText("2026-01-08")).toBe("Jan 8");
    expect(dayText("2026-01-08", true)).toBe("Jan 8, 2026");
    expect([0, 3, 12, 21].map(hourText)).toEqual(["midnight", "3 am", "noon", "9 pm"]);
  });

  it("leads the daily section with today's live total", () => {
    // Today 0.30 in live; the 7 days start before the archive, so no week.
    expect(dailyHeadline(raining)).toBe("0.30 in of rain so far today");
  });

  it("falls back to describing the chart when no window can be compared", () => {
    expect(dailyHeadline(dry)).toBe("Rain each day over the last 90 days");
  });

  it("adds the week when it can be compared, with ≈ when prorated", () => {
    const week = structuredClone(raining);
    const window = week.rain_tab.windows.find((w) => w.key === "7d")!;
    Object.assign(window.status, { status: "prorated", reason: "gap", missing_days: 0 });
    window.total_in = 0.85;

    expect(dailyHeadline(week)).toBe(
      "0.30 in of rain so far today, and ≈ 0.85 in over the last 7 days",
    );
  });

  it("says why windows are not compared, grouped by reason", () => {
    expect(windowsNote(raining)).toBe(
      "Not compared: 7 days, 30 days, 90 days, 365 days and water year " +
        "start before the archive began.",
    );
    expect(windowsNote(dry)).toBe(
      "Not compared: Today, yesterday, 7 days, 30 days, 90 days, 365 days " +
        "and water year are not archived yet.",
    );
  });

  it("states the dry spell with the calm lead's own numbers", () => {
    const calm = structuredClone(dry);
    Object.assign(calm.hero_lead.rules.find((r) => r.kind === "calm")!, { value: 12.5, reference: 4 });
    expect(dryHeadline(calm)).toBe(
      "12 d 12 h since it last rained, against a usual dry spell of 4 days",
    );
    // The 365 days are not archived, so there is no usual spell to compare.
    expect(dryHeadline(dry)).toBe("1 d 14 h since it last rained");
  });

  it("names the years compared, newest first, with ≈ where prorated", () => {
    const tab = {
      ...raining.rain_tab,
      day_index: 99,
      water_years: [
        year(2022, 364, 30.1),
        year(2023, 364, 18.5),
        year(2024, 364, null),
        year(2025, 364, 3.1, true),
        year(2026, 99, 4.2),
      ],
    };

    expect(yearsHeadline(tab)).toBe(
      "Water year 2026 has 4.20 in of rain since Oct 1, against " +
        "≈ 3.10 in (2025), 18.5 in (2023) and 30.1 in (2022) by this date",
    );
  });

  it("says which earlier years the headline leaves out at this date", () => {
    const cut = { ...year(2025, 364, 9.5), to_date_in: null };
    cut.cumulative_in = cut.cumulative_in.map((v, i) => (i < 50 ? v : null));
    const tab = withTab(raining, { day_index: 99, water_years: [year(2024, 364, 8), cut, year(2026, 99, 4.2)] });
    renderRain(panel, tab);

    expect(yearsHeadline(tab.rain_tab)).toBe(
      "Water year 2026 has 4.20 in of rain since Oct 1, against 8.00 in (2024) by this date",
    );
    expect(text(section("years").querySelector(".rain-source"))).toContain(
      "Not compared by this date, so left out of the headline: 2025.",
    );
  });

  it("does not compare a water year that cannot be", () => {
    expect(yearsHeadline(raining.rain_tab)).toBe("Water year 2026 cannot be compared yet");
    expect(yearLeftOut(raining.rain_tab.water_years[0]!)).toBe(
      "Water year 2026 is left out: it starts before the archive began.",
    );
  });

  it("names the wettest hour only when the 90 days can be compared", () => {
    expect(hoursHeadline(raining)).toBe("Rain by the hour over the last 90 days");
    const wet = withTab(raining, {
      recent: {
        ...raining.rain_tab.recent,
        wet_hours: 18.4,
        wettest_hour: { day: "2026-01-08", hour: 15, rain_in: 0.12 },
      },
    });
    Object.assign(wet.rain_tab.windows.find((w) => w.key === "90d")!.status, {
      status: "prorated",
      reason: "gap",
    });
    expect(hoursHeadline(wet)).toBe(
      "Rain fell in ≈ 18 hours of the last 90 days, the heaviest 0.12 in at 3 pm on Jan 8",
    );
  });

  it("picks round ticks above the largest value", () => {
    expect(niceTicks(0.36)).toEqual([0, 0.1, 0.2, 0.3, 0.4]);
    expect(niceTicks(9.2)).toEqual([0, 2.5, 5, 7.5, 10]);
  });
});

describe("the Rain tab", () => {
  it("shows a window by its rain status, not the archive's", () => {
    // A day with rows but no rain counter: the archive counts the 7 days,
    // the rain does not.
    const week = structuredClone(raining);
    Object.assign(week.gaps.windows.find((w) => w.key === "7d")!, { status: "valid", reason: null });
    const window = week.rain_tab.windows.find((w) => w.key === "7d")!;
    Object.assign(window.status, { status: "invalid", reason: "gap", missing_days: 1 });
    window.total_in = null;
    renderRain(panel, week);

    const dds = [...section("daily").querySelectorAll(".rain-windows dd")];
    expect(text(dds[2]!.querySelector(".gap-label"))).toBe("n/a");
    expect(windowsNote(week)).toContain("7 days");
  });

  it("renders its four sections in order", () => {
    renderRain(panel, raining);

    const ids = [...panel.querySelectorAll(".rain-section")].map((s) => s.getAttribute("data-section"));
    expect(ids).toEqual(["daily", "dry", "years", "hours"]);
    for (const host of panel.querySelectorAll(".rain-section")) {
      expect(host.querySelector("h2")?.textContent).toBeTruthy();
      expect(host.querySelector(".rain-source")?.textContent).toBeTruthy();
    }
  });

  it("shows today's rain as the same number as the Overview", () => {
    renderRain(panel, raining);
    const onRain = text(section("daily").querySelector(".rain-windows div dd"));
    const overview = document.createElement("div");
    renderOverview(overview, raining);
    const rightNow = [...overview.querySelectorAll(".right-now div")].find(
      (item) => text(item.querySelector("dt")) === "rain today",
    );

    expect(onRain).toBe("0.30 in");
    expect(text(rightNow?.querySelector("dd"))).toBe(onRain);
  });

  it("states days since rain as the same number as the Overview's calm lead", () => {
    renderRain(panel, dry);
    const overview = document.createElement("div");
    renderOverview(overview, dry);

    expect(dry.hero_lead.kind).toBe("calm");
    expect(text(overview)).toContain("1 d 14 h since it last rained");
    expect(text(section("dry").querySelector("h2"))).toBe("1 d 14 h since it last rained");
  });

  it("hatches each window that cannot be compared, with its reason", () => {
    renderRain(panel, raining);
    const cells = [...section("daily").querySelectorAll(".rain-windows dd")];

    expect(cells.map((dd) => dd.classList.contains("gap-hatch"))).toEqual([
      false, false, true, true, true, true, true,
    ]);
    expect(text(cells[2]?.querySelector(".gap-label"))).toBe("n/a");
    expect(cells[2]?.querySelector(".gap-sr")?.textContent).toContain("before the archive began");
  });

  it("hatches today while only the live total is known, and says so", () => {
    // A live reading from today, the archive not caught up.
    const live = structuredClone(dry);
    live.current.rain_today_in = 0;
    renderRain(panel, live);
    const today = section("daily").querySelector(".rain-windows dd")!;

    expect(text(today.querySelector(".gap-label"))).toBe("n/a");
    expect(today.querySelector(".gap-sr")?.textContent).toContain("does not reach this far yet");
    expect(text(section("daily").querySelector(".rain-source"))).toContain(
      "The live reading has 0.00 in today; only the live total is known",
    );
  });

  it("draws the days, marking today and where the record begins", () => {
    renderRain(panel, raining);
    const chart = section("daily").querySelector(".rain-chart")!;

    const bars = chart.querySelectorAll("rect.rain-bar");
    expect(bars.length).toBe(4); // Jan 4 dry; Jan 5, 7, 8 and today wet
    expect(chart.querySelectorAll("rect.rain-bar-now").length).toBe(1);
    const labels = [...chart.querySelectorAll("text")].map((t) => t.textContent);
    expect(labels).toContain("today 0.30 in");
    expect(labels).toContain("record begins Jan 4");
  });

  it("reads each day out from the keyboard", () => {
    renderRain(panel, raining);
    const chart = section("daily").querySelector<HTMLElement>(".rain-chart")!;
    const readout = section("daily").querySelector(".rain-readout")!;

    chart.dispatchEvent(new KeyboardEvent("keydown", { key: "End" }));
    expect(readout.textContent).toBe("Jan 8, 2026 (today, live): 0.30 in of rain.");
    chart.dispatchEvent(new KeyboardEvent("keydown", { key: "Home" }));
    expect(readout.textContent).toContain("before the archive began");
  });

  it("draws one line per comparable water year, the current one bold", () => {
    const tab = withTab(raining, {
      day_index: 99,
      water_years: [year(2025, 364, 12.5), year(2026, 99, 4.2)],
    });
    renderRain(panel, tab);
    const chart = section("years").querySelector(".rain-chart")!;

    expect(chart.querySelectorAll("polyline.rain-year-now").length).toBe(1);
    expect(chart.querySelectorAll("polyline.rain-year-0").length).toBe(1);
    const labels = [...chart.querySelectorAll("text")].map((t) => t.textContent);
    expect(labels).toContain("2026 now 4.20 in");
    expect(labels).toContain("2025 12.5 in");
  });

  it("leaves a water year out of the chart when it cannot be compared", () => {
    renderRain(panel, raining);

    expect(section("years").querySelectorAll("polyline").length).toBe(0);
    expect(text(section("years").querySelector(".gap-block .gap-label"))).toBe(
      "n/a: before the archive",
    );
    expect(text(section("years").querySelector(".rain-source"))).toContain(
      "Water year 2026 is left out",
    );
  });

  it("shades each hour by its rain and keys the shades", () => {
    renderRain(panel, raining);
    const grid = section("hours").querySelector(".rain-chart")!;

    // Today, 2 pm to 4 pm local: 0.12, 0.12 and 0.06 in.
    const wet = [...grid.querySelectorAll("rect.rain-cell:not(.rain-dry)")];
    expect(wet.length).toBeGreaterThan(0);
    expect(section("hours").querySelectorAll(".rain-key li").length).toBe(5);
    const readout = section("hours").querySelector(".rain-readout")!;
    grid.dispatchEvent(new KeyboardEvent("keydown", { key: "End" }));
    for (let i = 0; i < 14; i++) grid.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    expect(readout.textContent).toBe("Jan 8, 2026, 2 pm: 0.12 in of rain.");
    grid.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    grid.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    grid.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown" }));
    expect(readout.textContent).toBe("Jan 8, 2026, 5 pm: still to come.");
  });

  it("states the dry spell against the record", () => {
    renderRain(panel, dry);
    const stats = [...section("dry").querySelectorAll(".rain-dry-stats div")].map((item) => [
      text(item.querySelector("dt")),
      text(item.querySelector("dd")),
    ]);

    expect(stats[0]).toEqual(["Last rain", "Nov 17, 9:03 am"]);
    expect(stats[1]).toEqual(["Since it last rained", "1 d 14 h"]);
    expect(stats[2]?.[0]).toBe("Usual dry spell");
    expect(text(section("dry").querySelectorAll(".rain-dry-stats dd")[2]?.querySelector(".gap-label"))).toBe(
      "not archived yet",
    );
  });
});

describe("the Rain tab contract", () => {
  it("is in both goldens with its days parallel", () => {
    for (const snapshot of [dry, raining]) {
      const { recent } = snapshot.rain_tab;
      expect(recent.rain_in.length).toBe(90);
      expect(recent.hourly_in.every((hours) => hours.length === 24)).toBe(true);
      expect(snapshot.rain_tab.windows.map((w) => w.key)).toEqual(
        snapshot.gaps.windows.map((w) => w.key),
      );
    }
  });

  it("rejects a rain tab whose days are not parallel", () => {
    const broken = structuredClone(golden("raining", "dashboard")) as {
      rain_tab: { recent: { hourly_in: unknown[] } };
    };
    broken.rain_tab.recent.hourly_in.pop();

    expect(() => parseDashboard(broken)).toThrow(SnapshotContractError);
  });
});
