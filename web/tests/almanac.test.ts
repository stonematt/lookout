// @vitest-environment happy-dom
/**
 * The Overview's almanac grid and rolling-window rain table (#82; spec #66
 * seam 2). The October almanac is the one `tests/test_snapshot_almanac.py`
 * writes: a station from Aug 20, 2024 to noon on Oct 12, 2025, whose 30-day
 * window and season so far both cross Oct 1.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";

import { almanacGrid, rollingRain } from "../src/overview/almanac.js";
import {
  cellText,
  gridHeadline,
  monthSentence,
  rampBounds,
  rampStep,
  rollingHeadline,
  seasonSentence,
  seasonSoFar,
} from "../src/overview/almanac-text.js";
import { SECTIONS, renderOverview } from "../src/overview/index.js";
import type { Almanac, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { ratio, themeTokens } from "./contrast.js";
import { golden } from "./golden.js";

const ROOT = resolve(import.meta.dirname, "../..");
const october: Almanac = JSON.parse(
  readFileSync(`${ROOT}/tests/golden/almanac/october.json`, "utf8"),
).almanac;
const raining = parseDashboard(golden("raining", "dashboard"));
const dry = parseDashboard(golden("dry", "dashboard"));

/** The raining golden with the October almanac in it. */
function withAlmanac(almanac: Almanac): DashboardSnapshot {
  return { ...raining, almanac };
}

/** A deep copy to edit. */
function copy(almanac: Almanac): Almanac {
  return JSON.parse(JSON.stringify(almanac));
}

let host: HTMLElement;

beforeEach(() => {
  document.body.innerHTML = "<section id=host></section>";
  host = document.getElementById("host")!;
});

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const cell = (month: string) => host.querySelector<HTMLElement>(`[data-month="${month}"]`)!;

describe("the almanac's words", () => {
  it("names the newest month against the same month in earlier years", () => {
    expect(gridHeadline(october)).toBe(
      "September 2025 had 0.70 in of rain, against 0.50 in (2024)",
    );
  });

  it("says a dry month had no rain", () => {
    const almanac = copy(october);
    almanac.years[1]!.months[11]!.rain_in = 0;

    expect(gridHeadline(almanac)).toBe("September 2025 had no rain, against 0.50 in (2024)");
  });

  it("says an earlier dry month had no rain", () => {
    const almanac = copy(october);
    almanac.years[2]!.months[11]!.rain_in = 0;

    expect(gridHeadline(almanac)).toBe(
      "September 2025 had 0.70 in of rain, against no rain (2024)",
    );
  });

  it("names the last 30 days against the same days a year before", () => {
    expect(rollingHeadline(october)).toBe(
      "1.00 in of rain in the last 30 days, against 0.75 in (2024) over the same days",
    );
  });

  it("falls back to the week when the month cannot be compared", () => {
    const almanac = copy(october);
    almanac.rolling.find((w) => w.key === "30d")!.years[0]!.rain_in = null;

    expect(rollingHeadline(almanac)).toBe(
      "0.10 in of rain in the last 7 days, against 0.00 in (2024) over the same days",
    );
  });

  it("names an earlier season by its water year, not the calendar year", () => {
    // Oct 1 to 12, 2024 is water year 2025 (#147).
    const almanac = copy(october);
    for (const key of ["30d", "7d"])
      almanac.rolling.find((w) => w.key === key)!.years[0]!.rain_in = null;

    expect(rollingHeadline(almanac)).toBe(
      "0.30 in of rain since Oct 1, against 0.25 in (water year 2025) over the same days",
    );
  });

  it("writes the current month's rain so far, and why it is not compared", () => {
    const current = october.years[0]!.months[0]!;

    expect(monthSentence(current)).toBe(
      "October 2025 so far: 0.30 in of rain through Oct 12. " +
        "Oct 1 to Oct 31, 2025: it is not over yet, so it is not compared.",
    );
  });

  it("writes a complete month and a month before the archive", () => {
    const [, wy2025, wy2024] = october.years;

    expect(monthSentence(wy2025!.months[2]!)).toBe(
      "December 2024: 1.00 in of rain. Dec 1 to Dec 31, 2024: every day is in the record.",
    );
    expect(monthSentence(wy2024!.months[10]!)).toBe(
      "August 2024: Aug 1 to Aug 31, 2024: it starts before the archive began, so it is not compared.",
    );
  });

  it("writes a season's total, and the running season so far", () => {
    const [wy2026, wy2025] = october.years;

    expect(seasonSentence(wy2025!, null)).toBe(
      "Water year 2025, Oct 1 to Sep 30: 1.95 in of rain.",
    );
    expect(seasonSentence(wy2026!, seasonSoFar(october))).toBe(
      "Water year 2026 so far: 0.30 in of rain since Oct 1.",
    );
  });

  it("steps the ramp on a square-root scale", () => {
    // sqrt(v / 10) * 5: 0.39 in is under the second step, 0.41 in over it.
    expect([0, 0.39, 0.41, 1.7, 3.7, 6.5, 10].map((v) => rampStep(v, 10))).toEqual([
      0, 0, 1, 2, 3, 4, 4,
    ]);
    expect(rampBounds(10)).toEqual([0, 0.4, 1.6, 3.6, 6.4].map((v) => expect.closeTo(v, 9)));
  });

  it("fits a value in a cell", () => {
    expect([0.04, 0.7, 4.24, 12.3].map(cellText)).toEqual(["0.04", "0.70", "4.2", "12"]);
  });
});

describe("the almanac grid", () => {
  beforeEach(() => almanacGrid.render(host, withAlmanac(october)));

  it("has water years as rows, newest first, and months Oct to Sep", () => {
    const years = [...host.querySelectorAll("tbody th")].map(text);
    const months = [...host.querySelectorAll("thead .alm-long")].map(text);

    expect(years).toEqual(["2026", "2025", "2024"]);
    expect(months).toEqual([
      "Oct",
      "Nov",
      "Dec",
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
    ]);
  });

  it("fills a month from the ramp with its value", () => {
    expect(cell("2024-12").dataset.step).toBe("4"); // the wettest month
    expect(text(cell("2024-12").querySelector(".alm-value"))).toBe("1.0");
    expect(cell("2024-11").dataset.step).toBe("0"); // dry, not missing
    expect(text(cell("2024-11"))).toBe("0.00");
  });

  it("gives every month its gap status", () => {
    expect(cell("2025-10").dataset.kind).toBe("current");
    expect(text(cell("2025-10").querySelector(".alm-value"))).toBe("0.30");
    // Filled from the ramp by its total so far: no hatch.
    expect(cell("2025-10").dataset.step).toBeDefined();
    expect(cell("2025-10").classList).not.toContain("gap-hatch");
    expect(cell("2024-08").dataset.kind).toBe("before");
    expect(cell("2024-08").querySelector(".alm-value")).toBeNull();
    expect(cell("2024-12").dataset.kind).toBe("valid");
    // Nov 2025 onward has not begun: an empty slot, not a button.
    const row = host.querySelector('tr[data-water-year="2026"]')!;
    expect(row.querySelectorAll("button.alm-cell")).toHaveLength(1);
  });

  it("keeps a current month with no total so far hatched, with no ramp step", () => {
    const almanac = copy(october);
    Object.assign(almanac.years[0]!.months[0]!, { to_date_in: null });
    host.replaceChildren();
    almanacGrid.render(host, withAlmanac(almanac));
    expect(cell("2025-10").dataset.kind).toBe("current");
    expect(cell("2025-10").dataset.step).toBeUndefined();
    expect(cell("2025-10").querySelector(".alm-value")).toBeNull();
  });

  it("hatches an invalid month and marks a prorated one ≈", () => {
    const almanac = copy(october);
    const months = almanac.years[1]!.months;
    Object.assign(months[1]!.status, { status: "prorated", reason: "gap", missing_days: 2 });
    Object.assign(months[3]!, { rain_in: null });
    Object.assign(months[3]!.status, { status: "invalid", reason: "gap", missing_days: 5 });
    host.replaceChildren();
    almanacGrid.render(host, withAlmanac(almanac));

    expect(text(cell("2024-11").querySelector(".alm-value"))).toBe("≈0.00");
    expect(cell("2025-01").classList).toContain("gap-hatch");
    // A gap inside a shown month stays visible, worded without "n/a" (#470).
    expect(text(cell("2025-01").querySelector(".gap-label"))).toBe("gap");
    expect(cell("2025-01").title).toBe(
      "Jan 1 to Jan 31, 2025: 5 of 31 days are missing from the record, so it is not compared.",
    );
  });

  it("shows each season's total, and the running one so far", () => {
    const season = (year: number) =>
      host.querySelector(`tr[data-water-year="${year}"] td.alm-season`) as HTMLElement;

    expect(text(season(2025))).toBe("1.95");
    expect(season(2026).dataset.kind).toBe("current");
    expect(text(season(2026))).toBe("0.30");
    // 2024's season starts before the record: not shown, so not hatched (#470).
    expect(season(2024).classList).not.toContain("gap-hatch");
    expect(text(season(2024))).toBe("");
  });

  it("starts the readout on the newest month with a value", () => {
    const readout = host.querySelector("[data-part=readout]")!;

    expect(text(readout)).toBe(
      "September 2025: 0.70 in of rain. Sep 1 to Sep 30, 2025: every day is in the record. " +
        "Water year 2025, Oct 1 to Sep 30: 1.95 in of rain.",
    );
    expect(readout.getAttribute("aria-live")).toBe("polite");
    expect(cell("2025-09").tabIndex).toBe(0);
    expect(cell("2025-09").hasAttribute("data-selected")).toBe(true);
  });

  it("writes a month's sentence when it is tapped", () => {
    cell("2025-10").click();

    expect(text(host.querySelector("[data-part=readout]"))).toBe(
      "October 2025 so far: 0.30 in of rain through Oct 12. " +
        "Oct 1 to Oct 31, 2025: it is not over yet, so it is not compared. " +
        "Water year 2026 so far: 0.30 in of rain since Oct 1.",
    );
  });

  it("moves through the months with the arrow keys, one tab stop", () => {
    const press = (key: string) =>
      document.activeElement!.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
    cell("2025-09").focus();

    press("ArrowLeft");
    expect(document.activeElement).toBe(cell("2025-08"));
    press("ArrowUp");
    expect(document.activeElement).toBe(cell("2025-10")); // the nearest month above
    press("ArrowDown");
    press("Home");
    expect(document.activeElement).toBe(cell("2024-10"));
    expect(host.querySelectorAll('button.alm-cell[tabindex="0"]')).toHaveLength(1);
  });

  it("names every mark with its full sentence", () => {
    expect(cell("2024-12").getAttribute("aria-label")).toBe(
      "December 2024: 1.00 in of rain. Dec 1 to Dec 31, 2024: every day is in the record.",
    );
  });
});

describe("the almanac grid with no rain to scale", () => {
  it("drops the ramp's numbers and starts on the current month", () => {
    const almanac = copy(october);
    almanac.years = [almanac.years[0]!];
    almanacGrid.render(host, withAlmanac(almanac));

    expect(host.querySelector(".alm-key-ramp")).toBeNull();
    expect(cell("2025-10").hasAttribute("data-selected")).toBe(true);
  });
});

describe("the rolling-window table", () => {
  beforeEach(() => rollingRain.render(host, withAlmanac(october)));

  const row = (key: string) =>
    [...host.querySelectorAll(`tr[data-window="${key}"] > *`)].map((el) => text(el));

  it("has one column per year, newest first", () => {
    expect([...host.querySelectorAll("thead th")].map(text)).toEqual(["Window", "2025", "2024"]);
  });

  it("shows the week, month, season and year against the same days a year before", () => {
    expect(row("7d")).toEqual(["7 days", "0.10", "0.00"]);
    expect(row("30d")).toEqual(["30 days", "1.00", "0.75"]);
    expect(row("water_year")).toEqual(["Since Oct 1", "0.30", "0.25"]);
  });

  it("names the season row by water year under the calendar-year columns (#156)", () => {
    // Oct 12: each season so far ends in the next water year.
    const head = host.querySelector("tr[data-part=water-years]")!;
    expect([...head.children].map(text)).toEqual(["Water year", "2026", "2025"]);
    expect(head.nextElementSibling).toBe(host.querySelector('tr[data-window="water_year"]'));
    for (const th of head.querySelectorAll("th:not(:first-child)")) {
      expect(th.getAttribute("scope")).toBe("col");
    }
    // A season cell is read by its row and its water year, not the calendar column.
    const row = host.querySelector('tr[data-window="water_year"] th')!;
    const ids = [...head.querySelectorAll("th")].map((th) => th.id);
    const cells = [
      ...host.querySelectorAll<HTMLTableCellElement>('tr[data-window="water_year"] td'),
    ];
    expect(row.id).not.toBe("");
    expect(cells.map((td) => td.getAttribute("headers"))).toEqual([
      `${row.id} ${ids[1]}`,
      `${row.id} ${ids[2]}`,
    ]);
  });

  it("needs no water-year row when the season ends in the column's year", () => {
    const march = copy(october);
    const season = march.rolling.find((w) => w.key === "water_year")!;
    for (const y of season.years) y.water_year = y.year;
    host.replaceChildren();
    rollingRain.render(host, withAlmanac(march));

    expect(host.querySelector("tr[data-part=water-years]")).toBeNull();
  });

  it("leaves a window that reaches before the record empty, not hatched (#470)", () => {
    const earlier = host.querySelectorAll('tr[data-window="365d"] td')[1] as HTMLElement;

    expect(earlier.classList).not.toContain("gap-hatch");
    expect(text(earlier)).toBe("");
  });

  it("hatches a gap inside a shown window, worded without n/a", () => {
    const gappy = copy(october);
    const cell = gappy.rolling.find((w) => w.key === "30d")!.years[1]!;
    Object.assign(cell, { rain_in: null });
    Object.assign(cell.status, { status: "invalid", reason: "gap", missing_days: 4 });
    host.replaceChildren();
    rollingRain.render(host, withAlmanac(gappy));

    const td = host.querySelectorAll('tr[data-window="30d"] td')[1] as HTMLElement;
    expect(td.classList).toContain("gap-hatch");
    expect(text(td.querySelector(".gap-label"))).toBe("gap");
  });

  it("says when each window ends, in its caption and method, with no standfirst", () => {
    expect(host.querySelector("[data-part=standfirst]")).toBeNull();
    expect(text(host.querySelector("caption"))).toContain("ending Oct 12");
    expect(text(host.querySelector("[data-part=source]"))).toContain("ending Oct 12");
  });
});

describe("as tiles of their own", () => {
  it("left the Overview with the accordion (#472), for the rain zoom", () => {
    const ids = SECTIONS.map((s) => s.id);

    expect(ids).not.toContain("almanac");
    expect(ids).not.toContain("rolling-rain");
  });

  it.each([
    ["dry", dry],
    ["raining", raining],
  ])("renders the grid from the %s golden, and no rolling table it cannot cover", (_, snapshot) => {
    document.body.innerHTML = "<div id=panel></div>";
    const panel = document.getElementById("panel")!;
    renderOverview(panel, snapshot, [almanacGrid, rollingRain]);

    const section = panel.querySelector("[data-section=almanac]")!;
    expect(text(section.querySelector("[data-part=finding]"))).toBeTruthy();
    expect(section.querySelector("table")).not.toBeNull();
    // A young record covers no rolling window yet (#470).
    expect(panel.querySelector("[data-section=rolling-rain]")).toBeNull();
  });

  it("rejects an almanac row that is not twelve months", () => {
    const doc = golden("raining", "dashboard") as { almanac: Almanac };
    doc.almanac.years = [{ ...october.years[0]!, months: [] }];

    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });
});

describe("text on a ramp cell", () => {
  const css = readFileSync(`${ROOT}/web/src/overview/almanac.css`, "utf8");

  it.each([0, 1, 2, 3, 4])(
    "rain-%i takes whichever cell ink has more contrast, by day and by night",
    (step) => {
      // The cell takes its step's ink token; tokens.css sets it per theme.
      const rule = new RegExp(
        `\\.alm-cell\\[data-step="${step}"\\] \\{[^}]*color: var\\(--on-rain-${step}\\);`,
      );
      expect(css).toMatch(rule);
      for (const theme of ["light", "dark"] as const) {
        const t = themeTokens(theme);
        const fill = t[`rain-${step}`]!;
        const ink = t[`on-rain-${step}`]!;
        const other = ink === t["on-ramp-pale"] ? t["on-ramp-deep"]! : t["on-ramp-pale"]!;
        expect(ratio(fill, ink)).toBeGreaterThan(ratio(fill, other));
        expect(ratio(fill, ink)).toBeGreaterThanOrEqual(4.5);
      }
    },
  );
});
