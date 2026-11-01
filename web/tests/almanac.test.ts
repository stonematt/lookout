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
    expect(gridHeadline(october)).toBe("September 2025 had 0.70 in of rain, against 0.50 in (2024)");
  });

  it("says a dry month had no rain", () => {
    const almanac = copy(october);
    almanac.years[1]!.months[11]!.rain_in = 0;

    expect(gridHeadline(almanac)).toBe("September 2025 had no rain, against 0.50 in (2024)");
  });

  it("says an earlier dry month had no rain", () => {
    const almanac = copy(october);
    almanac.years[2]!.months[11]!.rain_in = 0;

    expect(gridHeadline(almanac)).toBe("September 2025 had 0.70 in of rain, against no rain (2024)");
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
    for (const key of ["30d", "7d"]) almanac.rolling.find((w) => w.key === key)!.years[0]!.rain_in = null;

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

    expect(seasonSentence(wy2025!, null)).toBe("Water year 2025, Oct 1 to Sep 30: 1.95 in of rain.");
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
    expect(months).toEqual(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]);
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
    expect(cell("2024-08").dataset.kind).toBe("before");
    expect(cell("2024-08").querySelector(".alm-value")).toBeNull();
    expect(cell("2024-12").dataset.kind).toBe("valid");
    // Nov 2025 onward has not begun: an empty slot, not a button.
    const row = host.querySelector('tr[data-water-year="2026"]')!;
    expect(row.querySelectorAll("button.alm-cell")).toHaveLength(1);
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
    expect(text(cell("2025-01").querySelector(".gap-label"))).toBe("n/a");
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
    expect(season(2024).classList).toContain("gap-hatch");
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

  it("hatches a window that reaches before the archive", () => {
    const earlier = host.querySelectorAll('tr[data-window="365d"] td')[1] as HTMLElement;

    expect(earlier.classList).toContain("gap-hatch");
    expect(text(earlier.querySelector(".gap-label"))).toBe("n/a");
  });

  it("says when each window ends", () => {
    expect(text(host.querySelector("[data-part=standfirst]"))).toContain("ending Oct 12");
  });
});

describe("on the Overview", () => {
  it("follows the hero lead, the grid before the table", () => {
    const ids = SECTIONS.map((s) => s.id);

    expect(ids.indexOf("almanac")).toBeGreaterThan(ids.indexOf("hero-lead"));
    expect(ids.indexOf("rolling-rain")).toBe(ids.indexOf("almanac") + 1);
  });

  it.each([
    ["dry", dry],
    ["raining", raining],
  ])("renders both sections from the %s golden", (_, snapshot) => {
    document.body.innerHTML = "<div id=panel></div>";
    const panel = document.getElementById("panel")!;
    renderOverview(panel, snapshot);

    for (const id of ["almanac", "rolling-rain"]) {
      const section = panel.querySelector(`[data-section=${id}]`)!;
      expect(text(section.querySelector("[data-part=headline]"))).toBeTruthy();
      expect(section.querySelector("table")).not.toBeNull();
    }
  });

  it("rejects an almanac row that is not twelve months", () => {
    const doc = golden("raining", "dashboard") as { almanac: Almanac };
    doc.almanac.years = [{ ...october.years[0]!, months: [] }];

    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });
});

describe("text on a ramp cell", () => {
  /** Every `--name: #hex` in a stylesheet, in order. */
  function values(file: string, name: string): string[] {
    const css = readFileSync(`${ROOT}/web/src/${file}`, "utf8");
    return [...css.matchAll(new RegExp(`--${name}:\\s*(#[0-9a-f]{6})`, "g"))].map((m) => m[1]!);
  }

  function luminance(hex: string): number {
    const [r, g, b] = [1, 3, 5].map((i) => {
      const c = parseInt(hex.slice(i, i + 2), 16) / 255;
      return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
  }

  function contrast(a: string, b: string): number {
    const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (hi! + 0.05) / (lo! + 0.05);
  }

  it.each([0, 1, 2, 3, 4])("rain-%i takes whichever ink has more contrast, by day and by night", (step) => {
    const fills = values("styles/tokens.css", `rain-${step}`);
    const inks = values("overview/almanac.css", `alm-on-${step}`);
    // Light first; then the OS dark block and the explicit dark block.
    expect(fills).toHaveLength(3);
    expect(inks).toHaveLength(3);
    fills.forEach((fill, i) => {
      const ink = inks[i]!;
      const other = ink === "#1b1a18" ? "#f7f4ee" : "#1b1a18";
      expect(contrast(fill, ink)).toBeGreaterThan(contrast(fill, other));
      expect(contrast(fill, ink)).toBeGreaterThanOrEqual(4.5);
    });
  });
});
