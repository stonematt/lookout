// @vitest-environment happy-dom
/**
 * The Rain tab's wet hours since Oct 1 against usual wet hours to date
 * (#663): the golden snapshots, and N = 0 to 4 earlier years as worked
 * examples.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { renderRain } from "../src/rain/index.js";
import { usualWetWords, wetToDateValue } from "../src/rain/text.js";
import type {
  DashboardSnapshot,
  EarlierWetYear,
  UsualWetHours,
  WetHoursToDate,
} from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { golden } from "./golden.js";

const seasoned = parseDashboard(golden("seasoned", "dashboard"));
const raining = parseDashboard(golden("raining", "dashboard"));

let panel: HTMLElement;

beforeEach(() => {
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const part = () => panel.querySelector("[data-part=wet-to-date]");

/** This year: 113 wet of 1,292 countable, nothing missing. */
function toDate(usual: UsualWetHours | null, missing = 0): WetHoursToDate {
  return {
    water_year: 2026,
    first_day: "2025-10-01",
    through_ms: 1_763_884_800_000,
    hours: 1292 + missing,
    wet_hours: 113,
    countable_hours: 1292,
    missing_hours: missing,
    usual,
  };
}

/** An earlier year whose share gives `usual_hours` this year. */
const earlier = (water_year: number, usual_hours: number): EarlierWetYear => ({
  water_year,
  wet_hours: usual_hours,
  countable_hours: 1292,
  usual_hours,
});

/** A usual from the years given, with a band when there is one. */
function usual(
  years: EarlierWetYear[],
  band: [number, number, "percentile" | "range"] | null = null,
): UsualWetHours {
  return {
    years: years.length,
    earlier: years,
    low_hours: band ? band[0] : null,
    high_hours: band ? band[1] : null,
    band: band ? band[2] : null,
  };
}

/** A copy of a snapshot with its wet hours to date replaced. */
function withToDate(snapshot: DashboardSnapshot, wet: unknown): DashboardSnapshot {
  const copy = structuredClone(snapshot);
  (copy.rain_tab as unknown as Record<string, unknown>).wet_hours_to_date = wet;
  return copy;
}

describe("wet hours since Oct 1", () => {
  it("reads wet hours of countable hours, as the Overview's meter does", () => {
    expect(wetToDateValue(toDate(null))).toBe("113 h of 1,292");
  });

  it("has no usual with no earlier year (N = 0)", () => {
    expect(usualWetWords(toDate(null))).toBe("no earlier year to compare yet");
  });

  it("gives the plain value of one earlier year (N = 1)", () => {
    expect(usualWetWords(toDate(usual([earlier(2025, 160)])))).toBe(
      "usual to date 160 h, from 2025",
    );
  });

  it("gives both plain values of two earlier years (N = 2)", () => {
    expect(usualWetWords(toDate(usual([earlier(2024, 150), earlier(2025, 170)])))).toBe(
      "usual to date 150 h (2024), 170 h (2025)",
    );
  });

  it("calls the band of three earlier years a range, not a percentile (N = 3)", () => {
    const three = usual(
      [earlier(2023, 140), earlier(2024, 150), earlier(2025, 190)],
      [145, 170, "range"],
    );
    expect(usualWetWords(toDate(three))).toBe("usual to date 145–170 h, range of 3 earlier years");
  });

  it("calls the band of four earlier years a percentile band (N = 4)", () => {
    const four = usual(
      [earlier(2022, 120), earlier(2023, 140), earlier(2024, 150), earlier(2025, 190)],
      [135, 160, "percentile"],
    );
    expect(usualWetWords(toDate(four))).toBe(
      "usual to date 135–160 h, 25th–75th percentile of 4 earlier years",
    );
  });

  it("says the hours missing are not dry", () => {
    expect(usualWetWords(toDate(null, 27))).toBe(
      "no earlier year to compare yet · 27 h missing, not dry",
    );
  });
});

describe("the Rain tab's wet hours since Oct 1", () => {
  it("draws the golden snapshot's wet hours under the window strip", () => {
    renderRain(panel, seasoned);
    const wet = seasoned.rain_tab.wet_hours_to_date!;
    const [label, value, note] = [...part()!.querySelectorAll("dt, dd")].map(text);
    expect([label, value, note]).toEqual([
      "Wet hours since Oct 1",
      wetToDateValue(wet),
      usualWetWords(wet),
    ]);
    expect(note).toBe("usual to date 3 h, from 2025 · 1 h missing, not dry");
    // Under the window strip, in the water-year tile.
    expect(part()!.previousElementSibling?.getAttribute("data-part")).toBe("windows");
  });

  it("draws nothing for a snapshot written before it (N−1)", () => {
    renderRain(panel, withToDate(raining, undefined));
    expect(part()).toBeNull();
  });

  it("refuses a usual whose years do not match its list", () => {
    const wrong = { ...toDate(usual([earlier(2025, 160)])), usual: { ...usual([]), years: 2 } };
    const doc = structuredClone(golden("raining", "dashboard")) as Record<string, unknown>;
    (doc.rain_tab as Record<string, unknown>).wet_hours_to_date = wrong;
    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });
});
