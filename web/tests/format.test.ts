import { describe, expect, it } from "vitest";

import {
  MONTHS,
  STALE_AFTER_MS,
  WATER_MONTHS,
  dataAge,
  dayText,
  degrees,
  formatDayCount,
  formatDays,
  formatDuration,
  formatHours,
  formatStamp,
  hourName,
  hourText,
  monthLongName,
  inches,
  inchesPerHour,
  kwhPerM2,
  mph,
  wattsPerM2,
} from "../src/format.js";

const MIN = 60_000;

describe("formatStamp", () => {
  it("writes station-local Pacific time in the One Format Rule", () => {
    // 2026-01-09T00:45:00Z is Jan 8, 4:45 pm in Pacific standard time.
    expect(formatStamp(Date.UTC(2026, 0, 9, 0, 45))).toBe("Jan 8, 4:45 pm");
    // 2025-07-04T18:05:00Z is Jul 4, 11:05 am in Pacific daylight time.
    expect(formatStamp(Date.UTC(2025, 6, 4, 18, 5))).toBe("Jul 4, 11:05 am");
  });
});

describe("formatDuration", () => {
  it("writes minutes, hours and days the One Format Rule way", () => {
    expect(formatDuration(45 * MIN)).toBe("45 m");
    expect(formatDuration((5 * 60 + 12) * MIN)).toBe("5 h 12 m");
    expect(formatDuration((34 * 60 + 5) * MIN)).toBe("1 d 10 h");
    expect(formatDuration(30_000)).toBe("under 1 m");
  });

  it("drops a zero minute count from whole hours", () => {
    expect(formatDuration(3 * 60 * MIN)).toBe("3 h");
    expect(formatDuration(60 * MIN)).toBe("1 h");
  });
});

describe("formatHours", () => {
  it("writes hours without going to days", () => {
    expect(formatHours(30 * 60 * MIN)).toBe("30 h");
    expect(formatHours((47 * 60 + 30) * MIN)).toBe("47 h 30 m");
    expect(formatHours(45 * MIN)).toBe("45 m");
  });

  it("can keep a duration on one line", () => {
    expect(formatHours((47 * 60 + 30) * MIN, { noBreak: true })).toBe("47\u00a0h\u00a030\u00a0m");
  });
});

describe("day counts", () => {
  it("writes a duration given in days", () => {
    expect(formatDays(1 + 22 / 24)).toBe("1 d 22 h");
    expect(formatDays(0.5)).toBe("12 h");
  });

  it("writes a spell length in days", () => {
    expect(formatDayCount(1)).toBe("1 day");
    expect(formatDayCount(2)).toBe("2 days");
    expect(formatDayCount(1.5)).toBe("1.5 days");
  });
});

describe("hours of the day", () => {
  it("names an hour", () => {
    expect([0, 3, 12, 21].map(hourText)).toEqual(["midnight", "3 am", "noon", "9 pm"]);
  });

  it("names a time's Pacific hour", () => {
    // 2026-01-08T20:30:00Z is 12:30 pm in Pacific standard time.
    expect(hourName(Date.UTC(2026, 0, 8, 20, 30))).toBe("noon");
    // 2025-07-04T07:10:00Z is midnight in Pacific daylight time.
    expect(hourName(Date.UTC(2025, 6, 4, 7, 10))).toBe("midnight");
  });
});

describe("months and days", () => {
  it("lists the calendar and the water year", () => {
    expect(MONTHS).toHaveLength(12);
    expect(MONTHS[0]).toBe("Jan");
    expect(WATER_MONTHS.slice(0, 4)).toEqual(["Oct", "Nov", "Dec", "Jan"]);
    expect(WATER_MONTHS.at(-1)).toBe("Sep");
    expect(monthLongName(10)).toBe("October");
    expect(monthLongName(1)).toBe("January");
  });

  it("writes an ISO day", () => {
    expect(dayText("2026-01-08")).toBe("Jan 8");
    expect(dayText("2025-12-30", true)).toBe("Dec 30, 2025");
  });
});

describe("units", () => {
  it("writes degrees", () => {
    expect(degrees(52.34)).toBe("52.3°");
    expect(degrees(59.6, 0)).toBe("60°");
  });

  it("writes inches, two decimals and one from 10 in", () => {
    expect(inches(0.3)).toBe("0.30 in");
    expect(inches(12.43)).toBe("12.4 in");
    expect(inches(12.43, { digits: 2 })).toBe("12.43 in");
    expect(inches(0.3, { unit: false })).toBe("0.30");
    expect(inchesPerHour(0.123)).toBe("0.12 in/hr");
  });

  it("writes wind, sun and solar energy", () => {
    expect(mph(4.25)).toBe("4.3 mph");
    expect(wattsPerM2(411.6)).toBe("412 W/m²");
    expect(kwhPerM2(2.414)).toBe("2.41 kWh/m²");
    expect(kwhPerM2(61.23)).toBe("61.2 kWh/m²");
    expect(kwhPerM2(123.4, { unit: false })).toBe("123");
  });
});

describe("dataAge", () => {
  const observed = Date.UTC(2026, 0, 9, 0, 45);

  it("is fresh up to 15 minutes", () => {
    expect(dataAge(observed, observed + 3 * MIN)).toEqual({
      ageMs: 3 * MIN,
      stale: false,
      text: "3 m ago",
    });
    expect(dataAge(observed, observed + STALE_AFTER_MS).stale).toBe(false);
  });

  it("turns stale past 15 minutes and keeps counting", () => {
    expect(dataAge(observed, observed + 16 * MIN)).toMatchObject({
      stale: true,
      text: "Stale: 16 m old",
    });
    expect(dataAge(observed, observed + (26 * 60 + 3) * MIN).text).toBe(
      "Stale: 1 d 2 h old",
    );
  });

  it("is stale with no observation at all", () => {
    expect(dataAge(null, observed)).toEqual({
      ageMs: null,
      stale: true,
      text: "No reading yet",
    });
  });

  it("treats a clock behind the observation as just now", () => {
    expect(dataAge(observed, observed - 2 * MIN).text).toBe("under 1 m ago");
  });
});
