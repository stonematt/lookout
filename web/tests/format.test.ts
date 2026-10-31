import { describe, expect, it } from "vitest";

import { STALE_AFTER_MS, dataAge, formatDuration, formatStamp } from "../src/format.js";

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
