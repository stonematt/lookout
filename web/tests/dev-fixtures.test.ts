import { describe, expect, it } from "vitest";

import { devResponse } from "../server/dev-fixtures.js";
import { golden } from "./golden.js";

const NOW = Date.UTC(2026, 9, 30, 12, 0);
const MIN = 60_000;

describe("the lane server's read functions", () => {
  it("serves the dry golden by default, as recorded", async () => {
    const response = await devResponse("/dashboard", {}, NOW);
    expect(response!.status).toBe(200);
    expect(await response!.json()).toEqual(golden("dry", "dashboard"));
  });

  it("serves another golden by name", async () => {
    const response = await devResponse("/dashboard?fixture=raining", {}, NOW);
    expect(await response!.json()).toEqual(golden("raining", "dashboard"));
  });

  it("takes the default golden from the environment", async () => {
    const response = await devResponse("/dashboard", { LOOKOUT_FIXTURE: "raining" }, NOW);
    expect(await response!.json()).toEqual(golden("raining", "dashboard"));
  });

  it("shifts every time so the newest observation is `age` minutes old", async () => {
    const recorded = golden("raining", "dashboard") as Record<string, number>;
    const response = await devResponse("/dashboard?fixture=raining&age=3", {}, NOW);
    const body = (await response!.json()) as Record<string, number> & {
      current: { last_rain_ms: number };
    };
    const delta = NOW - 3 * MIN - recorded.observed_at_ms!;
    expect(body.observed_at_ms).toBe(NOW - 3 * MIN);
    expect(body.built_at_ms).toBe(recorded.built_at_ms! + delta);
    expect(body.archive_newest_ms).toBe(recorded.archive_newest_ms! + delta);
    expect(body.current.last_rain_ms).toBe(recorded.observed_at_ms! + delta);
  });

  it("shifts every ISO day with the times, so day labels agree with them", async () => {
    type Doc = {
      gaps: { windows: { key: string; first_day: string; last_day: string }[]; months: { key: string }[] };
      rain_tab: { recent: { first_day: string; wettest_hour: { day: string } | null } };
    };
    const recorded = golden("raining", "dashboard") as unknown as Doc;
    const response = await devResponse("/dashboard?fixture=raining&age=3", {}, NOW);
    const body = (await response!.json()) as Doc;
    const window = (doc: Doc, key: string) => doc.gaps.windows.find((w) => w.key === key)!;

    // NOW - 3 m is Oct 30, 4:57 am Pacific: today is Oct 30, 2026.
    expect(window(body, "today").first_day).toBe("2026-10-30");
    expect(window(body, "7d")).toMatchObject({ first_day: "2026-10-24", last_day: "2026-10-30" });
    // Recorded on Jan 8, 2026: 295 days earlier.
    expect(window(recorded, "today").first_day).toBe("2026-01-08");
    expect(recorded.rain_tab.recent.first_day).toBe("2025-10-11");
    expect(body.rain_tab.recent.first_day).toBe("2026-08-02");
    // A day named in a key, or a span of days, moves with it.
    const dayKeys = (doc: Doc) =>
      JSON.stringify(doc).match(/"key":"\d{4}-\d{2}-\d{2}(\.\.\d{4}-\d{2}-\d{2})?"/g) ?? [];
    expect(dayKeys(recorded)).toContain('"key":"2026-01-08"');
    expect(dayKeys(body)).toContain('"key":"2026-10-30"');
    expect(dayKeys(body)).not.toContain('"key":"2026-01-08"');
  });

  it("serves /current from the context golden, with no live reading", async () => {
    const response = await devResponse("/current?fixture=raining", {}, NOW);
    const body = (await response!.json()) as Record<string, unknown>;
    expect(body.kind).toBe("current");
    expect(body.last_data).toBeNull();
    expect(body.context).toEqual(golden("raining", "context"));
  });

  it("refuses an unknown golden", async () => {
    const response = await devResponse("/dashboard?fixture=nope", {}, NOW);
    expect(response!.status).toBe(400);
  });

  it("leaves every other path to Vite", async () => {
    expect(await devResponse("/src/main.ts", {}, NOW)).toBeNull();
  });
});
