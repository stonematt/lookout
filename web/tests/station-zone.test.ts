// @vitest-environment happy-dom
/**
 * The site writes every station-local label in the zone the snapshot names
 * (ADR-0005): the Pacific golden re-zoned to Tokyo moves the labels, and the
 * Pacific golden itself is untouched.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { formatStamp, localDay, localHour, setStationZone, stationFormat } from "../src/format.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const PACIFIC = "America/Los_Angeles";
const TOKYO = "Asia/Tokyo";
/** Jan 8, 2026, 4:25 pm Pacific; Jan 9, 9:25 am in Tokyo. */
const OBSERVED_MS = 1767918300000;

let page: Page | null = null;

beforeEach(() => {
  vi.useFakeTimers();
  history.replaceState(null, "", "/");
  document.body.innerHTML = "<div id=masthead></div><main id=sections></main>";
});

afterEach(() => {
  page?.stop();
  page = null;
  vi.useRealTimers();
});

/** The raining golden, named in `tz`. */
function snapshotIn(tz: string) {
  const doc = golden("raining", "dashboard") as { station: { tz: string } };
  doc.station.tz = tz;
  return parseDashboard(doc);
}

/** Render the page for a snapshot in `tz`; the header and overview text. */
async function renderedIn(tz: string): Promise<string> {
  const snapshot = snapshotIn(tz);
  page = startPage(
    document.getElementById("masthead")!,
    document.getElementById("sections")!,
    { dashboard: async () => snapshot, current: () => new Promise<CurrentReading>(() => {}) },
  );
  await page.booted;
  return document.body.textContent ?? "";
}

describe("the station's zone", () => {
  it("is the zone the rendered snapshot names", async () => {
    setStationZone("UTC");
    await renderedIn(PACIFIC);
    expect(formatStamp(OBSERVED_MS)).toBe("Jan 8, 4:25 pm");
  });

  it("moves every label when the snapshot names another zone", async () => {
    await renderedIn(TOKYO);
    expect(formatStamp(OBSERVED_MS)).toBe("Jan 9, 9:25 am");
    expect(localDay(OBSERVED_MS)).toBe("2026-01-09");
    expect(localHour(OBSERVED_MS)).toBe(9);
  });

  it("changes what the page writes, and Pacific keeps its golden text", async () => {
    const pacific = await renderedIn(PACIFIC);
    page?.stop();
    document.body.innerHTML = "<div id=masthead></div><main id=sections></main>";
    const tokyo = await renderedIn(TOKYO);
    expect(tokyo).not.toBe(pacific);
    expect(pacific).toContain("Jan 8, 4:25 pm");
    expect(tokyo).toContain("Jan 9, 9:25 am");
    expect(tokyo).not.toContain("Jan 8, 4:25 pm");
  });

  it("refuses to format before any snapshot has named it", () => {
    vi.resetModules();
    return import("../src/format.js").then((fresh) => {
      expect(() => fresh.stationFormat("en-US", {})).toThrow(/time zone/);
    });
  });

  it("builds a formatter per zone", () => {
    setStationZone(PACIFIC);
    const pacific = stationFormat("en-US", { hour: "numeric" });
    setStationZone(TOKYO);
    expect(stationFormat("en-US", { hour: "numeric" })).not.toBe(pacific);
  });
});
