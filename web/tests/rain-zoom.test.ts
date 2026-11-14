// @vitest-environment happy-dom
/**
 * The rain zoom and the dry spell zoom (#475, spec #460 stories 36–37),
 * through the page on the goldens: each draws the Rain tab's own section
 * (one home per chart) without its source note (prototype decision 22),
 * the rain zoom adds the rain almanac and rolling-window rain, and both end
 * in "More in Rain", a link to the Rain tab.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot, WaterYearCurve } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

/** A stand-in `matchMedia` for a viewport width. */
function media(width: number): void {
  window.matchMedia = ((query: string) => {
    const found = /min-width:\s*(\d+)px/.exec(query);
    return {
      matches: found ? width >= Number(found[1]) : false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaQueryList;
  }) as typeof window.matchMedia;
}

/** Follow a link as a browser does: a new history entry and a hash change. */
function follow(event: Event): void {
  if (event.defaultPrevented) return;
  const link = (event.target as Element).closest?.("a[href^='#']");
  if (!link) return;
  event.preventDefault();
  history.pushState(null, "", link.getAttribute("href")!);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

beforeEach(() => document.addEventListener("click", follow));

afterEach(() => {
  page?.stop();
  page = null;
  document.removeEventListener("click", follow);
  window.matchMedia = realMatchMedia;
  vi.restoreAllMocks();
});

const seasoned = (): DashboardSnapshot =>
  parseDashboard(structuredClone(golden("seasoned", "dashboard")));

/** The seasoned golden with two comparable water years (no golden has one). */
function withYears(): DashboardSnapshot {
  const snapshot = seasoned();
  const tab = snapshot.rain_tab;
  const place = tab.day_index;
  const curve = (water_year: number, value: number): WaterYearCurve => {
    const cumulative = Array.from({ length: place + 1 }, () => value);
    return {
      water_year,
      first_day: `${water_year - 1}-10-01`,
      cumulative_in: cumulative,
      prorated: cumulative.map(() => false),
      status: {
        key: `water_year_${water_year}`,
        first_day: `${water_year - 1}-10-01`,
        last_day: `${water_year}-05-20`,
        days: place + 1,
        missing_days: 0,
        status: "valid",
        reason: null,
      },
      total_in: value,
      to_date_in: value,
    };
  };
  tab.water_years = [curve(2025, 3.1), curve(2026, 4.2)];
  return snapshot;
}

async function boot(hash: string, snapshot = seasoned(), width = 1440): Promise<void> {
  media(width);
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
  history.replaceState(null, "", `/${hash}`);
  globalThis.fetch = (() => new Promise<Response>(() => {})) as typeof fetch;
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
  });
  await page.booted;
}

const zoom = () => document.querySelector<HTMLElement>("#panel-overview .zoom");
const panels = () => [...zoom()!.querySelectorAll<HTMLElement>(".zoom-panel")];
const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();

/** The Rain tab's own section, drawn on the tab. */
async function rainSection(id: string, snapshot: DashboardSnapshot): Promise<HTMLElement> {
  await boot("#rain", snapshot);
  return document.querySelector<HTMLElement>(`#panel-rain [data-section=${id}]`)!;
}

describe.each([1440, 390])("the rain zoom at %ipx", (width) => {
  it("leads with the Rain tab's water-year section, then the almanac and rolling rain", async () => {
    await boot("#overview/rain", seasoned(), width);

    expect(words(zoom()!.querySelector(".zoom-heading"))).toBe("Rain since Oct 1");
    expect(panels().map((p) => p.dataset.panel)).toEqual([
      "water-year",
      "rain-windows",
      "almanac",
      "rolling-rain",
      "more",
    ]);
  });

  it("ends in More in Rain, a link to the Rain tab", async () => {
    await boot("#overview/rain", seasoned(), width);
    const more = panels().at(-1)!;
    const link = more.querySelector<HTMLAnchorElement>("a")!;

    expect(link.getAttribute("href")).toBe("#rain");
    expect(words(link)).toMatch(/^More in Rain\b/);
    link.click();
    expect(location.hash).toBe("#rain");
    expect(document.getElementById("panel-rain")!.hidden).toBe(false);
  });
});

describe("the rain zoom", () => {
  it("draws the Rain tab's water-year section, with no source note, its strip left to the gauge", async () => {
    const snapshot = withYears();
    const tab = await rainSection("years", snapshot);
    const onTab = {
      finding: words(tab.querySelector("[data-part=finding]")),
      chart: tab.querySelector(".rain-chart svg")?.getAttribute("aria-label"),
      strip: words(tab.querySelector(".rain-windows")),
    };
    page!.stop();

    await boot("#overview/rain", snapshot);
    const years = panels()[0]!;
    expect(words(years.querySelector("[data-part=finding]"))).toBe(onTab.finding);
    expect(onTab.finding).toMatch(/^Water year 2026 has 4\.20 in of rain since Oct 1/);
    expect(years.querySelector(".rain-chart svg")?.getAttribute("aria-label")).toBe(onTab.chart);
    expect(onTab.strip).toBeTruthy();
    expect(years.querySelector(".rain-windows")).toBeNull();
    expect(tab.querySelector("[data-part=source-note]")).not.toBeNull();
  });

  it("carries no source note or method fold (prototype decision 22)", async () => {
    await boot("#overview/rain");
    expect(zoom()!.querySelector("details, .tile-folds, [data-part=source]")).toBeNull();
  });

  it("holds the rain almanac grid and the rolling-window table", async () => {
    await boot("#overview/rain");
    expect(zoom()!.querySelector("[data-panel=almanac] table.alm-grid")).not.toBeNull();
    expect(zoom()!.querySelector("[data-panel=rolling-rain] table.alm-table")).not.toBeNull();
  });
});

describe("the dry spell zoom", () => {
  it("draws the Rain tab's dry spell section, then More in Rain", async () => {
    const snapshot = seasoned();
    const tab = await rainSection("dry", snapshot);
    const onTab = words(tab.querySelector(".rain-dry-stats"));
    page!.stop();

    await boot("#overview/dry-spell", snapshot);
    expect(words(zoom()!.querySelector(".zoom-heading"))).toBe("Dry spell");
    expect(panels().map((p) => p.dataset.panel)).toEqual(["dry-spells", "more"]);
    expect(words(panels()[0]!.querySelector(".rain-dry-stats"))).toBe(onTab);
    expect(zoom()!.querySelector("details, .tile-folds")).toBeNull();
    expect(panels()[1]!.querySelector("a")?.getAttribute("href")).toBe("#rain");
  });
});
