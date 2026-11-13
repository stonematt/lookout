// @vitest-environment happy-dom
/**
 * The Overview's voice pass (#595): a finding says what the reading is and
 * which side of usual it sits on, never the method. No percentile and no
 * ordinal ("62nd") in any Overview finding, tile name or zoom, visible or
 * spoken, on any golden; the method is stated once, in the sources footer.
 */
import { afterEach, describe, expect, it } from "vitest";

import { ZOOMS } from "../src/overview/zooms/index.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { SCENARIOS, golden } from "./golden.js";

/** A stand-in `matchMedia` at a viewport width. */
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

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

afterEach(() => {
  page?.stop();
  page = null;
  window.matchMedia = realMatchMedia;
});

async function boot(hash: string, snapshot: DashboardSnapshot, width: number): Promise<void> {
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

/**
 * Every word the page shows or speaks, the sources footer aside: its text
 * (hidden spoken words included) and every `aria-label`, `title` and `alt`.
 */
function pageWords(): string {
  const root = document.body.cloneNode(true) as HTMLElement;
  root.querySelectorAll("footer.ov-sources").forEach((footer) => footer.remove());
  const attrs = [...root.querySelectorAll("[aria-label], [title], [alt]")].flatMap((el) =>
    ["aria-label", "title", "alt"].map((name) => el.getAttribute(name) ?? ""),
  );
  return `${root.textContent ?? ""} ${attrs.join(" ")}`;
}

const METHOD = /percentile|\b\d+(st|nd|rd|th)\b/i;

type Doc = Record<string, unknown>;

/** A golden copy whose unusual feature is retold as another reading and side. */
function unusual(reading: string, value: number, unit: string, percentile: number) {
  const doc = structuredClone(golden("seasoned", "dashboard")) as Doc;
  doc.feature = { ...(doc.feature as Doc), reading, value, unit, percentile };
  return parseDashboard(doc);
}

const ROUTES = ["#overview", ...ZOOMS.map((zoom) => `#overview/${zoom.slug}`)];
const CASES = ROUTES.flatMap((route) =>
  SCENARIOS.flatMap((scenario) => [390, 1440].map((width) => [route, scenario, width] as const)),
);

describe("no percentile and no ordinal on the Overview or a zoom", () => {
  it.each(CASES)("%s on the %s golden at %ipx", async (route, scenario, width) => {
    await boot(route, parseDashboard(structuredClone(golden(scenario, "dashboard"))), width);
    const words = pageWords();

    expect(words.length).toBeGreaterThan(100);
    expect(words).not.toMatch(METHOD);
  });

  it.each([
    ["temperature", 41, "f", 4],
    ["humidity", 99, "pct", 97],
    ["pressure", 29.41, "inhg", 1],
    ["solar", 910, "wm2", 62],
  ])("an unusual %s on the Overview, nor in its zoom", async (reading, value, unit, pct) => {
    for (const route of ["#overview", `#overview/${reading === "solar" ? "sun" : reading}`]) {
      await boot(route, unusual(reading, value, unit, pct), 1440);
      expect(pageWords()).not.toMatch(METHOD);
      page?.stop();
      page = null;
    }
  });
});

describe("the method, stated once in the sources footer", () => {
  it.each(SCENARIOS)("in the feature's line on the %s golden", async (scenario) => {
    await boot("#overview", parseDashboard(structuredClone(golden(scenario, "dashboard"))), 1440);
    const footer = document.querySelector<HTMLElement>("footer.ov-sources")!;
    const line = footer.querySelector(
      '.ov-sources-how [data-source="The feature"] .ov-sources-method',
    )!;

    expect(line.textContent).toBe(
      "A rain event that is live or ended in the last 12 hours leads; then an unusual " +
        "reading, the one furthest out winning; then a dry spell longer than the usual one; " +
        "otherwise a usual day. A reading is unusual below the 10th or above the 90th " +
        "percentile of its readings at the same hour within 7 days either side of this date " +
        "in earlier years (its percentile is the share of those readings below it): above " +
        "the 90th it is higher than usual, below the 10th lower.",
    );
    // No other footer line or method repeats it.
    const others = [...footer.querySelectorAll("[data-source]")]
      .filter((item) => item !== line.parentElement)
      .map((item) => item.textContent)
      .join(" ");
    expect(others).not.toMatch(/percentile|10th|90th|either side of this date/);
  });
});
