// @vitest-environment happy-dom
/**
 * Every zoom's headings nest (spec #460 story 56, #571): the zoom's `h2`,
 * each panel's `h3` title, and below a title `h4` or no heading at all. A
 * section shared with a tab (the Rain tab's water year and dry spells, the
 * almanac, the temperature blocks, Sunlight now) draws its finding as an
 * `h4` in a zoom panel and keeps its `h2` on its tab.
 */
import { afterEach, describe, expect, it } from "vitest";

import { ZOOMS } from "../src/overview/zooms/index.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import { renderRain } from "../src/rain/index.js";
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

/** A heading's level: 2 for an `h2`. */
const level = (el: Element) => Number(el.tagName.slice(1));

const CASES = ZOOMS.flatMap((zoom) =>
  SCENARIOS.flatMap((scenario) =>
    [390, 1440].map((width) => [zoom.slug, scenario, width] as const),
  ),
);

describe("every zoom's headings nest", () => {
  it("covers all ten zooms", () => {
    expect(ZOOMS).toHaveLength(10);
  });

  it.each(CASES)("%s on the %s golden at %ipx", async (slug, scenario, width) => {
    await boot(
      `#overview/${slug}`,
      parseDashboard(structuredClone(golden(scenario, "dashboard"))),
      width,
    );
    const zoom = document.querySelector<HTMLElement>(`.zoom[data-zoom="${slug}"]`);
    if (!zoom) {
      // This golden does not draw the tile (Inside or Daylight on a station
      // without them), so there is no zoom to walk. The seasoned golden
      // draws every tile, so each zoom is walked there at least.
      expect(scenario).not.toBe("seasoned");
      expect(document.querySelector(".zoom")).toBeNull();
      return;
    }
    const headings = [...zoom.querySelectorAll("h1, h2, h3, h4, h5, h6")];
    // The zoom's own heading first, and the only h2.
    expect(headings[0]).toBe(zoom.querySelector(".zoom-heading"));
    expect(headings.filter((h) => level(h) <= 2)).toEqual([headings[0]]);
    // Each panel opens with its h3 title; under it, h4 or nothing.
    for (const panel of zoom.querySelectorAll<HTMLElement>(".zoom-panel")) {
      const inPanel = [...panel.querySelectorAll("h1, h2, h3, h4, h5, h6")];
      expect(inPanel[0]?.classList.contains("zoom-panel-title"), panel.dataset.panel).toBe(true);
      expect(inPanel.map(level), panel.dataset.panel).toEqual([
        3,
        ...inPanel.slice(1).map(() => 4),
      ]);
    }
    // In document order no heading skips a level on the way down.
    for (let i = 1; i < headings.length; i++) {
      expect(level(headings[i]!) - level(headings[i - 1]!)).toBeLessThanOrEqual(1);
    }
  });
});

describe("a section shared with a tab", () => {
  it("keeps its finding an h2 on the tab", () => {
    const host = document.createElement("div");
    document.body.replaceChildren(host);
    renderRain(host, parseDashboard(structuredClone(golden("seasoned", "dashboard"))));
    for (const id of ["years", "dry"]) {
      const finding = host.querySelector(`[data-section="${id}"] [data-part=finding]`);
      expect(finding?.tagName, id).toBe("H2");
    }
  });
});
