// @vitest-environment happy-dom
/**
 * A young record is silent (#720): what it cannot show yet is not drawn and
 * not explained. Through the page on the young goldens (`dry`, `raining`):
 * the Overview and every zoom its tiles open carry no "not compared yet",
 * next-unlock or too-few-hours words, and no zoom panel is left empty.
 *
 * Kept: true statements about the record ("No rain in the record yet", "No
 * dry spell yet") and gap reasons for data that should exist ("not archived
 * yet"), which are not about the record being too short.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

/** Words that explain a record too young to show something. */
const TOO_YOUNG =
  /not compared yet|compared yet|compare yet|not enough|complete enough|unlock|too few|no usual band|no reading yet|No month is in the record|no earlier year to set/i;

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

afterEach(() => {
  page?.stop();
  page = null;
  window.matchMedia = realMatchMedia;
  vi.restoreAllMocks();
});

const parsed = (scenario: "dry" | "raining"): DashboardSnapshot =>
  parseDashboard(structuredClone(golden(scenario, "dashboard")));

/**
 * Boot the page at a hash on a four-column screen.
 *
 * @param hash - The route, e.g. "#overview/rain".
 * @param snapshot - The snapshot the page reads.
 */
async function boot(hash: string, snapshot: DashboardSnapshot): Promise<void> {
  page?.stop();
  window.matchMedia = ((query: string) => {
    const found = /min-width:\s*(\d+)px/.exec(query);
    return {
      matches: found ? 1440 >= Number(found[1]) : false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaQueryList;
  }) as typeof window.matchMedia;
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

const overview = () => document.getElementById("panel-overview")!;

describe.each(["dry", "raining"] as const)("the young %s golden", (scenario) => {
  it("says nothing on the Overview about a record too young", async () => {
    await boot("#overview", parsed(scenario));
    expect(overview().textContent).not.toMatch(TOO_YOUNG);
    for (const el of overview().querySelectorAll("[aria-label]")) {
      expect(el.getAttribute("aria-label")).not.toMatch(TOO_YOUNG);
    }
  });

  it("says nothing in any zoom, and leaves no zoom panel empty", async () => {
    await boot("#overview", parsed(scenario));
    const slugs = [...overview().querySelectorAll<HTMLAnchorElement>('a[href^="#overview/"]')]
      .map((a) => a.getAttribute("href")!.slice("#overview/".length))
      .filter((slug, i, all) => all.indexOf(slug) === i);
    expect(slugs.length).toBeGreaterThan(0);

    for (const slug of slugs) {
      await boot(`#overview/${slug}`, parsed(scenario));
      const zoom = overview().querySelector<HTMLElement>(".zoom");
      if (!zoom) continue;
      expect(zoom.textContent, slug).not.toMatch(TOO_YOUNG);
      for (const panel of zoom.querySelectorAll<HTMLElement>(".zoom-panel")) {
        const title = panel.querySelector(".zoom-panel-title");
        const body = [...panel.children].filter((child) => child !== title);
        const said = body.some(
          (child) => (child.textContent ?? "").trim() !== "" || child.querySelector("svg, canvas"),
        );
        expect(said, `${slug}: ${panel.dataset.panel}`).toBe(true);
      }
    }
  });
});
