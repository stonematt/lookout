// @vitest-environment happy-dom
/**
 * The phone layout (#337), through `startPage` on the goldens: under 760px
 * each tab opens with its fixed top, then its other sections as closed
 * finding rows in one exclusive group; the tabs are the bottom tab bar and
 * the header is the compact one, holding the station selector and the stale
 * mark. From 760px the tile grid is unchanged.
 *
 * happy-dom evaluates no media queries, so the page's own layout switch gets
 * a stand-in `matchMedia` over a viewport width the test sets and moves; the
 * tests read the structure and attributes, never computed CSS.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import type { StationIndex } from "../src/stations.js";
import type { Scenario } from "./golden.js";
import { SCENARIOS, golden, twoStations } from "./golden.js";
import type { Viewport } from "./viewport.js";
import { viewport } from "./viewport.js";

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

/** The stand-in viewport the page's layout switch reads. */
let view: Viewport;

beforeEach(() => {
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
  view = viewport(390).install();
});

afterEach(() => {
  page?.stop();
  page = null;
  window.matchMedia = realMatchMedia;
  document.documentElement.removeAttribute("data-layout");
});

const parsed = (scenario: Scenario): DashboardSnapshot =>
  parseDashboard(golden(scenario, "dashboard"));

/** Boot the page at a tab; `/current` and `/event` never answer. */
async function boot(
  snapshot: DashboardSnapshot,
  tab = "overview",
  stations?: StationIndex,
): Promise<void> {
  history.replaceState(null, "", `/#${tab}`);
  globalThis.fetch = (() => new Promise<Response>(() => {})) as typeof fetch;
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
    ...(stations ? { stations: () => Promise.resolve(stations) } : {}),
  });
  await page.booted;
}

const panel = (tab: string) => document.getElementById(`panel-${tab}`)!;
const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const open = (tab: string) => document.querySelector<HTMLElement>(`[aria-controls=panel-${tab}]`)!;

/** The grid's children in order: a tile's section id, or "rows: <ids>". */
function shape(tab: string): string[] {
  return [...panel(tab).querySelectorAll<HTMLElement>(".tile-grid > *")].map((el) =>
    el.classList.contains("tile-rows")
      ? `rows: ${[...el.querySelectorAll<HTMLElement>("details.tile-row")].map((r) => r.dataset.section).join(" ")}`
      : el.dataset.section!,
  );
}

const rows = (tab: string) => [
  ...panel(tab).querySelectorAll<HTMLDetailsElement>("details.tile-row"),
];

/** Each tab's phone shape: its fixed top first, then one group of rows. */
const PHONE = {
  overview: [
    "first-screen",
    "hero-lead",
    "rows: temperature-periods against-normal ten-days almanac rolling-rain",
    // The quick facts (#340): open tiles after the rows, two to a row.
    "wind",
    "pressure",
    "air",
    "sun-moon", // #341
  ],
  rain: ["daily", "rows: years dry hours"],
  solar: ["now", "rows: daily day heatmaps"],
  // The event list is itself an accordion (#339): open, never a row.
  events: ["catalog", "events"],
} as const;

describe.each(SCENARIOS)("the %s golden on a phone", (scenario) => {
  it.each(Object.keys(PHONE) as (keyof typeof PHONE)[])(
    "%s renders its fixed top first, then its other sections as closed finding rows",
    async (tab) => {
      await boot(parsed(scenario), tab);
      expect(document.documentElement.dataset.layout).toBe("phone");
      expect(shape(tab)).toEqual(PHONE[tab]);
      for (const top of panel(tab).querySelectorAll("[data-phone-top]")) {
        expect(top.tagName).toBe("SECTION");
      }
      const group = new Set(rows(tab).map((row) => row.getAttribute("name")));
      expect(group.size).toBeLessThanOrEqual(1);
      expect(rows(tab).every((row) => !row.open)).toBe(true);
    },
  );

  it("puts the Overview's finding rows under an In context heading", async () => {
    await boot(parsed(scenario));
    const group = panel("overview").querySelector(".tile-rows")!;
    expect(words(group.querySelector("h2.tile-rows-heading"))).toBe("In context");
    expect(group.getAttribute("aria-label")).toBe("In context");
  });

  it.each(["overview", "rain", "solar"])(
    "%s: a closed row reads as its kicker and finding; opening it shows its chart and folded source",
    async (tab) => {
      await boot(parsed(scenario), tab);
      for (const row of rows(tab)) {
        const line = row.querySelector(":scope > summary")!;
        const kicker = line.querySelector("[data-part=kicker]");
        const finding = line.querySelector("[data-part=finding]");
        expect(words(kicker)).toBeTruthy();
        expect(words(finding)).toBeTruthy();
        // Nothing else on the closed line: the chevron is drawn, not spoken.
        expect(line.textContent).toBe(`${kicker!.textContent}${finding!.textContent}`);
        expect(line.querySelector(".tile-row-chevron")?.getAttribute("aria-hidden")).toBe("true");
        const body = row.querySelector(":scope > .tile-row-body")!;
        expect(body.querySelector("svg, figure, table, .gap-hatch")).not.toBeNull();
        expect(body.querySelector("details[data-part=source-note]")).not.toBeNull();
        expect(body.querySelector("[data-part=finding]")).toBeNull();
      }
    },
  );

  it.each(["overview", "rain", "solar"])(
    "%s: opening a finding row closes the one open before it",
    async (tab) => {
      await boot(parsed(scenario), tab);
      const [first, second] = rows(tab);
      first!.querySelector("summary")!.click();
      await new Promise((r) => setTimeout(r, 0));
      expect(rows(tab).filter((row) => row.open)).toEqual([first]);

      second!.querySelector("summary")!.click();
      await new Promise((r) => setTimeout(r, 0));
      expect(rows(tab).filter((row) => row.open)).toEqual([second]);

      // A source note opened inside the open row leaves that row open.
      second!.querySelector<HTMLElement>("details[data-part=source-note] > summary")!.click();
      await new Promise((r) => setTimeout(r, 0));
      expect(rows(tab).filter((row) => row.open)).toEqual([second]);
    },
  );
});

describe("the phone's bottom tab bar", () => {
  it("is the page's section navigation, marks the open tab, and switches tabs as before", async () => {
    await boot(parsed("dry"), "rain");
    const bar = document.querySelector("nav.tabs")!;
    expect(bar.getAttribute("aria-label")).toBe("Dashboard sections");
    expect(bar.querySelectorAll("[role=tab]")).toHaveLength(4);
    expect(open("rain").getAttribute("aria-selected")).toBe("true");
    expect(open("rain").tabIndex).toBe(0);

    open("solar").click();
    expect(location.hash).toBe("#solar");
    expect(open("solar").getAttribute("aria-selected")).toBe("true");
    expect(open("rain").getAttribute("aria-selected")).toBe("false");
    expect(panel("solar").hidden).toBe(false);
    expect(shape("solar")).toEqual(PHONE.solar);
  });

  it("is placed at the bottom on a phone, padded clear of the safe-area inset", () => {
    const css = readFileSync(resolve(import.meta.dirname, "../src/styles/base.css"), "utf8");
    const rule =
      /:root:not\(\[data-layout="two"\], \[data-layout="four"\]\) \.page > \.tabs \{([^}]*)\}/.exec(
        css,
      );
    expect(rule).not.toBeNull();
    expect(rule![1]).toContain("position: fixed");
    expect(rule![1]).toMatch(/padding:[^;]*env\(safe-area-inset-bottom\)/);
    expect(readFileSync(resolve(import.meta.dirname, "../index.html"), "utf8")).toContain(
      "viewport-fit=cover",
    );
  });
});

describe("the phone's compact header", () => {
  it("holds the station selector when there are two or more stations", async () => {
    await boot(parsed("dry"), "overview", twoStations());
    const select = document.querySelector<HTMLSelectElement>("#masthead select.station-select");
    expect(select).not.toBeNull();
    expect(select!.options).toHaveLength(2);
  });

  it("shows a stale golden with the hollow dot and hatched age, the same as the desktop header", async () => {
    const stale = parsed("dry");
    const mark = () => {
      const line = document.querySelector<HTMLElement>(".masthead-reading")!;
      return { stale: line.dataset.stale, age: words(line.querySelector(".masthead-age")) };
    };
    await boot(stale);
    const phone = mark();
    expect(phone.stale).toBe("true");
    expect(phone.age).toBeTruthy();
    page!.stop();

    document.body.innerHTML =
      "<div class=page><div id=masthead></div><main id=sections></main></div>";
    view = viewport(1440).install();
    await boot(stale);
    expect(mark()).toEqual(phone);
  });
});

describe("from 760px", () => {
  it.each([760, 1180])("at %ipx every section is a tile, as before", async (at) => {
    view = viewport(at).install();
    await boot(parsed("raining"), "rain");
    expect(panel("rain").querySelector("details.tile-row, .tile-rows")).toBeNull();
    expect(shape("rain")).toEqual(["daily", "years", "dry", "hours"]);
  });

  it("redraws the open tab as tiles when the viewport crosses 760px, and as rows again under it", async () => {
    await boot(parsed("raining"), "rain");
    expect(shape("rain")).toEqual(PHONE.rain);
    view.resize(1280);
    expect(document.documentElement.dataset.layout).toBe("four");
    expect(shape("rain")).toEqual(["daily", "years", "dry", "hours"]);
    expect(shape("overview")).toContain("temperature-periods");
    view.resize(900);
    expect(shape("rain")).toEqual(["daily", "years", "dry", "hours"]);
    view.resize(390);
    expect(shape("rain")).toEqual(PHONE.rain);
  });
});
