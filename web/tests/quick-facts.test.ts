// @vitest-environment happy-dom
/**
 * The Overview's quick-fact tiles (#340), through `startPage` on the
 * goldens: wind, pressure and air, each a kicker, one large value and one
 * sentence, read from the snapshot's current conditions. A missing value
 * reads as not known, never as zero; the pressure sentence follows the
 * snapshot's 3-hour trend word; wind and humidity agree with the now tile.
 * On a phone they sit after the finding rows, open, two to a row.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { setLayout } from "../src/layout.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot, SnapshotConditions } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { golden } from "./golden.js";

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

/** Hold the page's layout switch at one viewport width. */
function viewport(width: number): void {
  window.matchMedia = ((query: string) => {
    const min = Number(/min-width:\s*(\d+)px/.exec(query)![1]);
    return {
      matches: width >= min,
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaQueryList;
  }) as typeof window.matchMedia;
}

beforeEach(() => {
  history.replaceState(null, "", "/#overview");
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
  viewport(1440);
});

afterEach(() => {
  page?.stop();
  page = null;
  window.matchMedia = realMatchMedia;
  setLayout("four");
  document.documentElement.removeAttribute("data-layout");
});

const parsed = (scenario: Scenario): DashboardSnapshot =>
  parseDashboard(golden(scenario, "dashboard"));

/** A copy of a golden with some current conditions replaced. */
function withCurrent(scenario: Scenario, change: Partial<SnapshotConditions>): DashboardSnapshot {
  const copy = structuredClone(parsed(scenario));
  Object.assign(copy.current, change);
  return copy;
}

async function boot(snapshot: DashboardSnapshot): Promise<HTMLElement> {
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
  });
  await page.booted;
  return document.getElementById("panel-overview")!;
}

const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();

/** A quick-fact tile's kicker, value and sentence. */
function fact(panel: HTMLElement, id: string): [string?, string?, string?] {
  const tile = panel.querySelector<HTMLElement>(`.tile[data-section=${id}]`);
  expect(tile, id).not.toBeNull();
  return [
    words(tile!.querySelector("[data-part=kicker]")),
    words(tile!.querySelector("[data-part=fact]")),
    words(tile!.querySelector("[data-part=fact-sentence]")),
  ];
}

/** The now tile's Right-now strip, label to value. */
function rightNow(panel: HTMLElement): Record<string, string> {
  const out: Record<string, string> = {};
  for (const item of panel.querySelectorAll("[data-part=right-now] > div")) {
    out[words(item.querySelector("dt"))!] = words(item.querySelector("dd"))!;
  }
  return out;
}

describe("the quick-fact tiles on the goldens", () => {
  it("dry: wind, pressure and air from the current conditions", async () => {
    // dry current: 1.8 mph from 124°, gust 2.5; 29.673 inHg, steady;
    // 99% humidity, dew point 39.14°.
    const panel = await boot(parsed("dry"));
    expect(fact(panel, "wind")).toEqual(["Wind", "1.8 mph", "From the SE, gusting to 2.5 mph."]);
    expect(fact(panel, "pressure")).toEqual([
      "Pressure",
      "29.67",
      "inHg, steady over the last 3 hours.",
    ]);
    expect(fact(panel, "air")).toEqual(["Air", "99%", "Humidity, with the dew point at 39.1°."]);
  });

  it("raining: wind, pressure and air from the current conditions", async () => {
    // raining current: 3.0 mph from 180°, gust 6.0; 30.0 inHg, steady;
    // 66% humidity, dew point 54.3°.
    const panel = await boot(parsed("raining"));
    expect(fact(panel, "wind")).toEqual(["Wind", "3.0 mph", "From the S, gusting to 6.0 mph."]);
    expect(fact(panel, "pressure")).toEqual([
      "Pressure",
      "30.00",
      "inHg, steady over the last 3 hours.",
    ]);
    expect(fact(panel, "air")).toEqual(["Air", "66%", "Humidity, with the dew point at 54.3°."]);
  });

  it.each(["dry", "raining"] as const)(
    "%s: wind and humidity agree with the now tile",
    async (scenario) => {
      const panel = await boot(parsed(scenario));
      const strip = rightNow(panel);
      expect(fact(panel, "wind")[1]).toBe(strip.wind);
      expect(fact(panel, "air")[1]).toBe(strip.humidity);
    },
  );

  it("sit after the Overview's other tiles, one column each, in a row the sun and moon tile completes", async () => {
    const panel = await boot(parsed("dry"));
    const tiles = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")];
    expect(tiles.slice(-3).map((t) => t.dataset.section)).toEqual(["wind", "pressure", "air"]);
    for (const tile of tiles.slice(-3)) {
      expect(tile.dataset.span).toBe("1");
      expect(tile.dataset.fact).toBe("true");
    }
  });
});

describe("the pressure sentence follows the snapshot's 3-hour trend word", () => {
  it.each([
    ["rising", "inHg, rising over the last 3 hours."],
    ["falling", "inHg, falling over the last 3 hours."],
    ["steady", "inHg, steady over the last 3 hours."],
  ] as const)("%s", async (trend, sentence) => {
    // The reading is the same each time: only the snapshot's word moves it.
    const panel = await boot(withCurrent("dry", { barometer_trend: trend }));
    expect(fact(panel, "pressure")).toEqual(["Pressure", "29.67", sentence]);
  });
});

describe("a missing value reads as not known, never as zero", () => {
  it("wind speed", async () => {
    const panel = await boot(withCurrent("dry", { wind_speed_mph: null }));
    expect(fact(panel, "wind")).toEqual([
      "Wind",
      "n/a",
      "Wind speed not known. From the SE, gusting to 2.5 mph.",
    ]);
  });

  it("wind direction and gust", async () => {
    const panel = await boot(withCurrent("raining", { wind_dir_deg: null, wind_gust_mph: null }));
    expect(fact(panel, "wind")).toEqual(["Wind", "3.0 mph", "Direction and gust not known."]);
  });

  it("pressure and its trend", async () => {
    const panel = await boot(withCurrent("dry", { barometer_inhg: null, barometer_trend: null }));
    expect(fact(panel, "pressure")).toEqual([
      "Pressure",
      "n/a",
      "Reading and 3-hour trend not known.",
    ]);
  });

  it("the trend alone", async () => {
    const panel = await boot(withCurrent("dry", { barometer_trend: null }));
    expect(fact(panel, "pressure")).toEqual([
      "Pressure",
      "29.67",
      "inHg; the 3-hour trend is not known.",
    ]);
  });

  it("humidity and dew point", async () => {
    const panel = await boot(withCurrent("raining", { humidity_pct: null, dew_point_f: null }));
    expect(fact(panel, "air")).toEqual(["Air", "n/a", "Humidity and dew point not known."]);
  });

  it("the dew point alone", async () => {
    const panel = await boot(withCurrent("raining", { dew_point_f: null }));
    expect(fact(panel, "air")).toEqual(["Air", "66%", "Humidity; the dew point is not known."]);
  });

  it("a calm zero stays a zero", async () => {
    const panel = await boot(withCurrent("dry", { wind_speed_mph: 0, wind_gust_mph: 0 }));
    expect(fact(panel, "wind")).toEqual(["Wind", "0.0 mph", "From the SE, gusting to 0.0 mph."]);
  });
});

describe("the quick-fact tiles on a phone", () => {
  it("sit after the finding rows, open, with one cell left for sun and moon", async () => {
    viewport(390);
    const panel = await boot(parsed("raining"));
    expect(document.documentElement.dataset.layout).toBe("phone");
    const children = [...panel.querySelectorAll<HTMLElement>(".tile-grid > *")];
    const rows = children.findIndex((el) => el.classList.contains("tile-rows"));
    expect(rows).toBeGreaterThan(-1);
    const after = children.slice(rows + 1);
    expect(after.map((el) => el.dataset.section)).toEqual(["wind", "pressure", "air"]);
    for (const tile of after) {
      expect(tile.tagName).toBe("SECTION");
      expect(tile.dataset.fact).toBe("true");
      expect(tile.querySelector("details.tile-row")).toBeNull();
    }
    // Three facts in a grid of two leave the fourth cell for sun and moon.
    expect(after.length % 2).toBe(1);
  });
});
