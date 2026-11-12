// @vitest-environment happy-dom
/**
 * The Overview's quick-fact tiles (#340), through `startPage` on the
 * goldens: wind, pressure and air, each a kicker, one large value and one
 * sentence, read from the snapshot's current conditions. A missing value
 * reads as not known, never as zero; the pressure sentence follows the
 * snapshot's 3-hour trend word; wind and humidity agree with the now tile.
 * On a phone they sit after the finding rows, open, two to a row. The sun
 * and moon tile (#341) draws the snapshot's SunMoon block in station-local
 * time; a station with no coordinates (`sun_moon` null) keeps the kicker,
 * shows no value and says the station reports no location.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { setLayout } from "../src/layout.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot, SnapshotConditions } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { golden } from "./golden.js";
import { viewport } from "./viewport.js";

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

beforeEach(() => {
  history.replaceState(null, "", "/#overview");
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
  viewport(1440).install();
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

describe("the quick-fact tiles on the goldens", () => {
  it("dry: wind, pressure and air from the current conditions", async () => {
    // dry current: 1.8 mph from 124°, gust 2.5; 29.673 inHg, steady;
    // 99% humidity, dew point 39.14°.
    const panel = await boot(parsed("dry"));
    expect(fact(panel, "wind")).toEqual(["Wind", "1.8 mph", "From the SE, gusts to 2.5 mph."]);
    expect(fact(panel, "pressure")).toEqual([
      "Pressure",
      "29.67",
      "inHg, steady over the last 3 hours.",
    ]);
    expect(fact(panel, "air")).toEqual(["Air", "99%", "Humidity; dew point 39.1°."]);
  });

  it("raining: wind, pressure and air from the current conditions", async () => {
    // raining current: 3.0 mph from 180°, gust 6.0; 30.0 inHg, steady;
    // 66% humidity, dew point 54.3°.
    const panel = await boot(parsed("raining"));
    expect(fact(panel, "wind")).toEqual(["Wind", "3.0 mph", "From the S, gusts to 6.0 mph."]);
    expect(fact(panel, "pressure")).toEqual([
      "Pressure",
      "30.00",
      "inHg, steady over the last 3 hours.",
    ]);
    expect(fact(panel, "air")).toEqual(["Air", "66%", "Humidity; dew point 54.3°."]);
  });

  it("sit after the Overview's other tiles, a row of four one-column tiles", async () => {
    const panel = await boot(parsed("dry"));
    const tiles = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")];
    expect(tiles.slice(-4).map((t) => t.dataset.section)).toEqual([
      "wind",
      "pressure",
      "air",
      "sun-moon",
    ]);
    for (const tile of tiles.slice(-4)) {
      expect(tile.dataset.span).toBe("1");
      expect(tile.dataset.fact).toBe("true");
    }
  });
});

describe("the sun and moon tile", () => {
  it("dry: sunrise to sunset in station time, the daylight change and the moon", async () => {
    // dry sun_moon, America/Los_Angeles: sunrise 7:15 am, sunset 4:38 pm,
    // 563 minutes of daylight, 2 fewer than yesterday; new moon, 0% lit,
    // full in 15 days.
    const panel = await boot(parsed("dry"));
    expect(fact(panel, "sun-moon")).toEqual([
      "Sun and moon",
      "7:15 am to 4:38 pm",
      "9 h 23 m of daylight, 2 m less than yesterday; new moon, 0% lit, full in 15 days.",
    ]);
  });

  it("raining: a lengthening day and a waning moon", async () => {
    // raining sun_moon: 7:49 am to 4:48 pm, 538 minutes, 1 more than
    // yesterday; waning gibbous, 68% lit, full in 24 days.
    const panel = await boot(parsed("raining"));
    expect(fact(panel, "sun-moon")).toEqual([
      "Sun and moon",
      "7:49 am to 4:48 pm",
      "8 h 58 m of daylight, 1 m more than yesterday; waning gibbous moon, 68% lit, full in 24 days.",
    ]);
  });

  it("keeps each time whole, so the value breaks only between the two", async () => {
    const panel = await boot(parsed("dry"));
    const value = panel.querySelector(".tile[data-section=sun-moon] [data-part=fact]")!;
    expect(value.textContent).toBe("7:15\u00a0am to\u00a04:38\u00a0pm");
    // am, pm and "to" are set as units, like the wind's mph.
    expect([...value.querySelectorAll(".qf-unit")].map((u) => u.textContent)).toEqual([
      "am",
      "to",
      "pm",
    ]);
  });

  it("reads the times in the station's zone, not the browser's", async () => {
    const copy = structuredClone(parsed("dry"));
    copy.station.tz = "America/New_York";
    const panel = await boot(copy);
    expect(fact(panel, "sun-moon")[1]).toBe("10:15 am to 7:38 pm");
  });

  it.each([
    [0, "the same as yesterday"],
    [3, "3 m more than yesterday"],
    [-1, "1 m less than yesterday"],
  ])("a daylight change of %i reads %s", async (change, words) => {
    const copy = structuredClone(parsed("dry"));
    copy.sun_moon!.daylight_change_minutes = change;
    const panel = await boot(copy);
    expect(fact(panel, "sun-moon")[2]).toContain(`9 h 23 m of daylight, ${words}; `);
  });

  it.each([
    ["full", 100, 0, "full moon, 100% lit."],
    ["waxing gibbous", 97, 0, "waxing gibbous moon, 97% lit, full today."],
    ["first quarter", 50, 1, "first quarter moon, 50% lit, full in 1 day."],
  ] as const)("a %s moon, %i%% lit, %i days to full", async (phase, lit, days, words) => {
    const copy = structuredClone(parsed("dry"));
    Object.assign(copy.sun_moon!, {
      moon_phase: phase,
      moon_illumination_pct: lit,
      days_to_full_moon: days,
    });
    const panel = await boot(copy);
    expect(fact(panel, "sun-moon")[2]).toBe(
      `9 h 23 m of daylight, 2 m less than yesterday; ${words}`,
    );
  });

  it("with no coordinates keeps its kicker, shows no value and says the station reports no location", async () => {
    // Both goldens carry the block; the no-location case is a copy with it null.
    const copy = structuredClone(parsed("raining"));
    copy.sun_moon = null;
    const panel = await boot(copy);
    const tile = panel.querySelector<HTMLElement>(".tile[data-section=sun-moon]")!;
    expect(tile.dataset.fact).toBe("true");
    expect(fact(panel, "sun-moon")).toEqual([
      "Sun and moon",
      undefined,
      "Not known: the station reports no location.",
    ]);
    expect(tile.textContent).not.toMatch(/\d/);
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
      "Wind speed not known. From the SE, gusts to 2.5 mph.",
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
    expect(fact(panel, "air")).toEqual(["Air", "66%", "Humidity; dew point not known."]);
  });

  it("a calm zero stays a zero", async () => {
    const panel = await boot(withCurrent("dry", { wind_speed_mph: 0, wind_gust_mph: 0 }));
    expect(fact(panel, "wind")).toEqual(["Wind", "0.0 mph", "From the SE, gusts to 0.0 mph."]);
  });
});

describe("the quick-fact tiles on a phone", () => {
  it("sit after the finding rows, open, a 2×2 that sun and moon completes", async () => {
    viewport(390).install();
    const panel = await boot(parsed("raining"));
    expect(document.documentElement.dataset.layout).toBe("phone");
    const children = [...panel.querySelectorAll<HTMLElement>(".tile-grid > *")];
    const rows = children.findIndex((el) => el.classList.contains("tile-rows"));
    expect(rows).toBeGreaterThan(-1);
    const after = children.slice(rows + 1);
    expect(after.map((el) => el.dataset.section)).toEqual(["wind", "pressure", "air", "sun-moon"]);
    for (const tile of after) {
      expect(tile.tagName).toBe("SECTION");
      expect(tile.dataset.fact).toBe("true");
      expect(tile.querySelector("details.tile-row")).toBeNull();
    }
    // Four facts fill a grid of two: no empty cell.
    expect(after.length % 2).toBe(0);
  });
});
