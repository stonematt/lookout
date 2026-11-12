// @vitest-environment happy-dom
/**
 * The mosaic (#472, #473, spec #460 stories 18–28, 52, 56), rendered from
 * the goldens: the ten tiles in their fixed order, a tile the record does
 * not cover left out with the later ones moving up, each tile one link to
 * `#overview/<slug>` named by its own text (the finding once, no
 * `aria-label`), its mini chart hidden from screen readers, and every number
 * the snapshot's: the Temperature tile's delta is the hero's. Air's and Sun
 * and moon's cards are gone, their content in Humidity, Sun and Daylight;
 * Inside draws a row per indoor sensor and hides on a station with none.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { setLayout } from "../src/layout.js";
import { renderOverview } from "../src/overview/index.js";
import { MOSAIC_TILES } from "../src/overview/mosaic-tiles.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { SCENARIOS, golden } from "./golden.js";

type Doc = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

const doc = (scenario: Scenario): Doc => structuredClone(golden(scenario, "dashboard")) as Doc;
const parsed = (scenario: Scenario): DashboardSnapshot => parseDashboard(doc(scenario));

const MOSAIC = [
  "temperature",
  "rain",
  "last-10-days",
  "dry-spell",
  "wind",
  "pressure",
  "humidity",
  "sun",
  "daylight",
  "inside",
];

let panel: HTMLElement;

beforeEach(() => {
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const tile = (slug: string) => panel.querySelector<HTMLElement>(`[data-section="${slug}"]`)!;
const ids = () =>
  [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map((t) => t.dataset.section);
const mosaicIds = () => ids().filter((id) => MOSAIC.includes(id!));

/**
 * An element's accessible name from its content, as a browser computes it
 * for a link: the text of every node not hidden by `aria-hidden`, element
 * by element.
 */
function nameFrom(el: Element): string {
  const parts: string[] = [];
  const walk = (node: Node) => {
    if (node.nodeType === Node.TEXT_NODE) parts.push(node.textContent ?? "");
    else if (node instanceof Element && node.getAttribute("aria-hidden") !== "true") {
      for (const child of node.childNodes) walk(child);
      parts.push(" ");
    }
  };
  walk(el);
  return parts.join("").replace(/\s+/g, " ").trim();
}

describe("the mosaic's order", () => {
  it.each(SCENARIOS)("follows the feature on the %s golden: all ten tiles, Inside last", (s) => {
    renderOverview(panel, parsed(s));

    expect(ids()).toEqual(["first-screen", "feature", ...MOSAIC]);
    expect(MOSAIC_TILES.map((t) => t.slug)).toEqual(MOSAIC);
  });

  it.each(["phone", "two", "four"] as const)("keeps the phone's order at %s", (layout) => {
    setLayout(layout);
    renderOverview(panel, parsed("seasoned"));

    expect(mosaicIds()).toEqual(MOSAIC);
  });

  it("leaves out a tile the record does not cover, and the later tiles move up", () => {
    const d = doc("raining");
    for (const t of d.record_depth.tiles)
      if (t.key === "rain" || t.key === "dry_spell") t.shown = false;
    renderOverview(panel, parseDashboard(d));

    expect(mosaicIds()).toEqual(MOSAIC.filter((slug) => slug !== "rain" && slug !== "dry-spell"));
    expect(ids().indexOf("last-10-days")).toBe(ids().indexOf("temperature") + 1);
    expect(panel.querySelector('[href="#overview/rain"]')).toBeNull();
  });

  it("draws every tile for a snapshot without record depth (N−1)", () => {
    const d = doc("dry");
    delete d.record_depth;
    renderOverview(panel, parseDashboard(d));

    expect(mosaicIds()).toEqual(MOSAIC);
  });
});

describe.each(SCENARIOS)("each mosaic tile on the %s golden", (scenario) => {
  beforeEach(() => renderOverview(panel, parsed(scenario)));

  it.each(MOSAIC_TILES.map((t) => [t.slug, t] as const))(
    "%s is one link to its zoom, named by its own text, once",
    (slug, def) => {
      const host = tile(slug);
      const links = host.querySelectorAll("a");

      expect(links).toHaveLength(1);
      const link = links[0]!;
      expect([...host.children]).toEqual([link]);
      expect(link.getAttribute("href")).toBe(`#overview/${slug}`);
      expect(host.id).toBe(`overview-${slug}`);
      // No aria-label anywhere in the tile: nothing repeats what the link reads.
      expect(host.hasAttribute("aria-label")).toBe(false);
      expect(host.querySelector("[aria-label]")).toBeNull();
      // Its name: the title, the number and the delta, then where it goes.
      const name = nameFrom(link);
      const value = words(link.querySelector("[data-part=value]"));
      const delta = words(link.querySelector("[data-part=delta]"));
      expect(name.startsWith(`${def.title} `)).toBe(true);
      expect(name.endsWith(`Open the ${def.zoomName} view`)).toBe(true);
      if (value) expect(name).toContain(value);
      if (delta) expect(name).toContain(delta);
      const list = link.querySelector("[data-part=rows]");
      const rows = list ? nameFrom(list) : null;
      if (rows) expect(name).toContain(rows);
      expect(value ?? delta ?? rows).toBeTruthy();
      // The finding is said once: the number appears once in the name.
      if (value) expect(name.split(value).length - 1).toBe(1);
      // The mini chart is drawn for sight alone.
      const chart = link.querySelector("[data-part=mini-chart]")!;
      expect(chart.getAttribute("aria-hidden")).toBe("true");
      for (const svg of chart.querySelectorAll("svg")) {
        expect(svg.getAttribute("aria-hidden")).toBe("true");
      }
      // Nothing inside the link is interactive in its own right.
      expect(link.querySelector("a, button, details, input")).toBeNull();
    },
  );

  it("titles each tile with a heading inside its link", () => {
    const titles = MOSAIC.map((slug) => tile(slug).querySelector("a > h2"));

    expect(titles.map(words)).toEqual(MOSAIC_TILES.map((t) => t.title));
  });
});

describe("the Temperature tile", () => {
  it("prints the hero's reading and the hero's delta on usual, one value", () => {
    const seasoned = parsed("seasoned");
    renderOverview(panel, seasoned);
    const hero = words(panel.querySelector("[data-section=first-screen] [data-part=delta]"))!;
    const delta = words(tile("temperature").querySelector("[data-part=delta]"))!;

    expect(seasoned.temperature_context.usual_band!.now_delta!.delta_f).toBe(12);
    expect(hero).toBe("+12° on usual for 3 pm");
    expect(delta).toBe("high 72°, +12° on usual");
    expect(words(tile("temperature").querySelector("[data-part=value]"))).toBe("72.0°");
  });

  it("says at usual for a delta of 0, as the hero does", () => {
    renderOverview(panel, parsed("raining"));

    expect(words(tile("temperature").querySelector("[data-part=delta]"))).toBe(
      "high 60°, at usual",
    );
  });

  it.each([null, undefined])("prints no delta for a %s now_delta, never one of its own", (gone) => {
    const d = doc("seasoned");
    if (gone === null) d.temperature_context.usual_band.now_delta = null;
    else delete d.temperature_context.usual_band.now_delta;
    renderOverview(panel, parseDashboard(d));

    expect(words(tile("temperature").querySelector("[data-part=delta]"))).toBe("high 72°");
    expect(words(tile("temperature").querySelector("[data-part=delta]"))).not.toMatch(/usual/);
  });

  it("names no usual row a young record cannot draw (raining)", () => {
    renderOverview(panel, parsed("raining"));
    const svg = tile("temperature").querySelector("svg")!;

    expect([...svg.querySelectorAll(".mosaic-label")].map(words)).toEqual(["today"]);
    expect(svg.querySelectorAll(".mosaic-band")).toHaveLength(0);
  });

  it("draws the usual and today rows, and the now rule", () => {
    renderOverview(panel, parsed("seasoned"));
    const svg = tile("temperature").querySelector("svg")!;

    expect([...svg.querySelectorAll(".mosaic-label")].map(words)).toEqual(["usual", "today"]);
    expect(svg.querySelectorAll(".mosaic-band")).toHaveLength(1);
    expect(svg.querySelectorAll(".mosaic-iqr")).toHaveLength(1);
    expect(svg.querySelectorAll(".mosaic-now")).toHaveLength(1);
  });
});

describe("the Rain tile", () => {
  it("says not compared yet when this water year has no total to date", () => {
    renderOverview(panel, parsed("seasoned"));

    expect(tile("rain").querySelector("[data-part=value]")).toBeNull();
    expect(words(tile("rain").querySelector("[data-part=delta]"))).toBe("not compared yet");
  });

  it("prints this year to date against the newest earlier year by the same date", () => {
    const d = doc("seasoned");
    const tab = d.rain_tab;
    const at = tab.day_index;
    for (const [i, curve] of tab.water_years.entries()) {
      curve.cumulative_in = curve.cumulative_in.map((_: unknown, p: number) => (p * (i + 1)) / 100);
      curve.prorated = curve.cumulative_in.map(() => false);
      curve.to_date_in = curve.cumulative_in[at] ?? null;
    }
    renderOverview(panel, parseDashboard(d));
    const years = tab.water_years;
    const now = years.at(-1);
    const before = years.at(-2);

    expect(words(tile("rain").querySelector("[data-part=value]"))).toBe(
      `${now.to_date_in.toFixed(2)} in`,
    );
    expect(words(tile("rain").querySelector("[data-part=delta]"))).toBe(
      `${before.water_year} had ${before.cumulative_in[at].toFixed(2)} in`,
    );
    const svg = tile("rain").querySelector("svg")!;
    expect(svg.querySelectorAll(".mosaic-curve-now")).toHaveLength(1);
    expect(svg.querySelectorAll(".mosaic-now-dot")).toHaveLength(1);
  });
});

describe("the Last 10 days tile", () => {
  it("prints the newest day's high against its usual high", () => {
    renderOverview(panel, parsed("seasoned"));

    expect(words(tile("last-10-days").querySelector("[data-part=value]"))).toBe("72°");
    expect(words(tile("last-10-days").querySelector("[data-part=delta]"))).toBe(
      "today's high, usual 60°",
    );
    // One bar a day the record covers.
    expect(tile("last-10-days").querySelectorAll("svg .mosaic-day")).toHaveLength(10);
    // The band under the bars is named by a legend swatch, not in the plot.
    const legend = tile("last-10-days").querySelector("[data-part=legend]")!;
    expect(words(legend)).toBe("usual range");
    expect(legend.closest("[aria-hidden=true]")).not.toBeNull();
    expect(tile("last-10-days").querySelector("svg text")).toBeNull();
  });
});

describe("the Dry spell tile", () => {
  it("prints the snapshot's dry spell as the Rain tab and the feature do", () => {
    const d = doc("dry");
    d.dry_spell = { days_since_rain: 9.4, usual_days: 3 };
    renderOverview(panel, parseDashboard(d));

    expect(words(tile("dry-spell").querySelector("[data-part=value]"))).toBe("9 d 9 h");
    expect(words(tile("dry-spell").querySelector("[data-part=delta]"))).toBe("usual is 3 days");
  });

  it("reads the calm lead rule in a snapshot without dry_spell (N−1)", () => {
    const d = doc("seasoned");
    delete d.dry_spell;
    const calm = d.hero_lead.rules.find((r: Doc) => r.kind === "calm");
    calm.reference = 2;
    renderOverview(panel, parseDashboard(d));

    expect(words(tile("dry-spell").querySelector("[data-part=value]"))).toBe("3 d 1 h");
    expect(words(tile("dry-spell").querySelector("[data-part=delta]"))).toBe("usual is 2 days");
  });

  it("draws the last 30 days, today outlined", () => {
    renderOverview(panel, parsed("seasoned"));
    const svg = tile("dry-spell").querySelector("svg")!;

    const days = [...svg.children].filter(
      (el) => el.tagName.toLowerCase() === "rect" && !el.classList.contains("mosaic-today"),
    );
    expect(days).toHaveLength(30);
    expect(svg.querySelectorAll(".mosaic-today")).toHaveLength(1);
    expect(svg.querySelectorAll(".mosaic-wet").length).toBe(2);
  });
});

describe("the quick-fact cards are gone, their content in the tiles", () => {
  it.each(SCENARIOS)("%s: no Wind, Pressure, Air or Sun and moon card", (s) => {
    renderOverview(panel, parsed(s));

    expect(panel.querySelector("[data-section=air], [data-section=sun-moon]")).toBeNull();
    expect(panel.querySelector("[data-fact], .qf-value, .qf-sentence")).toBeNull();
    expect(panel.querySelectorAll(".tile-grid > .tile:not([data-mosaic])")).toHaveLength(2);
  });
});

const value = (slug: string) => words(tile(slug).querySelector("[data-part=value]"));
const delta = (slug: string) => words(tile(slug).querySelector("[data-part=delta]"));

describe("the Wind tile", () => {
  it("prints the live speed, where it blows from and its gust (dry)", () => {
    renderOverview(panel, parsed("dry"));

    expect(value("wind")).toBe("1.8 mph");
    expect(delta("wind")).toBe("from the SE, gusts to 2.5 mph");
    expect(tile("wind").querySelectorAll("svg .mosaic-arrow-head")).toHaveLength(1);
  });

  it("reads a missing speed, direction and gust as not known, never zero", () => {
    const d = doc("dry");
    Object.assign(d.current, { wind_speed_mph: null, wind_dir_deg: null, wind_gust_mph: null });
    renderOverview(panel, parseDashboard(d));

    expect(value("wind")).toBe("n/a");
    expect(delta("wind")).toBe("direction and gust not known");
    expect(tile("wind").querySelector("svg")).toBeNull();
  });

  it("keeps a calm zero a zero", () => {
    const d = doc("dry");
    d.current.wind_speed_mph = 0;
    renderOverview(panel, parseDashboard(d));

    expect(value("wind")).toBe("0.0 mph");
  });
});

describe("the Pressure tile", () => {
  it("prints the barometer and the snapshot's 3-hour trend word", () => {
    const d = doc("dry");
    d.current.barometer_trend = "falling";
    renderOverview(panel, parseDashboard(d));

    expect(value("pressure")).toBe("29.67 inHg");
    expect(delta("pressure")).toBe("falling over 3 hours");
  });

  it("reads a missing reading and trend as not known", () => {
    const d = doc("dry");
    Object.assign(d.current, { barometer_inhg: null, barometer_trend: null });
    renderOverview(panel, parseDashboard(d));

    expect(value("pressure")).toBe("n/a");
    expect(delta("pressure")).toBe("3-hour trend not known");
  });
});

describe("the Humidity tile, which takes the Air card", () => {
  it("prints the humidity and the dew point (dry)", () => {
    renderOverview(panel, parsed("dry"));

    expect(value("humidity")).toBe("99%");
    expect(delta("humidity")).toBe("dew point 39.1°");
  });

  it("reads a missing humidity and dew point as not known", () => {
    const d = doc("dry");
    Object.assign(d.current, { humidity_pct: null, dew_point_f: null });
    renderOverview(panel, parseDashboard(d));

    expect(value("humidity")).toBe("n/a");
    expect(delta("humidity")).toBe("dew point not known");
  });
});

describe("the Sun tile", () => {
  it("prints today's energy so far and yesterday's, from the Solar tab's windows", () => {
    renderOverview(panel, parsed("seasoned"));

    expect(words(tile("sun").querySelector("h2"))).toBe("Sun today");
    expect(value("sun")).toBe("4.92 kWh/m²");
    expect(delta("sun")).toBe("all of yesterday 6.11 kWh/m²");
  });

  it("draws the last 24 hours over the usual range, now as the dot", () => {
    renderOverview(panel, parsed("seasoned"));
    const svg = tile("sun").querySelector("svg")!;

    expect(svg.querySelectorAll(".mosaic-sun-line").length).toBeGreaterThan(0);
    expect(svg.querySelectorAll(".mosaic-now-dot")).toHaveLength(1);
  });

  it("is not drawn without a solar record", () => {
    const d = doc("dry");
    d.solar_tab = null;
    renderOverview(panel, parseDashboard(d));

    expect(mosaicIds()).not.toContain("sun");
    expect(mosaicIds().at(-2)).toBe("daylight");
  });
});

describe("the Daylight tile, which takes the sun and moon card's times", () => {
  it("prints the day's length and its change, with sunrise and sunset in station time", () => {
    renderOverview(panel, parsed("dry"));

    expect(value("daylight")).toBe("9 h 23 m");
    expect(delta("daylight")).toBe("2 m less than yesterday");
    const labels = [...tile("daylight").querySelectorAll("svg .mosaic-label")].map(words);
    expect(labels).toEqual(["7:15 am", "4:38 pm"]);
    // The chart is for sight alone, so the link says the two times.
    expect(nameFrom(tile("daylight").querySelector("a")!)).toContain(
      "2 m less than yesterday sunrise 7:15 am, sunset 4:38 pm Open the daylight view",
    );
    // Before sunrise: the arc, and no sun on it.
    expect(tile("daylight").querySelector(".mosaic-now-dot")).toBeNull();
  });

  it("says a lengthening day, and puts the sun on the arc in daylight (seasoned)", () => {
    renderOverview(panel, parsed("seasoned"));

    expect(delta("daylight")).toBe("2 m more than yesterday");
    expect(tile("daylight").querySelectorAll(".mosaic-now-dot")).toHaveLength(1);
    expect(tile("daylight").querySelectorAll(".mosaic-arc-day")).toHaveLength(1);
  });

  it("is not drawn for a station with no location", () => {
    const d = doc("dry");
    d.sun_moon = null;
    renderOverview(panel, parseDashboard(d));

    expect(mosaicIds()).not.toContain("daylight");
  });
});

describe("the Inside tile", () => {
  it("draws one row per indoor sensor: its name, reading, humidity and line", () => {
    renderOverview(panel, parsed("seasoned"));
    const rows = [...tile("inside").querySelectorAll<HTMLElement>("[data-part=row]")];

    expect(rows.map((r) => words(r.querySelector(".mosaic-row-name")))).toEqual([
      "Console",
      "Sensor 1",
    ]);
    expect(rows.map((r) => words(r.querySelector(".mosaic-row-reading")))).toEqual([
      "68.0° humidity 45%",
      "66.0° humidity 55%",
    ]);
    for (const row of rows) {
      const line = row.querySelector(".mosaic-row-line")!;
      expect(line.getAttribute("aria-hidden")).toBe("true");
      expect(line.querySelectorAll(".mosaic-now-dot")).toHaveLength(1);
    }
    expect(nameFrom(tile("inside").querySelector("a")!)).toBe(
      "Inside Console 68.0° humidity 45% Sensor 1 66.0° humidity 55% Open the inside view",
    );
  });

  it("keeps the row without humidity when a sensor reports none", () => {
    const d = doc("raining");
    d.indoor[0].humidity_pct = null;
    renderOverview(panel, parseDashboard(d));

    expect(words(tile("inside").querySelector(".mosaic-row-reading"))).toBe("68.0°");
  });

  it.each([
    ["an empty list", []],
    ["no indoor field (N−1)", undefined],
  ])("hides on a station with no indoor sensor: %s", (_, indoor) => {
    const d = doc("seasoned");
    if (indoor === undefined) delete d.indoor;
    else d.indoor = indoor;
    renderOverview(panel, parseDashboard(d));

    expect(mosaicIds()).toEqual(MOSAIC.filter((slug) => slug !== "inside"));
    expect(panel.querySelector('[href="#overview/inside"]')).toBeNull();
  });
});

describe("the sources footer", () => {
  it("lists a line for each of the ten tiles drawn", () => {
    renderOverview(panel, parsed("seasoned"));
    const names = [...panel.querySelectorAll("[data-part=sources] dt")].map(words);

    for (const t of MOSAIC_TILES) expect(names).toContain(t.title);
  });
});

describe("the Overview's landmarks and headings, through the page", () => {
  let page: Page | null = null;

  afterEach(() => {
    page?.stop();
    page = null;
  });

  async function boot(snapshot: DashboardSnapshot, hash = "#overview"): Promise<void> {
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

  it("sits in main, under the page's one h1, the station name, with a heading on each tile", async () => {
    const snapshot = parsed("seasoned");
    await boot(snapshot);
    const overview = document.getElementById("panel-overview")!;

    expect(overview.closest("main")).not.toBeNull();
    const h1 = document.querySelectorAll("h1");
    expect(h1).toHaveLength(1);
    expect(words(h1[0])).toBe(snapshot.station.name);
    const headings = [...overview.querySelectorAll("h2")].map(words);
    for (const t of MOSAIC_TILES) expect(headings).toContain(t.title);
  });

  it("keeps the Overview open on a zoom link nothing handles yet", async () => {
    await boot(parsed("seasoned"));
    location.hash = "#overview/temperature";
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    expect(document.getElementById("panel-overview")!.hidden).toBe(false);
    expect(
      document.querySelector("[aria-controls=panel-overview]")!.getAttribute("aria-selected"),
    ).toBe("true");
  });
});
