// @vitest-environment happy-dom
/**
 * The other zooms (#477, spec #460): Sun, Last 10 days, Pressure, Humidity,
 * Daylight and Inside, through the page. Each tile opens its zoom; each
 * zoom carries today's card content at full size and no source note
 * (prototype decision 22); nothing the pre-mosaic cards said is lost: the
 * full pressure and Air sentences, the sunrise-to-sunset times, the
 * daylight sentence and the moon. The Sun zoom draws the Solar tab's own
 * Sunlight now section, then "More in Solar". A station with no location
 * has no Daylight tile, and the sources footer says why in its place. An
 * unusual humidity or pressure feature opens that reading's zoom.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { airSentence } from "../src/overview/zooms/humidity.js";
import { dayTrace } from "../src/overview/zooms/parts.js";
import { pressureSentence } from "../src/overview/zooms/pressure.js";
import { daylightZoom, moonSentence } from "../src/overview/zooms/daylight.js";
import { isZoomRoute } from "../src/overview/zoom.js";
import { ZOOMS } from "../src/overview/zooms/index.js";
import { MOSAIC_TILES } from "../src/overview/mosaic-tiles.js";
import { featureView } from "../src/overview/feature.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import { renderSolar } from "../src/solar/index.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const parsed = (): DashboardSnapshot =>
  parseDashboard(structuredClone(golden("seasoned", "dashboard")));

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
  vi.restoreAllMocks();
});

/** Boot the page on a hash, from a snapshot (the seasoned golden by default). */
async function boot(hash: string, snapshot = parsed(), width = 1440): Promise<DashboardSnapshot> {
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
  return snapshot;
}

const overview = () => document.getElementById("panel-overview")!;
const zoom = () => overview().querySelector<HTMLElement>(".zoom");
const panel = (id: string) => zoom()!.querySelector<HTMLElement>(`[data-panel="${id}"]`);
const words = (el: Element | null | undefined) =>
  (el?.textContent ?? "").replace(/\s+/g, " ").trim();

const MINE = [
  ["last-10-days", "Last 10 days"],
  ["pressure", "Pressure"],
  ["humidity", "Humidity"],
  ["sun", "Sun today"],
  ["daylight", "Daylight"],
  ["inside", "Inside"],
] as const;

describe("every tile of #477 opens a zoom", () => {
  it("registers the zooms in the mosaic's order", () => {
    const order = MOSAIC_TILES.map((t) => t.slug);
    const slugs = ZOOMS.map((z) => z.slug);
    expect(slugs).toEqual(order.filter((slug) => slugs.includes(slug)));
  });

  it.each(MINE)("%s: its tile controls its zoom, and its route opens it", async (slug, title) => {
    await boot("#overview");
    const link = overview().querySelector<HTMLAnchorElement>(
      `[data-section="${slug}"] a.mosaic-link`,
    )!;
    expect(link.getAttribute("href")).toBe(`#overview/${slug}`);
    expect(link.dataset.zoom).toBe(slug);
    expect(link.getAttribute("aria-controls")).toBe(`zoom-${slug}`);
    expect(isZoomRoute(`overview/${slug}`)).toBe(true);
    page!.stop();

    await boot(`#overview/${slug}`);
    const el = zoom()!;
    expect(el.dataset.zoom).toBe(slug);
    expect(words(el.querySelector(".zoom-heading"))).toBe(title);
    expect(el.querySelectorAll(".zoom-panel").length).toBeGreaterThan(0);
    // No source note or method fold in a zoom (prototype decision 22).
    expect(el.querySelector("details")).toBeNull();
    expect(el.querySelector("[data-part=source]")).toBeNull();
  });
});

describe("the Sun zoom", () => {
  it("draws the Solar tab's Sunlight now section, then More in Solar", async () => {
    const snapshot = await boot("#overview/sun");
    const sun = panel("sunlight-now")!;
    expect(words(sun.querySelector(".zoom-panel-title"))).toBe("Sunlight now");
    const finding = words(sun.querySelector("[data-part=finding]"));
    expect(sun.querySelector("figure.solar-chart svg")).not.toBeNull();
    expect(sun.querySelector(".solar-readout")).not.toBeNull();
    // "More in Solar" is the zoom's last panel, the one "More in" card.
    const more = panel("more")!;
    expect(sun.nextElementSibling).toBe(more);
    const link = more.querySelector<HTMLAnchorElement>("a.zoom-more")!;
    expect(link.getAttribute("href")).toBe("#solar");
    expect(words(link.querySelector(".zoom-panel-title"))).toBe("More in Solar");
    expect(link.querySelector("[aria-hidden=true]")?.textContent).toBe("›");

    // The Solar tab draws the same section, with its source note kept.
    const tab = document.createElement("div");
    document.body.append(tab);
    renderSolar(tab, snapshot, () => "");
    const now = tab.querySelector<HTMLElement>('[data-section="now"]')!;
    expect(words(now.querySelector("[data-part=finding]"))).toBe(finding);
    expect(now.querySelector("details")).not.toBeNull();
  });
});

describe("the Last 10 days zoom", () => {
  it("draws the ten-day high-low list at full width, no fold", async () => {
    await boot("#overview/last-10-days");
    const days = panel("ten-days")!;
    expect(days.dataset.wide).toBe("true");
    expect(words(days.querySelector("[data-part=finding]"))).toMatch(/^The last ten days/);
    expect(days.querySelector("[data-part=days-chart] svg")).not.toBeNull();
    expect(days.querySelector("details")).toBeNull();
  });
});

describe("the Pressure and Humidity zooms", () => {
  it("carry the Pressure card's full sentence", async () => {
    await boot("#overview/pressure");
    expect(words(panel("trend")!.querySelector("[data-part=sentence]"))).toBe(
      "30.00 inHg, steady over the last 3 hours.",
    );
  });

  it("carry the Air card's humidity and dew point sentence", async () => {
    await boot("#overview/humidity");
    expect(words(panel("air")!.querySelector("[data-part=sentence]"))).toBe(
      "Humidity 65%; dew point 51.5°.",
    );
  });

  it.each([
    ["pressure", "30.00"],
    ["humidity", "95%"],
  ])("%s: draws the tile's 24 hours at full size (#554)", async (slug, high) => {
    await boot(`#overview/${slug}`);
    const day = panel("day")!;
    expect(day.dataset.wide).toBe("true");
    expect(words(day.querySelector("h3"))).toBe("The last 24 hours");
    const box = day.querySelector("[data-part=day-line]")!;
    expect(box.getAttribute("aria-hidden")).toBe("true");
    const svg = box.querySelector("svg")!;
    expect(svg.querySelectorAll(".mosaic-line")).toHaveLength(1);
    expect(svg.querySelectorAll(".mosaic-now-dot")).toHaveLength(1);
    const labels = [...svg.querySelectorAll(".mosaic-label")].map(words);
    expect(labels[0]).toBe(high);
    expect(labels.slice(-2)).toEqual(["24 h ago", "now"]);
    // The now dot carries its value, like the wind rose's (#598).
    const now = svg.querySelectorAll(".zoom-now-label");
    expect(now).toHaveLength(1);
    expect(words(now[0])).toMatch(slug === "pressure" ? /^now \d\d\.\d\d$/ : /^now \d+%$/);
  });

  it("puts the now label on the side the line is not (#598)", () => {
    const at = (values: number[]) => {
      const svg = dayTrace(values, 400, (v) => v.toFixed(2), 0.05)!;
      const label = svg.querySelector(".zoom-now-label")!;
      const dot = svg.querySelector(".mosaic-now-dot")!;
      return Number(label.getAttribute("y")) - Number(dot.getAttribute("cy"));
    };
    expect(at([30, 30.1, 30.2, 30.0, 29.9, 29.95])).toBeLessThan(0);
    expect(at([30, 29.9, 29.8, 30.0, 30.1, 29.95])).toBeGreaterThan(0);
  });

  it.each([
    ["null", null],
    ["absent (N−1)", undefined],
  ])("draw no 24-hour panel when outdoor_trace is %s", async (_, trace) => {
    const raw = structuredClone(golden("seasoned", "dashboard")) as Record<string, unknown>;
    if (trace === undefined) delete raw.outdoor_trace;
    else raw.outdoor_trace = trace;
    await boot("#overview/pressure", parseDashboard(raw));
    expect(panel("trend")).not.toBeNull();
    expect(panel("day")).toBeNull();
  });

  it("say what is not known, never zero", () => {
    const s = parsed();
    s.current.barometer_inhg = null;
    s.current.humidity_pct = null;
    expect(pressureSentence(s)).toBe("Reading not known; steady over the last 3 hours.");
    expect(airSentence(s)).toBe("Humidity not known; dew point 51.5°.");
    s.current.barometer_trend = null;
    s.current.dew_point_f = null;
    expect(pressureSentence(s)).toBe("Reading and 3-hour trend not known.");
    expect(airSentence(s)).toBe("Humidity and dew point not known.");
  });
});

describe("the Daylight zoom", () => {
  it("carries sunrise to sunset, the arc, the daylight sentence and the moon", async () => {
    const s = await boot("#overview/daylight");
    const sun = panel("sun")!;
    expect(words(sun.querySelector("[data-part=sun-times]"))).toMatch(
      /^\d{1,2}:\d\d (am|pm) to \d{1,2}:\d\d (am|pm)$/,
    );
    expect(sun.querySelector("[data-part=arc] svg")).not.toBeNull();
    expect(words(sun.querySelector("[data-part=daylight]"))).toMatch(
      /^\d+ h \d+ m of daylight, (.+ (more|less) than yesterday|the same as yesterday)\.$/,
    );
    const moon = words(panel("moon")!.querySelector("[data-part=moon]"));
    expect(moon).toBe(moonSentence(s.sun_moon!));
    expect(moon).toMatch(/ moon, \d+% lit/);
  });

  it("lets its lead break between hours and minutes, never inside either", () => {
    const value = daylightZoom.lead!(parsed()).value!;
    expect(value).toMatch(/^\d+\u00a0h \d+\u00a0m$/);
  });

  it("words the moon's full day and its last days", () => {
    const block = parsed().sun_moon!;
    expect(
      moonSentence({
        ...block,
        moon_phase: "waxing gibbous",
        moon_illumination_pct: 97,
        days_to_full_moon: 1,
      }),
    ).toBe("Waxing gibbous moon, 97% lit, full in 1 day.");
    expect(
      moonSentence({
        ...block,
        moon_phase: "waxing gibbous",
        moon_illumination_pct: 99,
        days_to_full_moon: 0,
      }),
    ).toBe("Waxing gibbous moon, 99% lit, full today.");
    expect(
      moonSentence({
        ...block,
        moon_phase: "full",
        moon_illumination_pct: 100,
        days_to_full_moon: 0,
      }),
    ).toBe("Full moon, 100% lit.");
  });

  it("on a station with no location: no tile, and the footer says why", async () => {
    const s = parsed();
    s.sun_moon = null;
    await boot("#overview", s);
    expect(overview().querySelector('[data-section="daylight"]')).toBeNull();
    const line = overview().querySelector<HTMLElement>('.ov-sources [data-source="Daylight"]')!;
    expect(words(line.querySelector("dt"))).toBe("Daylight");
    expect(words(line.querySelector("dd"))).toBe("Not known: the station reports no location.");
    expect(line.querySelector(".ov-sources-method")).toBeNull();
  });

  it("on a located station, the footer gives Daylight's source, not the note", async () => {
    await boot("#overview");
    const line = overview().querySelector('.ov-sources [data-source="Daylight"]');
    expect(words(line)).not.toContain("no location");
    expect(line?.querySelector(".ov-sources-method")).not.toBeNull();
  });
});

describe("the Inside zoom", () => {
  it("draws each sensor's name, reading and 24 hours", async () => {
    const s = await boot("#overview/inside");
    const sensors = [...panel("sensors")!.querySelectorAll<HTMLElement>("[data-part=sensor]")];
    expect(sensors).toHaveLength(s.indoor!.length);
    sensors.forEach((el, i) => {
      const sensor = s.indoor![i]!;
      expect(words(el.querySelector(".zoom-sensor-name"))).toBe(sensor.name);
      expect(words(el.querySelector("[data-part=sensor-value]"))).toBe(
        `${sensor.temperature_f.toFixed(1)}°`,
      );
      // Read in the tile's own words and order: "68.0° humidity 45%".
      if (sensor.humidity_pct !== null)
        expect(words(el.querySelector(".zoom-sensor-reading"))).toBe(
          `${sensor.temperature_f.toFixed(1)}° humidity ${sensor.humidity_pct}%`,
        );
      expect(el.querySelector("[data-part=sensor-line] svg")).not.toBeNull();
    });
  });
});

describe("the feature's humidity and pressure links", () => {
  it.each([
    ["humidity", 22, "pct", "#overview/humidity"],
    ["pressure", 29.41, "inhg", "#overview/pressure"],
  ])("an unusual %s opens its zoom, a registered route", (reading, value, unit, href) => {
    const s = parsed();
    // The seasoned golden's unusual temperature, retold as this reading.
    s.feature = {
      ...s.feature!,
      reading,
      value,
      unit,
      percentile: 3,
    } as DashboardSnapshot["feature"];
    const view = featureView(s);
    expect(view?.target.href).toBe(href);
    expect(isZoomRoute(href)).toBe(true);
  });
});
