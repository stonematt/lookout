// @vitest-environment happy-dom
/**
 * Overview C, the rain tile (#661, spec #647 "Overview C"; boards MainDark,
 * Desktop, StoppedC, DesktopStopped): while rain leads, one tile under the
 * now tile carries the now story over the season, and Wet hours and Rain
 * since Oct 1 leave the mosaic. Raining renders from the raining golden; the
 * stopped state is no golden's (each golden is mid event or past the 3
 * hours), so it renders from the raining golden with `rain_now` edited to a
 * stop 40 minutes before the build. Later and dry states render the mosaic
 * of decision 24.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it, vi } from "vitest";

import { setLayout } from "../src/layout.js";
import { renderOverview } from "../src/overview/index.js";
import { followSeason, isSeasonRoute } from "../src/overview/rain-lead.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { nameFrom } from "./accessible-name.js";
import { contrastPairs } from "./contrast.js";
import { golden } from "./golden.js";

type Doc = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

const doc = (): Doc => structuredClone(golden("raining", "dashboard")) as Doc;
const raining = (): DashboardSnapshot => parseDashboard(doc());

/** The raining golden, rain stopped `minutes` before the build. */
function stopped(minutes = 40): DashboardSnapshot {
  const d = doc();
  d.rain_now = {
    ...d.rain_now,
    raining: false,
    last_rain_ms: d.built_at_ms - minutes * 60_000,
    rate_in_hr: 0.03,
  };
  return parseDashboard(d);
}

let panel: HTMLElement;

beforeEach(() => {
  setLayout("phone");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
  history.replaceState(null, "", "/");
});

const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const tile = () => panel.querySelector<HTMLElement>(".tile[data-section=rain]:not([data-mosaic])");
const now = () => tile()!.querySelector<HTMLAnchorElement>("a[data-part=now]")!;
const season = () => tile()!.querySelector<HTMLAnchorElement>("a[data-part=season]")!;
const ids = () =>
  [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map((t) => t.dataset.section);
const svgText = () => [...now().querySelectorAll("svg text")].map((t) => t.textContent);

describe("the rain tile, raining (the raining golden)", () => {
  beforeEach(() => renderOverview(panel, raining()));

  it("sits under the now tile, in the feature's place", () => {
    expect(ids().slice(0, 2)).toEqual(["first-screen", "rain"]);
    expect(panel.querySelector("[data-section=feature]")).toBeNull();
  });

  it("is two links: the now part to the rain zoom, the season foot to the Rain tab's season", () => {
    expect([...tile()!.children]).toEqual([now(), season()]);
    expect(now().getAttribute("href")).toBe("#overview/rain");
    expect(season().getAttribute("href")).toBe("#rain/season");
    for (const link of [now(), season()]) {
      expect(link.querySelector("a, button, details, input")).toBeNull();
    }
  });

  it("labels the tile with a filled dot and Raining", () => {
    const kicker = now().querySelector("[data-part=kicker]")!;
    expect(words(kicker)).toBe("Raining");
    expect(kicker.querySelector("[data-part=live]")).not.toBeNull();
    expect(kicker.querySelector("[data-part=stopped]")).toBeNull();
  });

  it("tells the story: the rate in in/hr, then the event's rain since its start", () => {
    expect(words(now().querySelector("[data-part=value]"))).toBe("0.12in/hr");
    expect(words(now().querySelector("[data-part=finding]"))).toBe("0.30 in since 2:00 pm");
    expect(nameFrom(now())).toBe("Raining, 0.12 in/hr: 0.30 in since 2:00 pm Open the rain view.");
    expect(words(now())).not.toContain("in/h ");
  });

  it("draws the dual chart: a column per 5 minutes with rain, the line labelled in place", () => {
    const slots = raining().rain_now!.last_3_hours!;
    const wet = slots.rain_in.filter((v) => v !== null && v > 0).length;
    expect(now().querySelectorAll("rect.rain-lead-col")).toHaveLength(wet);
    expect(now().querySelectorAll("path.rain-lead-line")).toHaveLength(1);
    expect(svgText()).toContain("0.30 in so far");
    expect(svgText()).toContain("last hour 0.12 in");
  });

  it("keeps the axes at 10:1, tick for tick", () => {
    const left = [...now().querySelectorAll<SVGTextElement>("text.rain-lead-tick")]
      .filter((t) => t.getAttribute("text-anchor") === "end")
      .map((t) => parseFloat(t.textContent!.replace(/^\./, "0.")));
    const right = [...now().querySelectorAll<SVGTextElement>("text.rain-lead-tick")]
      .filter((t) => t.getAttribute("text-anchor") !== "end")
      .map((t) => parseFloat(t.textContent!.replace(/^\./, "0.")));
    expect(left.length).toBeGreaterThan(0);
    expect(right.map((v) => Math.round(v * 1000))).toEqual(left.map((v) => Math.round(v * 10_000)));
  });

  it("hatches the 5 minutes with no reading, never dry", () => {
    const slots = raining().rain_now!.last_3_hours!;
    expect(slots.rain_in.some((v) => v === null)).toBe(true);
    expect(now().querySelectorAll("rect.rain-lead-gap").length).toBeGreaterThan(0);
  });

  it("draws at its real width, its text 12px, never a scaled viewBox", () => {
    const svg = now().querySelector<SVGSVGElement>("svg.rain-lead-svg")!;
    expect(svg.getAttribute("viewBox")).toBe(
      `0 0 ${svg.getAttribute("width")} ${svg.getAttribute("height")}`,
    );
    const css = readFileSync(resolve(import.meta.dirname, "../src/overview/rain-lead.css"), "utf8");
    for (const cls of ["tick", "time", "label", "bracket-text"]) {
      const rule = new RegExp(`\\.rain-lead-${cls}[^{]*\\{([^}]*)\\}`).exec(css)![1]!;
      expect(rule, cls).toContain("font-size: var(--type-label)");
    }
    const tokens = readFileSync(resolve(import.meta.dirname, "../src/styles/tokens.css"), "utf8");
    expect(tokens).toMatch(/--type-label: 12px;/);
  });

  it("draws the chart and its legend for sight alone: the link says the story", () => {
    expect(now().querySelector("svg")!.getAttribute("aria-hidden")).toBe("true");
    expect(now().querySelector(".rain-lead-legend")!.getAttribute("aria-hidden")).toBe("true");
    expect([...now().querySelectorAll(".rain-lead-key")].map(words)).toEqual([
      "rain every 5 min (left)",
      "total since 2:00 (right)",
    ]);
  });

  it("carries the wet-hour bars in the board's words, then Rain since Oct 1, in its foot", () => {
    expect(words(season().querySelector(".wet-meter-title"))).toBe(
      "Wet hours · each bar is every hour; dark is wet",
    );
    expect([...season().querySelectorAll("[data-part=count]")].map(words)).toEqual([
      "12 h of 111",
      "12 h of 111",
    ]);
    expect(words(season().querySelector("[data-part=hatch-key]"))).toBe(
      "hatched: missing, not dry",
    );
    expect(words(season().querySelector("[data-part=rain-since]"))).toMatch(/^Rain since Oct 1/);
    expect(nameFrom(season())).toMatch(/Open the Rain tab at the season\.$/);
  });

  it("takes Wet hours and Rain since Oct 1 out of the mosaic and the now tile, so no count shows twice", () => {
    expect(panel.querySelector("[data-section=wet-hours]")).toBeNull();
    expect(panel.querySelectorAll("[data-section=rain]")).toHaveLength(1);
    expect(panel.querySelector("[data-section=first-screen] [data-part=wet-meter]")).toBeNull();
    expect(panel.querySelectorAll("[data-part=wet-meter]")).toHaveLength(1);
    const grid = words(panel.querySelector(".tile-grid")) ?? "";
    expect(grid.match(/Rain since Oct 1/g)).toHaveLength(1);
  });
});

describe("the rain tile, stopped (the raining golden, rain stopped 40 minutes ago)", () => {
  beforeEach(() => renderOverview(panel, stopped()));

  it("labels the tile with a hollow ring and Rain", () => {
    const kicker = now().querySelector("[data-part=kicker]")!;
    expect(words(kicker)).toBe("Rain");
    expect(kicker.querySelector("[data-part=stopped]")).not.toBeNull();
    expect(kicker.querySelector("[data-part=live]")).toBeNull();
  });

  it("tells the stop in one line of two runs, with no rate", () => {
    const finding = now().querySelector("[data-part=finding]")!;
    expect(words(finding)).toBe("Rain stopped 40 min ago · 0.30 in since 2:00 pm");
    expect([...finding.querySelectorAll(".rain-lead-run")].map(words)).toEqual([
      "Rain stopped 40 min ago ·",
      "0.30 in since 2:00 pm",
    ]);
    expect(now().querySelector("[data-part=value]")).toBeNull();
    expect(words(now())).not.toContain("in/hr");
  });

  it("brackets the dry stretch from last rain to now, the line's total in place", () => {
    expect(svgText()).toContain("dry 40 min");
    expect(svgText()).toContain("0.30 in total");
    expect(svgText().some((t) => t?.startsWith("last hour"))).toBe(false);
  });

  it("words a stop over an hour ago in whole hours, as the boards do", () => {
    renderOverview(panel, stopped(130));
    expect(words(now().querySelector("[data-part=finding]"))).toMatch(/^Rain stopped 2 h ago/);
    expect(svgText()).toContain("dry 2 h");
  });
});

describe("the rain tile's edges", () => {
  it("tells the rate alone when no event holds last rain: no line", () => {
    const d = doc();
    d.rain_now = { ...d.rain_now, event_start_ms: null, event_total_in: null };
    d.rain_now.last_3_hours.total_in = d.rain_now.last_3_hours.total_in.map(() => null);
    renderOverview(panel, parseDashboard(d));
    expect(words(now().querySelector("[data-part=finding]"))).toBe("in the last hour");
    expect(now().querySelector("path.rain-lead-line")).toBeNull();
  });

  it("draws no chart from a snapshot without the slots (version 4 before #661)", () => {
    const d = doc();
    delete d.rain_now.last_3_hours;
    renderOverview(panel, parseDashboard(d));
    expect(tile()).not.toBeNull();
    expect(now().querySelector("svg")).toBeNull();
  });

  it("refuses slots whose two series differ in length", () => {
    const d = doc();
    d.rain_now.last_3_hours.total_in.pop();
    expect(() => parseDashboard(d)).toThrow(SnapshotContractError);
  });

  it("says the water year's verdict and the usual by now when the snapshot has them", () => {
    const d = doc();
    d.rain_tab.usual_to_date = {
      usual_in: 7.9,
      years: 3,
      percent: 50,
      verdict: "drier",
      water_years: [2023, 2024, 2025],
    };
    renderOverview(panel, parseDashboard(d));
    const year = season().querySelector("[data-part=rain-since]")!;
    expect(words(year.querySelector("[data-part=verdict]"))).toBe("drier than usual");
    expect(words(year)).toContain("usual by now 7.90 in");
  });

  it("in dry mode drops the bars from the foot, and Dry spell stays in the mosaic", () => {
    const d = doc();
    d.wet_mode = { mode: "dry", decided_by: "weather" };
    d.tile_titles = { ...d.tile_titles, wet_or_dry: "Dry spell" };
    renderOverview(panel, parseDashboard(d));
    expect(season().querySelector("[data-part=wet-meter]")).toBeNull();
    expect(season().querySelector("[data-part=rain-since]")).not.toBeNull();
    expect(ids()).toContain("dry-spell");
    expect(ids().filter((id) => id === "rain")).toHaveLength(1);
  });
});

describe("later and dry: the mosaic of decision 24", () => {
  it("draws the feature and the wet-mode mosaic once rain stops leading", () => {
    const d = doc();
    d.rain_now = {
      ...d.rain_now,
      raining: false,
      rate_in_hr: null,
      event_start_ms: null,
      event_total_in: null,
      last_3_hours: null,
    };
    d.feature = { ...d.feature, kind: "quiet" };
    renderOverview(panel, parseDashboard(d));
    expect(tile()).toBeNull();
    expect(ids().slice(0, 5)).toEqual([
      "first-screen",
      "feature",
      "temperature",
      "wet-hours",
      "rain",
    ]);
  });

  it("draws the dry-mode mosaic on the seasoned golden", () => {
    renderOverview(panel, parseDashboard(golden("seasoned", "dashboard")));
    expect(tile()).toBeNull();
    expect(ids().slice(0, 4)).toEqual(["first-screen", "feature", "temperature", "dry-spell"]);
  });
});

describe("the tile's taps", () => {
  it("opens the rain zoom under the tile from the now part", () => {
    setLayout("four");
    history.replaceState(null, "", "/#overview/rain");
    renderOverview(panel, raining());
    expect(panel.querySelector('.zoom[data-zoom="rain"]')).not.toBeNull();
    expect(now().getAttribute("aria-expanded")).toBe("true");
  });

  it("makes both links at least 44px tall", () => {
    const css = readFileSync(resolve(import.meta.dirname, "../src/overview/rain-lead.css"), "utf8");
    for (const cls of ["now", "season"]) {
      const rule = new RegExp(`\\.rain-lead-${cls} \\{([^}]*)\\}`).exec(css)![1]!;
      expect(rule, cls).toContain("min-height: 44px");
    }
  });

  it("keeps the season route and brings the Rain tab's season part into view", () => {
    expect(isSeasonRoute("#rain/season")).toBe(true);
    expect(isSeasonRoute("#rain")).toBe(false);
    const rain = document.createElement("div");
    const part = document.createElement("p");
    part.dataset.part = "season";
    rain.append(part);
    const scroll = vi.fn();
    part.scrollIntoView = scroll;
    history.replaceState(null, "", "/#rain/season");
    followSeason(rain);
    expect(scroll).toHaveBeenCalledTimes(1);
    history.replaceState(null, "", "/#rain");
    followSeason(rain);
    expect(scroll).toHaveBeenCalledTimes(1);
  });
});

describe("the rain tile's marks", () => {
  const pairs = contrastPairs();
  const onTile = (fg: string) =>
    pairs.some((p) => p.fg.includes(fg) && p.bg === "tile" && p.min >= 3 && p.themes.length === 2);

  it("draws every mark a reader needs 3:1 on the tile, in both themes", () => {
    const css = readFileSync(resolve(import.meta.dirname, "../src/overview/rain-lead.css"), "utf8");
    const token = (selector: string, property: string) => {
      const rule = new RegExp(`${selector.replace(/\./g, "\\.")} \\{([^}]*)\\}`).exec(css)![1]!;
      return new RegExp(`${property}: var\\(--([a-z0-9-]+)\\)`).exec(rule)![1]!;
    };
    expect(onTile(token(".rain-lead-col", "fill"))).toBe(true);
    expect(onTile(token(".rain-lead-line", "stroke"))).toBe(true);
    expect(onTile(token(".rain-lead-axis", "stroke"))).toBe(true);
    expect(onTile(token(".rain-lead-bars .wet-meter-wet", "background"))).toBe(true);
  });
});
