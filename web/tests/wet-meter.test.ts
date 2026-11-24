// @vitest-environment happy-dom
/**
 * The wet-hour bars (#657, spec #647 "Meter bars"; the C boards' words,
 * #661), through the Overview on the goldens: two bars, 7 days and 30 days,
 * each reading wet hours of countable hours ("19 h of 168"), never a
 * percent, under the header "Wet hours · each bar is every hour; dark is
 * wet". The missing hours are hatched at the bar's end, and the key under
 * the bars reads "hatched: missing, not dry". Every mark draws in a token
 * whose 3:1 pair on the now surface the Contrast table holds, in both
 * themes. The bars sit at the now tile's foot only for a snapshot without a
 * mode; with one, the Wet hours tile or the rain tile's foot carries them,
 * and dry mode (#658) hides them. A snapshot without `wet_hours` (N−1)
 * draws none.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import { renderOverview } from "../src/overview/index.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { nameFrom } from "./accessible-name.js";
import { contrastPairs } from "./contrast.js";
import { golden } from "./golden.js";
import type { Scenario } from "./golden.js";

const parsed = (scenario: Scenario): DashboardSnapshot =>
  parseDashboard(structuredClone(golden(scenario, "dashboard")));

/**
 * A golden without a mode (N−1): the one case the bars sit at the now
 * tile's foot. With a mode, the Wet hours tile carries them in wet mode and
 * the rain tile's foot while rain leads (#661, Call 4 on #647).
 */
const noMode = (scenario: Scenario): DashboardSnapshot => {
  const doc = structuredClone(golden(scenario, "dashboard")) as Record<string, unknown>;
  delete doc.wet_mode;
  delete doc.tile_titles;
  return parseDashboard(doc);
};

let panel: HTMLElement;

beforeEach(() => {
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const words = (node: Element | null | undefined) => node?.textContent?.replace(/\s+/g, " ").trim();
const hero = () => panel.querySelector<HTMLElement>("[data-section=first-screen]")!;
const meter = () => hero().querySelector<HTMLAnchorElement>("[data-part=wet-meter]");
const all = (part: string) => [...(meter()?.querySelectorAll(`[data-part=${part}]`) ?? [])];

describe("the wet-hour meter", () => {
  it("reads the board's words: wet hours of countable hours for 7 and 30 days, never a percent", () => {
    renderOverview(panel, noMode("seasoned"));
    expect(words(meter()!.querySelector(".wet-meter-title"))).toBe(
      "Wet hours · each bar is every hour; dark is wet",
    );
    expect([...meter()!.querySelectorAll(".wet-meter-name")].map(words)).toEqual([
      "7 days",
      "30 days",
    ]);
    expect(all("count").map(words)).toEqual(["2 h of 168", "6 h of 720"]);
    expect(nameFrom(meter()!)).toBe(
      "Wet hours · each bar is every hour; dark is wet 7 days 2 h of 168 30 days 6 h of 720 Open the rain view",
    );
    expect(words(meter())).not.toContain("%");
  });

  it("is titled by the snapshot's tile title, the literal only without one (#694)", () => {
    const doc = structuredClone(golden("raining", "dashboard")) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
    doc.tile_titles = { ...doc.tile_titles, wet_or_dry: "Wet time" };
    renderOverview(panel, parseDashboard(doc));
    const title = () => panel.querySelector("[data-section=rain] .wet-meter-title");
    expect(words(title())).toBe("Wet time · each bar is every hour; dark is wet");
    delete doc.tile_titles;
    renderOverview(panel, parseDashboard(doc));
    expect(words(title())).toBe("Wet hours · each bar is every hour; dark is wet");
  });

  it("is one link to the Rain tab, at the now tile's foot, without a mode", () => {
    renderOverview(panel, noMode("seasoned"));
    expect(meter()!.getAttribute("href")).toBe("#rain");
    expect(hero().lastElementChild).toBe(meter());
  });

  it("carries no hatch key when every hour counts", () => {
    renderOverview(panel, noMode("seasoned"));
    expect(all("hatch-key")).toHaveLength(0);
    expect(meter()!.querySelectorAll(".wet-meter-missing")).toHaveLength(0);
  });

  it("hatches the missing hours at each bar's end, with the key in the board's words", () => {
    renderOverview(panel, noMode("dry"));
    expect(all("count").map(words)).toEqual(["8 h of 43", "8 h of 43"]);
    for (const bar of all("bar")) {
      const hatch = bar.lastElementChild!;
      expect(hatch.classList.contains("wet-meter-missing")).toBe(true);
      expect(hatch.classList.contains("gap-hatch")).toBe(true);
    }
    const [key] = all("hatch-key");
    expect(words(key)).toBe("hatched: missing, not dry");
    expect(key!.firstElementChild!.classList.contains("hatch-swatch")).toBe(true);
  });

  it("draws each part as its share of every hour in the window", () => {
    renderOverview(panel, noMode("dry"));
    const week = all("bar")[0]!;
    const share = (cls: string) =>
      week.querySelector<HTMLElement>(`.${cls}`)!.style.getPropertyValue("--share");
    expect(share("wet-meter-wet")).toBe(`${(100 * 8) / 168}%`);
    expect(share("wet-meter-missing")).toBe(`${(100 * 125) / 168}%`);
  });

  it("draws no wet fill when no hour was wet", () => {
    const snapshot = noMode("seasoned");
    snapshot.wet_hours!.last_7_days.wet_hours = 0;
    renderOverview(panel, snapshot);
    expect(all("bar")[0]!.querySelector(".wet-meter-wet")).toBeNull();
    expect(words(all("count")[0])).toBe("0 h of 168");
  });

  it("is hidden in dry mode, where it would read 0 h for months (seasoned)", () => {
    const snapshot = parsed("seasoned");
    expect(snapshot.wet_mode?.mode).toBe("dry");
    renderOverview(panel, snapshot);
    expect(meter()).toBeNull();
    expect(hero().querySelector("[data-part=temperature]")).not.toBeNull();
    const sources = words(panel.querySelector("footer.ov-sources"));
    expect(sources).not.toContain("wet-hour bars");
  });

  it("leaves the now tile's foot in wet mode, where the Wet hours tile carries the bars (dry)", () => {
    renderOverview(panel, parsed("dry"));
    expect(meter()).toBeNull();
    expect(panel.querySelector("[data-section=wet-hours]")).not.toBeNull();
  });

  it("leaves the now tile's foot while rain leads, for the rain tile's (raining)", () => {
    renderOverview(panel, parsed("raining"));
    expect(meter()).toBeNull();
    const foot = panel.querySelector("[data-section=rain] [data-part=season]");
    expect(foot?.querySelectorAll("[data-part=bar]")).toHaveLength(2);
  });

  it("is drawn from a version 4 snapshot without the mode, as before it", () => {
    renderOverview(panel, noMode("seasoned"));
    expect(meter()).not.toBeNull();
  });

  it("is not drawn from a snapshot without wet hours", () => {
    for (const absent of [undefined, null]) {
      panel.replaceChildren();
      const doc = structuredClone(golden("dry", "dashboard")) as Record<string, unknown>;
      delete doc.wet_mode;
      if (absent === undefined) delete doc.wet_hours;
      else doc.wet_hours = null;
      renderOverview(panel, parseDashboard(doc));
      expect(meter()).toBeNull();
      expect(hero().querySelector("[data-part=temperature]")).not.toBeNull();
    }
  });
});

describe("the wet_hours contract", () => {
  it("refuses a window without its counts", () => {
    const doc = structuredClone(golden("raining", "dashboard")) as {
      wet_hours: { last_7_days: Record<string, unknown> };
    };
    delete doc.wet_hours.last_7_days.missing_days;
    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });

  it("reads a strip day with no countable hour as null", () => {
    const snapshot = parsed("raining");
    const gap = snapshot.wet_hours!.days.find((d) => d.countable_hours === null);
    expect(gap?.wet_hours).toBeNull();
    expect(snapshot.wet_hours!.days).toHaveLength(30);
  });
});

describe("the meter's marks", () => {
  const CSS = readFileSync(
    resolve(import.meta.dirname, "../src/overview/wet-meter.css"),
    "utf8",
  ).replace(/\/\*[\s\S]*?\*\//g, "");
  const pairs = contrastPairs();
  const onNow = (fg: string) =>
    pairs.some(
      (p) => p.fg.includes(fg) && p.bg === "now-surface" && p.min >= 3 && p.themes.length === 2,
    );
  const token = (selector: string, property: string) => {
    const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const rule = new RegExp(`(?:^|\\})\\s*${escaped}\\s*\\{([^}]*)\\}`).exec(CSS);
    expect(rule, selector).not.toBeNull();
    return /var\(--([a-z0-9-]+)\)/.exec(
      new RegExp(`(?:^|;|\\s)${property}:([^;]*);`).exec(rule![1]!)![1]!,
    )![1]!;
  };

  it("draw the wet fill and the track's edge 3:1 on the now surface, in both themes", () => {
    expect(onNow(token(".wet-meter-wet", "background"))).toBe(true);
    expect(onNow(token(".wet-meter-bar", "border"))).toBe(true);
  });

  it("rules the bars off with a `rule` hairline, as every divider is (#703)", () => {
    expect(token(".wet-meter", "border-top")).toBe("rule");
  });

  it("hatch in ink-3 on the now tile, which holds 3:1 there in both themes", () => {
    const tiles = readFileSync(resolve(import.meta.dirname, "../src/styles/tiles.css"), "utf8");
    expect(tiles).toMatch(/\[data-surface="now"\][^}]*--hatch: var\(--ink-3\)/);
    expect(onNow("ink-3")).toBe(true);
  });
});
