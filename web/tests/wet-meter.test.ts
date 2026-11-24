// @vitest-environment happy-dom
/**
 * The wet-hour meter (#657, spec #647 "Meter bars"), through the Overview on
 * the goldens: two bars at the now tile's foot, 7 days and 30 days, each
 * reading wet hours of countable hours ("19 h of 168"), never a percent. The
 * missing hours are hatched at the bar's end and named under it, "N h
 * missing, not dry", with the missing days ("5 d missing"). Every mark draws
 * in a token whose 3:1 pair on the now surface the Contrast table holds, in
 * both themes. A snapshot without `wet_hours` (N−1) draws no meter.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import { renderOverview } from "../src/overview/index.js";
import { missingWords } from "../src/overview/wet-meter.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { nameFrom } from "./accessible-name.js";
import { contrastPairs } from "./contrast.js";
import { golden } from "./golden.js";
import type { Scenario } from "./golden.js";

const parsed = (scenario: Scenario): DashboardSnapshot =>
  parseDashboard(structuredClone(golden(scenario, "dashboard")));

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
  it("reads wet hours of countable hours for 7 and 30 days, never a percent", () => {
    renderOverview(panel, parsed("seasoned"));
    expect(all("count").map(words)).toEqual(["2 h of 168", "6 h of 720"]);
    expect(nameFrom(meter()!)).toBe(
      "Wet hours 7 days 2 h of 168 30 days 6 h of 720 Open the rain view",
    );
    expect(words(meter())).not.toContain("%");
  });

  it("is one link to the Rain tab, at the now tile's foot", () => {
    renderOverview(panel, parsed("seasoned"));
    expect(meter()!.getAttribute("href")).toBe("#rain");
    expect(hero().lastElementChild).toBe(meter());
  });

  it("names nothing missing when every hour counts", () => {
    renderOverview(panel, parsed("seasoned"));
    expect(all("missing")).toHaveLength(0);
    expect(meter()!.querySelectorAll(".wet-meter-missing")).toHaveLength(0);
  });

  it("hatches the missing hours at each bar's end and names them, not dry", () => {
    renderOverview(panel, parsed("raining"));
    expect(all("count").map(words)).toEqual(["12 h of 111", "12 h of 111"]);
    expect(all("missing").map(words)).toEqual([
      "57 h missing, not dry · 3 d missing",
      "609 h missing, not dry · 26 d missing",
    ]);
    for (const bar of all("bar")) {
      const hatch = bar.lastElementChild!;
      expect(hatch.classList.contains("wet-meter-missing")).toBe(true);
      expect(hatch.classList.contains("gap-hatch")).toBe(true);
    }
    // Each note carries the hatch as a swatch beside its words.
    for (const note of all("missing")) {
      expect(note.firstElementChild!.classList.contains("hatch-swatch")).toBe(true);
    }
  });

  it("draws each part as its share of every hour in the window", () => {
    const snapshot = parsed("raining");
    renderOverview(panel, snapshot);
    const week = all("bar")[0]!;
    const share = (cls: string) =>
      week.querySelector<HTMLElement>(`.${cls}`)!.style.getPropertyValue("--share");
    expect(share("wet-meter-wet")).toBe(`${(100 * 12) / 168}%`);
    expect(share("wet-meter-missing")).toBe(`${(100 * 57) / 168}%`);
  });

  it("draws no wet fill when no hour was wet", () => {
    const snapshot = parsed("seasoned");
    snapshot.wet_hours!.last_7_days.wet_hours = 0;
    renderOverview(panel, snapshot);
    expect(all("bar")[0]!.querySelector(".wet-meter-wet")).toBeNull();
    expect(words(all("count")[0])).toBe("0 h of 168");
  });

  it("is not drawn from a snapshot without wet hours", () => {
    for (const absent of [undefined, null]) {
      panel.replaceChildren();
      const doc = structuredClone(golden("raining", "dashboard")) as Record<string, unknown>;
      if (absent === undefined) delete doc.wet_hours;
      else doc.wet_hours = null;
      renderOverview(panel, parseDashboard(doc));
      expect(meter()).toBeNull();
      expect(hero().querySelector("[data-part=temperature]")).not.toBeNull();
    }
  });
});

describe("the missing words", () => {
  const window = { days: 7, hours: 168, wet_hours: 19, countable_hours: 150 };

  it("name the hours, then the whole days, missing", () => {
    expect(missingWords({ ...window, missing_hours: 18, missing_days: 0 })).toEqual([
      "18 h missing, not dry",
    ]);
    expect(missingWords({ ...window, missing_hours: 120, missing_days: 5 })).toEqual([
      "120 h missing, not dry",
      "5 d missing",
    ]);
    expect(missingWords({ ...window, missing_hours: 0, missing_days: 0 })).toEqual([]);
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
    expect(onNow(token(".wet-meter", "border-top"))).toBe(true);
  });

  it("hatch in ink-3 on the now tile, which holds 3:1 there in both themes", () => {
    const tiles = readFileSync(resolve(import.meta.dirname, "../src/styles/tiles.css"), "utf8");
    expect(tiles).toMatch(/\[data-surface="now"\][^}]*--hatch: var\(--ink-3\)/);
    expect(onNow("ink-3")).toBe(true);
  });
});
