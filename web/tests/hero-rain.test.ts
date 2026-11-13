// @vitest-environment happy-dom
/**
 * The hero's Rain column (#619), through the Overview on the goldens: rain
 * today (the live total) and yesterday (the Yesterday window), shown only
 * when either is above 0. From 760 px up it is a column of two bars on the
 * ribbon chart's baseline; below 760 px, one line under the temperature's
 * caption. Both are in the DOM; the layout's CSS shows one.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { renderOverview } from "../src/overview/index.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
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
const line = () => hero().querySelector<HTMLElement>(".now-line [data-part=rain-line]");
const columnBox = () => hero().querySelector<HTMLElement>(":scope > [data-part=rain-column]");
const texts = (cls: string) =>
  [...(columnBox()?.querySelectorAll(`text.${cls}`) ?? [])].map((t) => t.textContent);

describe("the hero Rain column", () => {
  it("shows rain today and yesterday when either is above 0 (raining)", () => {
    const snapshot = parsed("raining");
    renderOverview(panel, snapshot);
    expect(words(line())).toBe("Rain today 0.30 in · yesterday 0.24 in");
    expect(words(columnBox()?.querySelector("text.hero-rain-head"))).toBe("Rain");
    expect(texts("hero-rain-value")).toEqual(["0.30", "0.24"]);
    expect(texts("hero-rain-name")).toEqual(["Today", "Yesterday"]);
    expect(hero().dataset.rain).toBe("true");
  });

  it("is removed when neither day had rain, and the chart takes the width back", () => {
    for (const scenario of ["seasoned", "dry"] as const) {
      panel.replaceChildren();
      renderOverview(panel, parsed(scenario));
      expect(line()).toBeNull();
      expect(columnBox()).toBeNull();
      expect(hero().dataset.rain).toBeUndefined();
    }
  });

  it("shows when only yesterday had rain", () => {
    const snapshot = parsed("seasoned");
    snapshot.rain_tab.windows.find((w) => w.key === "yesterday")!.total_in = 0.12;
    renderOverview(panel, snapshot);
    expect(words(line())).toBe("Rain today 0.00 in · yesterday 0.12 in");
    expect(texts("hero-rain-value")).toEqual(["0.00", "0.12"]);
  });

  it("keeps Today as the live rain today during a rain event", () => {
    const snapshot = parsed("raining");
    snapshot.current.rain_today_in = 0.41;
    renderOverview(panel, snapshot);
    expect(texts("hero-rain-value")[0]).toBe("0.41");
  });

  it("stands its bars on the ribbon chart's x-axis line, names on its x-label line", () => {
    renderOverview(panel, parsed("raining"));
    const chart = hero().querySelector(".today-chart svg")!;
    const floor = Number(chart.querySelector(".today-rain-floor")!.getAttribute("y1"));
    const labelY = chart.querySelector(".today-x")!.getAttribute("y");
    const svg = columnBox()!.querySelector("svg")!;
    expect(svg.getAttribute("height")).toBe(chart.getAttribute("height"));
    for (const bar of svg.querySelectorAll("rect.hero-rain-bar")) {
      expect(Number(bar.getAttribute("y")) + Number(bar.getAttribute("height"))).toBe(floor);
    }
    expect(Number(svg.querySelector("line.hero-rain-floor")!.getAttribute("y1"))).toBe(floor);
    for (const name of svg.querySelectorAll("text.hero-rain-name")) {
      expect(name.getAttribute("y")).toBe(labelY);
    }
  });
});
