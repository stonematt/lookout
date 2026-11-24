// @vitest-environment happy-dom
/**
 * The Rain Events page's headline and scatter (#666, board EventsBC),
 * rendered from the golden snapshots: the headline counts Python's events,
 * the scatter draws Python's points from a zero baseline with its size key
 * in place, and no two dots overlap at phone width, however crowded a row.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EventsDeps } from "../src/events/tab.js";
import { mountEvents } from "../src/events/tab.js";
import type { ScatterInput, ScatterLayout } from "../src/events/scatter.js";
import { RADIUS, layoutScatter, scatterInput } from "../src/events/scatter.js";
import type { DashboardSnapshot, EventCategory, ScatterYear } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { golden } from "./golden.js";

/** The chart's width in a 390px phone's tile (16px gutters, 16px padding). */
const PHONE_CHART = 326;

let panel: HTMLElement;
let deps: EventsDeps;

function snapshot(scenario: Scenario): DashboardSnapshot {
  return parseDashboard(structuredClone(golden(scenario, "dashboard")));
}

function mount(snap: DashboardSnapshot): void {
  mountEvents(panel, snap, deps);
}

const text = (selector: string) =>
  panel.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim();

beforeEach(() => {
  localStorage.clear();
  history.replaceState(null, "", "/#events");
  document.body.innerHTML = "<section id=panel></section>";
  panel = document.getElementById("panel")!;
  deps = {
    fetchCurve: vi.fn(async () => {
      throw new Error("no curve in these tests");
    }),
    draw: vi.fn(async () => {}),
    nowMs: () => Date.UTC(2026, 0, 9, 0, 30),
  };
});

describe("the headline", () => {
  it.each([
    ["raining", "3 events this water year · 3 in the last 7 days"],
    ["seasoned", "2 events this water year · 1 in the last 7 days"],
    ["dry", "1 event this water year · 1 in the last 7 days"],
  ] as const)("counts the %s station's events as Python counted them", (scenario, words) => {
    mount(snapshot(scenario));

    expect(text("[data-section=catalog] h2.ev-headline")).toBe(words);
  });

  it("opens the page: the headline, then the scatter, then the pick's finding", () => {
    mount(snapshot("seasoned"));

    const parts = [...panel.querySelectorAll("[data-section=catalog] [data-part]")].map(
      (el) => (el as HTMLElement).dataset.part,
    );
    expect(parts.indexOf("finding")).toBeLessThan(parts.indexOf("scatter"));
    expect(parts.indexOf("scatter")).toBeLessThan(parts.indexOf("pick"));
    expect(panel.querySelector("[data-part=standfirst]")).toBeNull();
  });

  it("opens as before on a snapshot written before the counts", () => {
    const old = snapshot("raining");
    delete old.rain_events.counts;
    delete old.rain_events.scatter;
    mount(old);

    expect(panel.querySelector(".ev-headline")).toBeNull();
    expect(panel.querySelector(".ev-scatter")).toBeNull();
    expect(panel.querySelector("h2.ev-title")).not.toBeNull();
    expect(panel.querySelector("[data-part=standfirst]")).not.toBeNull();
  });
});

describe("the scatter", () => {
  it("draws one row per water year, newest on top, a dot per completed event", () => {
    mount(snapshot("seasoned"));

    const years = [...panel.querySelectorAll(".ev-sc-year")].map((t) => t.textContent);
    expect(years).toEqual(["2026", "2025", "2024"]);
    expect(panel.querySelectorAll(".ev-sc-dot")).toHaveLength(5);
    expect(text(".ev-sc-years")).toBe("3 water years");
    expect(panel.querySelector(".ev-sc-svg")!.getAttribute("aria-label")).toContain(
      "2026: 2 events, the largest",
    );
  });

  it("sizes each dot by Python's category and keys the sizes in place", () => {
    const snap = snapshot("seasoned");
    mount(snap);

    const dots = [...panel.querySelectorAll<SVGCircleElement>(".ev-sc-dot")];
    const categories = snap.rain_events.scatter!.flatMap((row) => row.category);
    expect(dots.map((d) => Number(d.getAttribute("r")))).toEqual(categories.map((c) => RADIUS[c]));
    expect(dots.every((d, i) => d.classList.contains(`ev-sc-${categories[i]}`))).toBe(true);
    const key = [...panel.querySelectorAll(".ev-sc-key li")].map((li) => li.textContent);
    expect(key).toEqual(["sprinkle", "shower", "rain", "storm"]);
  });

  it("starts its axis at zero, under every dot", () => {
    const layout = layoutScatter(scatterInput(snapshot("seasoned").rain_events)!, PHONE_CHART);

    expect(layout.ticks[0]).toEqual({ x: layout.x0, value: 0 });
    for (const dot of layout.rows.flatMap((r) => r.dots)) {
      expect(dot.cx).toBeGreaterThanOrEqual(layout.x0);
      expect(dot.cy + dot.r).toBeLessThan(layout.baseline);
    }
    mount(snapshot("seasoned"));
    expect(panel.querySelector(".ev-sc-tick")!.textContent).toBe("0");
  });

  it("rings the live event in now under this year's dots, not as a point", () => {
    const snap = snapshot("raining");
    mount(snap);

    expect(panel.querySelectorAll(".ev-sc-dot")).toHaveLength(2);
    expect(panel.querySelectorAll(".ev-sc-now")).toHaveLength(1);
    expect(text(".ev-sc-now-label")).toBe("now 0.30 in");
    const layout = layoutScatter(scatterInput(snap.rain_events)!, PHONE_CHART);
    const lowest = Math.max(...layout.rows[0]!.dots.map((d) => d.cy + d.r));
    expect(layout.now!.cy - 5).toBeGreaterThan(lowest);
  });

  it("draws the usual event as the catalog's median", () => {
    const snap = snapshot("raining");
    mount(snap);

    expect(panel.querySelector(".ev-sc-usual")).not.toBeNull();
    expect(text(".ev-sc-label")).toBe("usual event 0.42 in");
  });
});

/** A deterministic stand-in for a wet station: many small events per year. */
function crowded(): ScatterInput {
  let seed = 7;
  const random = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  const rows: ScatterYear[] = [2027, 2026, 2025, 2024].map((year) => {
    const n = year === 2027 ? 20 : 90;
    const total_in = Array.from({ length: n }, () => Math.round(random() ** 3 * 250) / 100);
    const category = total_in.map((t): EventCategory =>
      t < 0.05 ? "sprinkle" : t >= 2 ? "storm" : t >= 0.5 ? "rain" : "shower",
    );
    return {
      water_year: year,
      start_ms: total_in.map((_, i) => Date.UTC(year - 1, 10, 1) + i * 3_600_000),
      total_in,
      category,
    };
  });
  return { rows, usualIn: 0.2, liveIn: 0.31 };
}

/** Every pair of dots (and the now ring) that overlap in a layout. */
function overlaps(layout: ScatterLayout): string[] {
  const dots = layout.rows.flatMap((r) => r.dots.map((d) => ({ ...d, year: r.waterYear })));
  if (layout.now) dots.push({ ...dots[0]!, cx: layout.now.cx, cy: layout.now.cy, r: 5 });
  const found: string[] = [];
  for (let i = 0; i < dots.length; i++) {
    for (let j = i + 1; j < dots.length; j++) {
      const a = dots[i]!;
      const b = dots[j]!;
      if (Math.hypot(a.cx - b.cx, a.cy - b.cy) < a.r + b.r - 1e-6) found.push(`${i}/${j}`);
    }
  }
  return found;
}

describe("the axis", () => {
  it("runs from zero on a square root, its round ticks apart, the last with its unit", () => {
    const layout = layoutScatter(crowded(), PHONE_CHART);
    const xs = layout.ticks.map((t) => t.x);

    expect(layout.ticks[0]!.value).toBe(0);
    expect(xs.slice(1).every((x, i) => x - xs[i]! >= 40)).toBe(true);
    // A dot's distance from zero goes as the square root of its total.
    const dots = layout.rows.flatMap((r) => r.dots).filter((d) => d.totalIn > 0);
    const [a, b] = [dots[0]!, dots.find((d) => d.totalIn !== dots[0]!.totalIn)!];
    expect(((a.cx - layout.x0) / (b.cx - layout.x0)) ** 2).toBeCloseTo(a.totalIn / b.totalIn, 6);
    mountEvents(panel, snapshot("seasoned"), deps);
    expect(panel.querySelector(".ev-sc-tick:last-of-type")!.textContent).toMatch(/ in$/);
  });
});

describe("no overlapping dots", () => {
  it.each([PHONE_CHART, 288, 1100])("keeps every dot apart at %ipx", (width) => {
    const layout = layoutScatter(crowded(), width);

    expect(layout.rows.flatMap((r) => r.dots)).toHaveLength(290);
    expect(overlaps(layout)).toEqual([]);
    for (const dot of layout.rows.flatMap((r) => r.dots)) {
      expect(dot.cx - dot.r).toBeGreaterThanOrEqual(0);
      expect(dot.cx + dot.r).toBeLessThanOrEqual(width);
    }
  });

  it("keeps each row's swarm clear of the next row's", () => {
    const layout = layoutScatter(crowded(), PHONE_CHART);

    for (let i = 1; i < layout.rows.length; i++) {
      const above = Math.max(...layout.rows[i - 1]!.dots.map((d) => d.cy + d.r));
      const below = Math.min(...layout.rows[i]!.dots.map((d) => d.cy - d.r));
      expect(below).toBeGreaterThan(above);
    }
  });
});
