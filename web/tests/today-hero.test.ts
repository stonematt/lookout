// @vitest-environment happy-dom
/**
 * The usual band hero (#468): the fixed top shows today's temperature against
 * the usual band for this date, drawn from the goldens through the Overview's
 * public entry point. The band, its basis and today's slots are the
 * snapshot's; the hero only draws them.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { localDay } from "../src/format.js";
import { renderOverview } from "../src/overview/index.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { golden, SCENARIOS } from "./golden.js";

const parsed = (scenario: (typeof SCENARIOS)[number]) =>
  parseDashboard(golden(scenario, "dashboard"));
const raining = parsed("raining");
const dry = parsed("dry");
const seasoned = parsed("seasoned");

/** A golden dashboard document with `usual_band` taken out (N−1, ADR-0008). */
function withoutBand(scenario: (typeof SCENARIOS)[number]): Record<string, unknown> {
  const doc = structuredClone(golden(scenario, "dashboard")) as Record<string, unknown>;
  delete (doc.temperature_context as Record<string, unknown>).usual_band;
  return doc;
}

let panel: HTMLElement;

beforeEach(() => {
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const words = (node: Element | null | undefined) => node?.textContent?.replace(/\s+/g, " ").trim();
const hero = () => panel.querySelector<HTMLElement>("[data-section=first-screen]")!;
const one = (selector: string) => hero().querySelector<SVGElement>(selector);
const all = (selector: string) => [...hero().querySelectorAll<SVGElement>(selector)];
/** The x of each point of a polyline. */
const xs = (line: SVGElement | null) =>
  (line?.getAttribute("points") ?? "")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((p) => Number(p.split(",")[0]));

/** Today's slots of the ribbon: the ones in the "today" period's day. */
function todaySlots(snapshot: DashboardSnapshot): { rain: (number | null)[]; temps: number[] } {
  const ribbon = snapshot.condition_ribbon!;
  const today = snapshot.temperature_context.periods[0]!.status.first_day;
  const slotMs = ribbon.slot_minutes * 60_000;
  const idx = ribbon.temperature_f
    .map((_, i) => i)
    .filter((i) => localDay(ribbon.start_ms + i * slotMs) === today);
  return {
    rain: idx.map((i) => ribbon.rain_in[i]!),
    temps: idx.map((i) => ribbon.temperature_f[i]).filter((t): t is number => t !== null),
  };
}

describe("the usual band hero", () => {
  it.each(SCENARIOS)("renders from the %s golden", (scenario) => {
    renderOverview(panel, parsed(scenario));
    const chart = hero().querySelector("[data-part=today-chart] svg[role=img]");
    expect(chart).not.toBeNull();
    expect(chart!.getAttribute("aria-label")).toMatch(/^Today's temperature/);
    expect(all(".today-median").length).toBeGreaterThan(0);
  });

  it("replaces the condition sentence, the Right-now strip and the ribbon", () => {
    renderOverview(panel, raining);
    const parts = [...hero().querySelectorAll("[data-part]")].map((el) =>
      el.getAttribute("data-part"),
    );
    expect(parts).toEqual(["kicker", "finding", "temperature", "delta", "today-chart"]);
    // No folded source either: its line is in the Overview's footer.
    expect(hero().querySelector("details")).toBeNull();
    expect(panel.querySelector("[data-part=ribbon], .ribbon-band, .right-now")).toBeNull();
    expect(panel.querySelector("[data-part=condition]")).toBeNull();
  });

  it("draws the middle half as a band and the median as a dashed line, hour by hour", () => {
    renderOverview(panel, seasoned);
    const band = one(".today-band");
    expect(band?.tagName).toBe("path");
    // one median point per hour, plus the two ends held to midnight
    expect(xs(one(".today-median"))).toHaveLength(24 + 2);
  });

  it("runs the band to midnight, past now", () => {
    renderOverview(panel, seasoned);
    const median = xs(one(".today-median"));
    const plotRight =
      Number(one(".today-plot")!.getAttribute("x")) +
      Number(one(".today-plot")!.getAttribute("width"));
    const now = Number(one(".today-dot")!.getAttribute("cx"));
    expect(median.at(-1)).toBeCloseTo(plotRight, 1);
    expect(median.at(-1)!).toBeGreaterThan(now);
  });

  it("ends today's line at the orange now dot, labelled with the live reading", () => {
    renderOverview(panel, raining);
    const line = xs(one(".today-line"));
    const dot = Number(one(".today-dot")!.getAttribute("cx"));
    const lastPoint = Math.max(...line, ...xs(one(".today-join")));
    expect(lastPoint).toBeCloseTo(dot, 1);
    expect(words(one(".today-now"))).toBe("now 59.3°");
    // every archived slot of today is a point on the line, and none of yesterday
    expect(line).toHaveLength(todaySlots(raining).temps.length - 1);
  });

  it("joins the live reading over the unarchived slots with a dotted line", () => {
    renderOverview(panel, raining);
    expect(xs(one(".today-join"))).toHaveLength(2);
  });

  it("draws today's rain as quarter-hour bars along the foot, never over the band", () => {
    renderOverview(panel, raining);
    const wet = todaySlots(raining).rain.filter((r) => r !== null && r > 0).length;
    const bars = all(".today-rain rect[data-rain]");
    expect(wet).toBeGreaterThan(0);
    expect(bars).toHaveLength(wet);
    const plot = one(".today-plot")!;
    const plotBottom = Number(plot.getAttribute("y")) + Number(plot.getAttribute("height"));
    for (const bar of bars)
      expect(Number(bar.getAttribute("y"))).toBeGreaterThanOrEqual(plotBottom);
    // a heavier step stands taller
    const height = (step: string) =>
      Math.max(
        0,
        ...all(`.today-rain rect[data-rain="${step}"]`).map((r) =>
          Number(r.getAttribute("height")),
        ),
      );
    const steps = [...new Set(bars.map((b) => b.dataset.rain!))].sort();
    for (let i = 1; i < steps.length; i++) {
      expect(height(steps[i]!)).toBeGreaterThan(height(steps[i - 1]!));
    }
  });

  it("draws no rain bar on a dry day", () => {
    renderOverview(panel, seasoned);
    expect(all(".today-rain rect[data-rain]")).toEqual([]);
  });

  it("marks the times of day under the plot", () => {
    renderOverview(panel, seasoned);
    const ticks = all(".today-x").map(words);
    expect(ticks.slice(0, 4)).toEqual(["midnight", "6 am", "noon", "6 pm"]);
  });

  it("draws no line and no dot for a today with no reading yet", () => {
    // Built at midnight from yesterday's 23:55 reading.
    renderOverview(panel, dry);
    expect(one(".today-line")).toBeNull();
    expect(one(".today-dot")).toBeNull();
    expect(all(".today-median").length).toBe(1);
  });
});

describe("the band's basis label, from the snapshot", () => {
  const label = () => words(one(".today-basis"));

  it.each(SCENARIOS)("sits under the plot and the times of day on %s, clear of every line", (s) => {
    renderOverview(panel, parsed(s));
    const basis = one(".today-basis");
    expect(basis).not.toBeNull();
    const baseline = Number(basis!.getAttribute("y"));
    const plot = one(".today-plot")!;
    const plotFoot = Number(plot.getAttribute("y")) + Number(plot.getAttribute("height"));
    const floor = Number(one(".today-rain-floor")!.getAttribute("y1"));
    expect(baseline).toBeGreaterThan(plotFoot);
    expect(baseline).toBeGreaterThan(floor);
    for (const x of all(".today-x")) {
      expect(baseline).toBeGreaterThan(Number(x.getAttribute("y")));
    }
    const svgHeight = Number(basis!.closest("svg")!.getAttribute("height"));
    expect(svgHeight).toBeGreaterThan(baseline);
  });

  it("names the earlier years for an earlier-years band", () => {
    renderOverview(panel, seasoned);
    expect(label()).toBe("usual, this week in 2025 and 2024");
  });

  it("names the station's own recent days for a recent-days band", () => {
    const recent = structuredClone(seasoned);
    recent.temperature_context.usual_band = {
      ...recent.temperature_context.usual_band!,
      basis: "recent_days",
      years: [],
      days: 14,
    };
    renderOverview(panel, recent);
    expect(label()).toBe("usual, your last 14 days");
    expect(one(".today-band")).not.toBeNull();
  });

  it("draws only the median, and says so, for a median-only band", () => {
    renderOverview(panel, raining);
    expect(label()).toBe("median, your last 4 days");
    expect(one(".today-band")).toBeNull();
    expect(one(".today-median")).not.toBeNull();
  });

  it("speaks of one day as one day", () => {
    renderOverview(panel, dry);
    expect(label()).toBe("median, your last day");
  });

  it("describes the basis in the chart's accessible name", () => {
    renderOverview(panel, seasoned);
    const name = one("svg[role=img]")!.getAttribute("aria-label")!;
    expect(name).toContain("within a week of this date in 2025 and 2024");
    renderOverview(panel, raining);
    expect(one("svg[role=img]")!.getAttribute("aria-label")).toContain(
      "the median of your last 4 days",
    );
  });
});

describe("the delta line on usual", () => {
  const line = () => words(hero().querySelector("[data-part=delta]"));

  /** A golden dashboard document with `now_delta` set (or deleted when undefined). */
  function withDelta(delta: unknown): Record<string, unknown> {
    const doc = structuredClone(golden("seasoned", "dashboard")) as Record<string, unknown>;
    const band = (doc.temperature_context as { usual_band: Record<string, unknown> }).usual_band;
    if (delta === undefined) delete band.now_delta;
    else band.now_delta = delta;
    return doc;
  }

  it("says how far the reading is above the usual for its hour (seasoned)", () => {
    renderOverview(panel, seasoned);
    expect(line()).toBe("+12° on usual for 3 pm");
  });

  it("says how far below, with a real minus sign (dry)", () => {
    renderOverview(panel, dry);
    expect(line()).toBe("−1° on usual for 11 pm");
  });

  it("says at usual for a delta of 0 (raining)", () => {
    renderOverview(panel, raining);
    expect(line()).toBe("at usual for 4 pm");
  });

  it("writes a below-zero reading and delta with the true minus, in the tile and its words", () => {
    const doc = withDelta({ hour: 6, delta_f: -9, basis: "earlier_years" });
    (doc.current as Record<string, unknown>).temperature_f = -4.26;
    renderOverview(panel, parseDashboard(doc));
    expect(words(hero().querySelector("[data-part=temperature]"))).toBe("−4.3°F");
    expect(line()).toBe("−9° on usual for 6 am");
    expect(hero().textContent).not.toMatch(/-\d/);
  });

  it("stands under the temperature, read once with it (no tile name repeats it)", () => {
    renderOverview(panel, seasoned);
    const parts = [...hero().querySelectorAll("[data-part]")].map((el) =>
      el.getAttribute("data-part"),
    );
    expect(parts.indexOf("delta")).toBe(parts.indexOf("temperature") + 1);
    expect(line()).toBe("+12° on usual for 3 pm");
    expect(hero().hasAttribute("aria-label")).toBe(false);
  });

  it("prints no line and no element for a null delta", () => {
    renderOverview(panel, parseDashboard(withDelta(null)));
    expect(hero().querySelector("[data-part=delta]")).toBeNull();
    expect(hero().textContent).not.toContain("on usual");
  });

  it("prints no line for an absent delta (N−1), the rest of the hero unchanged", () => {
    renderOverview(panel, parseDashboard(withDelta(undefined)));
    expect(hero().querySelector("[data-part=delta]")).toBeNull();
    expect(one(".today-band")).not.toBeNull();
  });

  it("refuses a delta that is there but malformed", () => {
    for (const bad of [
      { hour: 24, delta_f: 3, basis: "earlier_years" },
      { hour: 15, delta_f: "warm", basis: "earlier_years" },
      { hour: 15, delta_f: 3, basis: "normal" },
      { delta_f: 3, basis: "earlier_years" },
      "hot",
    ]) {
      expect(() => parseDashboard(withDelta(bad))).toThrow(SnapshotContractError);
    }
  });
});

describe("a snapshot without the usual band (N−1)", () => {
  it.each(SCENARIOS)("parses the %s golden without it", (scenario) => {
    const snapshot = parseDashboard(withoutBand(scenario));
    expect(snapshot.temperature_context.usual_band ?? null).toBeNull();
  });

  it("draws today's line and rain with no band, median or basis", () => {
    renderOverview(panel, parseDashboard(withoutBand("raining")));
    expect(one(".today-line")).not.toBeNull();
    expect(one(".today-dot")).not.toBeNull();
    expect(all(".today-rain rect[data-rain]").length).toBeGreaterThan(0);
    expect(one(".today-band")).toBeNull();
    expect(one(".today-median")).toBeNull();
    expect(one(".today-basis")).toBeNull();
    expect(one("svg[role=img]")!.getAttribute("aria-label")).toContain("no usual band");
  });

  it("refuses a usual band that is there but malformed", () => {
    const doc = structuredClone(golden("seasoned", "dashboard")) as Record<string, unknown>;
    const context = doc.temperature_context as Record<string, unknown>;
    context.usual_band = { basis: "normal", hours: [] };
    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
    context.usual_band = {
      ...(
        golden("seasoned", "dashboard") as never as {
          temperature_context: { usual_band: object };
        }
      ).temperature_context.usual_band,
      hours: [{ hour: 0, median: "warm" }],
    };
    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });
});
