// @vitest-environment happy-dom
/**
 * The Rain, Solar and Rain events tabs as tiles (#335), through `startPage`
 * on the goldens: each section is a tile opening with its kicker and
 * finding, declares whether it is in its tab's phone fixed top, folds its
 * source with the Overview's chip rule, and shows each number the same as
 * the Overview does (story 53).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { lagCaveat } from "../src/tiles.js";
import type { Scenario } from "./golden.js";
import { SCENARIOS, golden } from "./golden.js";

let page: Page | null = null;

beforeEach(() => {
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
});

afterEach(() => {
  page?.stop();
  page = null;
});

const parsed = (scenario: Scenario): DashboardSnapshot =>
  parseDashboard(golden(scenario, "dashboard"));

/** The snapshot with its archive caught up to its reading: no lag to name. */
function level(snapshot: DashboardSnapshot): DashboardSnapshot {
  const copy = structuredClone(snapshot);
  copy.archive_newest_ms = copy.observed_at_ms;
  return copy;
}

/**
 * Boot the page on a snapshot at a tab, `/current` and `/event` never
 * answering, and draw each other tab named by selecting it.
 */
async function boot(snapshot: DashboardSnapshot, ...tabs: string[]): Promise<void> {
  history.replaceState(null, "", `/#${tabs[0] ?? "rain"}`);
  // Rain events reads a completed event's curve; it never answers here.
  globalThis.fetch = (() => new Promise<Response>(() => {})) as typeof fetch;
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
  });
  await page.booted;
  for (const tab of tabs.slice(1)) {
    document.querySelector<HTMLElement>(`[aria-controls=panel-${tab}]`)!.click();
  }
}

const panel = (tab: string) => document.getElementById(`panel-${tab}`)!;
const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const tiles = (tab: string) => [...panel(tab).querySelectorAll<HTMLElement>(".tile-grid > .tile")];

/** Each tile's chips, by section id, for the tiles that have any. */
function chips(tab: string): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const tile of tiles(tab)) {
    const found = [...tile.querySelectorAll("[data-part=caveat]")].map((c) => words(c)!);
    if (found.length) out[tile.dataset.section!] = found;
  }
  return out;
}

/** The tabs, their sections in the desktop board's order, and the phone top. */
const TABS = {
  rain: { sections: ["daily", "years", "dry", "hours"], top: ["daily"] },
  solar: { sections: ["now", "daily", "day", "heatmaps"], top: ["now"] },
  events: { sections: ["catalog", "events"], top: ["catalog"] },
} as const;

describe.each(SCENARIOS)("the %s golden's other tabs", (scenario) => {
  it.each(Object.keys(TABS) as (keyof typeof TABS)[])(
    "renders every %s section as a tile whose first text is its kicker, then its finding",
    async (tab) => {
      await boot(parsed(scenario), tab);

      expect(tiles(tab).map((t) => t.dataset.section)).toEqual(TABS[tab].sections);
      for (const tile of tiles(tab)) {
        const [first, second] = [...tile.children];
        expect(first!.getAttribute("data-part"), tile.dataset.section).toBe("kicker");
        expect(second!.getAttribute("data-part"), tile.dataset.section).toBe("finding");
        expect(words(first), tile.dataset.section).not.toBe("");
        expect(words(second), tile.dataset.section).not.toBe("");
      }
    },
  );

  it.each(Object.keys(TABS) as (keyof typeof TABS)[])(
    "marks the %s tab's phone fixed top, and no other tile",
    async (tab) => {
      await boot(parsed(scenario), tab);
      const top = tiles(tab).filter((t) => t.dataset.phoneTop === "true");

      expect(top.map((t) => t.dataset.section)).toEqual(TABS[tab].top);
    },
  );

  it.each(Object.keys(TABS) as (keyof typeof TABS)[])(
    "folds every %s source paragraph into one closed note per tile",
    async (tab) => {
      await boot(parsed(scenario), tab);

      for (const tile of tiles(tab)) {
        const notes = tile.querySelectorAll<HTMLDetailsElement>("details[data-part=source-note]");
        expect(notes.length, tile.dataset.section).toBe(1);
        const note = notes[0]!;
        expect(note.open).toBe(false);
        expect(words(note.querySelector("summary"))).toMatch(/^Source · .+, archived /);
        const method = note.querySelector("[data-part=source]")!;
        expect(method.closest("details")).toBe(note);
        expect(words(method)!.length, tile.dataset.section).toBeGreaterThan(40);
      }
      // No source paragraph is left open beside a tile's chart: the old
      // open source lines are gone, and every method paragraph is folded.
      expect(panel(tab).querySelectorAll(".rain-source, .ev-source, .solar-title")).toHaveLength(0);
      for (const p of panel(tab).querySelectorAll("p:not([data-part=kicker])")) {
        if (/^(Source\b|The (station|energy catalog|Event Catalog|live station))/.test(words(p)!)) {
          expect(p.closest("details"), words(p)).not.toBeNull();
        }
      }
    },
  );

  it.each(Object.keys(TABS) as (keyof typeof TABS)[])(
    "names the archive lag on every %s tile, the golden lagging past 15 minutes",
    async (tab) => {
      const snapshot = parsed(scenario);
      await boot(snapshot, tab);
      const lag = lagCaveat(snapshot)!;

      expect(lag).not.toBeNull();
      for (const tile of tiles(tab)) {
        expect(chips(tab)[tile.dataset.section!], tile.dataset.section).toContain(lag);
      }
    },
  );
});

describe("the gap chip on these tabs", () => {
  it("names what each chart hatches on the dry golden, archive caught up", async () => {
    // The dry golden's archive stops two days short: every window is not
    // archived yet, and the bars hatch those two days.
    await boot(level(parsed("dry")), "rain", "solar", "events");

    expect(chips("rain")).toEqual({
      daily: ["7 windows, 2 days not compared"],
      years: ["1 water year not compared"],
      dry: ["3 figures not compared"],
      hours: ["2 days not compared"],
    });
    expect(chips("solar")).toEqual({
      now: ["usual range not compared"],
      daily: ["6 windows, 2 days not compared"],
      // The day it opens on is today, not archived yet: hatched in place.
      day: ["1 day not compared"],
      heatmaps: ["2 days not compared"],
    });
    expect(chips("events")).toEqual({ catalog: ["1 water year not compared"] });
  });

  it("shows no chip where nothing lags and no gap hides a chart", async () => {
    // The raining golden, caught up: its windows only start before the
    // archive, which the bars leave blank.
    await boot(level(parsed("raining")), "rain", "solar", "events");

    expect(chips("rain").hours).toBeUndefined();
    expect(chips("solar").day).toBeUndefined();
    expect(chips("events").events).toBeUndefined();
  });
});

describe("the event list through the page", () => {
  const open = () => [...panel("events").querySelectorAll<HTMLElement>(".ev-row[data-open]")];

  it("opens the live event's row with its curve on the raining golden", async () => {
    // Plotly may not draw in happy-dom; the chart host is what counts here.
    const quiet = vi.spyOn(console, "error").mockImplementation(() => {});
    const snapshot = parsed("raining");
    await boot(snapshot, "events");
    const live = snapshot.rain_events.events.find((e) => e.live)!;

    expect(open().map((r) => Number(r.dataset.start))).toEqual([live.start_ms]);
    expect(open()[0]!.querySelector(".ev-live")).not.toBeNull();
    // The live curve is the snapshot's own: drawn at once, no read.
    const chart = open()[0]!.querySelector<HTMLElement>(".ev-charts .ev-chart")!;
    expect(chart.getAttribute("aria-label")).toContain("Rain so far in this event");
    quiet.mockRestore();
  });

  it("opens the newest row, with the latest chip, on the dry golden", async () => {
    const snapshot = parsed("dry");
    await boot(snapshot, "events");
    const newest = Math.max(...snapshot.rain_events.events.map((e) => e.start_ms));

    expect(open().map((r) => Number(r.dataset.start))).toEqual([newest]);
    expect(words(open()[0]!.querySelector(".ev-latest"))).toBe("latest");
    // Its curve file is read when the row opens; here the read never answers.
    expect(words(open()[0]!.querySelector(".ev-charts"))).toBe("Loading the curve…");
  });
});

describe("an empty Event Catalog", () => {
  it("keeps both tiles, the list saying there are no events under its filter bar", async () => {
    const snapshot = parsed("raining");
    snapshot.rain_events = {
      events: [],
      all_time: { count: 0, total_in: 0, median_in: null, mean_duration_minutes: null },
      seasons: [],
      open_curve: null,
    };
    await boot(snapshot, "events");

    expect(tiles("events").map((t) => t.dataset.section)).toEqual(["catalog", "events"]);
    expect(words(panel("events").querySelector("[data-section=events] .ev-bar ~ .ev-list"))).toBe(
      "No rain events on record yet.",
    );
  });
});

describe("one number wherever it appears (story 53)", () => {
  it("keeps the Rain tab stat exact while the Overview calm rule speaks it", async () => {
    await boot(parsed("dry"), "rain", "overview");
    const since = words(
      [...panel("rain").querySelectorAll("[data-section=dry] dl > div")]
        .find((d) => words(d.querySelector("dt")) === "Since it last rained")
        ?.querySelector("dd"),
    );
    const why = words(
      panel("overview").querySelector("[data-section=hero-lead] details[data-part=why]"),
    );

    expect(since).toBe("1 d 14 h");
    expect(why).toContain("calm: about 39 hours since it last rained");
  });

  it("names the catalog's wettest event with the catalog's own total", async () => {
    const snapshot = parsed("raining");
    await boot(snapshot, "events");
    const wettest = Math.max(...snapshot.rain_events.events.map((e) => e.total_in)).toFixed(2);

    expect(words(panel("events").querySelector("[data-section=catalog] h2"))).toContain(
      `${wettest} in`,
    );
  });
});
