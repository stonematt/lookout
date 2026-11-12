// @vitest-environment happy-dom
/**
 * The Overview as tiles (#334), through `startPage` on the goldens: each
 * tile opens with its kicker and finding, folds its source and "Why this
 * leads", carries a hatched chip only when the archive lags or a gap hides
 * part of its chart, and states every number in the live value.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { heroGaps } from "../src/overview/hero.js";
import { archiveLagMs, gapReasons, lagCaveat, sourceLine } from "../src/tiles.js";
import type { Scenario } from "./golden.js";
import { SCENARIOS, golden, twoStations } from "./golden.js";

let page: Page | null = null;

beforeEach(() => {
  history.replaceState(null, "", "/#overview");
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
});

afterEach(() => {
  page?.stop();
  page = null;
});

const parsed = (scenario: Scenario): DashboardSnapshot =>
  parseDashboard(golden(scenario, "dashboard"));

/**
 * The snapshot with its archive caught up to its reading: no lag to name,
 * and the ribbon's unarchived tail archived (dry: no rain there).
 */
function level(snapshot: DashboardSnapshot): DashboardSnapshot {
  const copy = structuredClone(snapshot);
  copy.archive_newest_ms = copy.observed_at_ms;
  const rain = copy.condition_ribbon?.rain_in;
  if (rain) {
    for (let i = rain.length - 1; i >= 0 && rain[i] === null; i -= 1) rain[i] = 0;
  }
  return copy;
}

/** Boot the page on a snapshot, `/current` never answering. */
async function boot(snapshot: DashboardSnapshot, stations = false): Promise<HTMLElement> {
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
    ...(stations ? { stations: () => Promise.resolve(twoStations()) } : {}),
  });
  await page.booted;
  return document.getElementById("panel-overview")!;
}

const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();

/** Each tile's chips, by section id, for the tiles that have any. */
function chips(panel: HTMLElement): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const tile of panel.querySelectorAll<HTMLElement>(".tile")) {
    const found = [...tile.querySelectorAll("[data-part=caveat]")].map((c) => words(c)!);
    if (found.length) out[tile.dataset.section!] = found;
  }
  return out;
}

describe.each(SCENARIOS)("the %s golden's Overview", (scenario) => {
  it("renders every section as a tile whose first text is its kicker, then its finding", async () => {
    const panel = await boot(parsed(scenario));
    const tiles = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")];

    expect(tiles.map((t) => t.dataset.section)).toEqual([
      "first-screen",
      "hero-lead",
      "temperature-periods",
      "against-normal",
      "ten-days",
      "almanac",
      "rolling-rain",
      "wind",
      "pressure",
      "air",
      "sun-moon",
    ]);
    for (const tile of tiles) {
      const [first, second] = [...tile.children];
      // A quick-fact tile's kicker sits over its one value (DESIGN.md "Quick-fact tiles").
      const next = tile.dataset.fact ? "fact" : "finding";
      expect(first!.getAttribute("data-part"), tile.dataset.section).toBe("kicker");
      expect(second!.getAttribute("data-part"), tile.dataset.section).toBe(next);
      expect(words(first)).not.toBe("");
      expect(words(second)).not.toBe("");
    }
  });

  it("folds each tile's source, closed, its full method paragraph inside", async () => {
    const snapshot = parsed(scenario);
    const panel = await boot(snapshot);

    // A quick fact is the live reading, a value and a sentence: no source to fold.
    for (const tile of panel.querySelectorAll<HTMLElement>(".tile[data-fact]")) {
      expect(tile.querySelector("details"), tile.dataset.section).toBeNull();
    }
    for (const tile of panel.querySelectorAll<HTMLElement>(".tile:not([data-fact])")) {
      const notes = tile.querySelectorAll<HTMLDetailsElement>("details[data-part=source-note]");
      expect(notes.length, tile.dataset.section).toBe(1);
      const note = notes[0]!;
      expect(note.open).toBe(false);
      expect(words(note.querySelector("summary"))).toMatch(/^Source · .+, archived /);
      const method = note.querySelector("[data-part=source]")!;
      // Inside the disclosure, not beside it: one tap away, never in view closed.
      expect(method.closest("details")).toBe(note);
      expect(words(method)!.length).toBeGreaterThan(40);
    }
  });

  it('folds "Why this leads" closed, listing every rule with its live value', async () => {
    const snapshot = parsed(scenario);
    const panel = await boot(snapshot);
    const why = panel.querySelector<HTMLDetailsElement>(
      "[data-section=hero-lead] details[data-part=why]",
    )!;

    expect(why.open).toBe(false);
    expect(words(why.querySelector("summary"))).toBe("Why this leads");
    const rules = [...why.querySelectorAll("li")];
    expect(rules.map((li) => li.dataset.rule)).toEqual(snapshot.hero_lead.rules.map((r) => r.kind));
    for (const li of rules) expect(words(li)).toMatch(/\d/);
    // "Why this leads" comes before the source note, both at the tile's foot.
    const folds = why.parentElement!;
    expect(folds.className).toBe("tile-folds");
    expect([...folds.children].map((el) => el.getAttribute("data-part"))).toEqual([
      "why",
      "source-note",
    ]);
  });

  it("holds the four Right-now stats, then the ribbon, in the now tile", async () => {
    const panel = await boot(parsed(scenario));
    const now = panel.querySelector<HTMLElement>("[data-section=first-screen]")!;

    expect(now.dataset.surface).toBe("now");
    expect([...now.querySelectorAll("[data-part=right-now] dt")].map(words)).toEqual([
      "feels like",
      "wind",
      "rain today",
      "humidity",
    ]);
    const strip = now.querySelector("[data-part=right-now]")!;
    const ribbon = now.querySelector("[data-part=ribbon]")!;
    expect(strip.compareDocumentPosition(ribbon) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("marks the now and leading tiles as the phone's fixed top, and no other", async () => {
    const panel = await boot(parsed(scenario));
    const top = [...panel.querySelectorAll<HTMLElement>(".tile[data-phone-top]")];

    expect(top.map((t) => t.dataset.section)).toEqual(["first-screen", "hero-lead"]);
  });

  it("states findings in the live value, never the archive (story 54)", async () => {
    const snapshot = parsed(scenario);
    const panel = await boot(snapshot);

    for (const finding of panel.querySelectorAll("[data-part=finding]")) {
      expect(words(finding)).not.toMatch(/archiv/i);
    }
    // The archive lag sits only in the source notes.
    const lag = lagCaveat(snapshot)!;
    expect(lag).not.toBeNull();
    for (const node of panel.querySelectorAll("[data-part=caveat]")) {
      if (words(node)!.startsWith("archive")) {
        expect(node.closest("details[data-part=source-note]")).not.toBeNull();
      }
    }
    expect(words(panel.querySelector("[data-part=temperature]"))).toBe(
      `${snapshot.current.temperature_f!.toFixed(1)}°F`,
    );
  });

  it("shows each number the same wherever it appears on the page (story 53)", async () => {
    const snapshot = parsed(scenario);
    const panel = await boot(snapshot);
    const text = words(panel)!;
    const t = snapshot.current.temperature_f!.toFixed(1);

    // Every "now <temperature>" and "<temperature> now" on the Overview is the live one.
    const nows = [...text.matchAll(/now (\d+\.\d)°|(\d+\.\d)° now/g)].map((m) => m[1] ?? m[2]);
    expect(nows.length).toBeGreaterThan(1);
    expect(new Set(nows)).toEqual(new Set([t]));
    // The Right-now stats are the live reading's.
    const stat = (label: string) =>
      words(
        [...panel.querySelectorAll("[data-part=right-now] > div")]
          .find((d) => words(d.querySelector("dt")) === label)
          ?.querySelector("dd"),
      );
    const c = snapshot.current;
    expect(stat("humidity")).toBe(`${Math.round(c.humidity_pct!)}%`);
    expect(stat("wind")).toBe(`${c.wind_speed_mph!.toFixed(1)} mph`);
    expect(stat("feels like")).toBe(`${c.feels_like_f!.toFixed(1)}°`);
    // Rain today: the stat, and every "<x> in so far today".
    const today = snapshot.current.rain_today_in;
    const strip = words(
      [...panel.querySelectorAll("[data-part=right-now] > div")]
        .find((d) => words(d.querySelector("dt")) === "rain today")
        ?.querySelector("dd"),
    );
    if (today === null) expect(strip).toBe("n/a");
    else {
      expect(strip).toBe(`${today.toFixed(2)} in`);
      for (const m of text.matchAll(/(\d+\.\d+) in so far today/g))
        expect(m[1]).toBe(today.toFixed(2));
    }
  });

  it("still hatches each invalid comparison in place, with its reason", async () => {
    const snapshot = parsed(scenario);
    const panel = await boot(snapshot);
    const invalid = snapshot.temperature_context.periods.filter(
      (p) => p.status.status === "invalid",
    );

    expect(invalid.length).toBeGreaterThan(0);
    const hatched = panel.querySelectorAll(
      "[data-part=periods-chart] .tc-hatch-label, [data-part=periods-chart] text",
    );
    expect([...hatched].map((el) => el.textContent).join(" ")).toMatch(/n\/a|not archived yet/);
  });
});

describe("the hatched chip on a tile's closed source line", () => {
  it("names the archive lag on every tile of a golden that lags past 15 minutes", async () => {
    // Every golden lags: dry by 3 h, raining and seasoned by 30 m.
    for (const scenario of SCENARIOS) {
      const snapshot = parsed(scenario);
      expect(archiveLagMs(snapshot)).toBeGreaterThan(15 * 60_000);
      const panel = await boot(snapshot);
      const lag = lagCaveat(snapshot)!;
      // A quick fact states the live reading alone, so the archive lag is not its to name.
      for (const tile of panel.querySelectorAll<HTMLElement>(".tile:not([data-fact])")) {
        expect(chips(panel)[tile.dataset.section!], tile.dataset.section).toContain(lag);
      }
      for (const tile of panel.querySelectorAll<HTMLElement>(".tile[data-fact]")) {
        expect(chips(panel)[tile.dataset.section!], tile.dataset.section).toBeUndefined();
      }
      page!.stop();
    }
    expect(lagCaveat(parsed("dry"))).toBe("archive 3 h behind");
  });

  it("names a gap that hides part of a chart, on that tile alone", async () => {
    // The raining golden, archive level with its reading: only gaps remain.
    const panel = await boot(level(parsed("raining")));

    expect(chips(panel)).toEqual({
      "temperature-periods": ["4 windows not compared"],
      "against-normal": ["2 strips not compared"],
      "ten-days": ["5 days not compared"],
      "rolling-rain": ["5 windows not compared"],
    });
  });

  it("shows nowhere when nothing lags and no gap hides any chart", async () => {
    // A copy of the raining golden with its gaps closed: every window and
    // day valid, the strips set aside as not compared dropped.
    const snapshot = structuredClone(level(parsed("raining")));
    const ctx = snapshot.temperature_context;
    ctx.periods = ctx.periods.filter((p) => p.status.status === "valid");
    ctx.ten_days = ctx.ten_days.filter((d) => d.status.status === "valid");
    for (const w of snapshot.almanac.rolling) w.years = [];
    ctx.strips = ctx.strips.map((s) => ({ ...s, missing: "no_reading" as const }));
    const panel = await boot(snapshot);

    expect(chips(panel)).toEqual({});
  });

  it("names the one status changed in a copy of a golden, and only it", async () => {
    const snapshot = structuredClone(level(parsed("raining")));
    const ctx = snapshot.temperature_context;
    ctx.periods = ctx.periods.filter((p) => p.status.status === "valid");
    ctx.ten_days = ctx.ten_days.filter((d) => d.status.status === "valid");
    for (const w of snapshot.almanac.rolling) w.years = [];
    ctx.strips = ctx.strips.map((s) => ({ ...s, missing: "no_reading" as const }));
    // Yesterday's window gets a gap in the record.
    const yesterday = ctx.periods.find((p) => p.status.key === "yesterday")!;
    yesterday.status = { ...yesterday.status, status: "invalid", reason: "gap", missing_days: 1 };
    const panel = await boot(snapshot);

    expect(chips(panel)).toEqual({ "temperature-periods": ["1 window not compared"] });
  });
});

describe("the chip's conditions at their edges", () => {
  it("names the lag only past 15 minutes", () => {
    const snapshot = parsed("raining");
    const at = (lagMs: number) =>
      lagCaveat({ ...snapshot, archive_newest_ms: snapshot.observed_at_ms! - lagMs });

    expect(at(15 * 60_000)).toBeNull();
    expect(at(15 * 60_000 + 60_000)).toBe("archive 16 m behind");
    expect(lagCaveat({ ...snapshot, archive_newest_ms: null })).toBeNull();
  });

  it("names a lead whose usual band has no comparison set", async () => {
    const raining = level(parsed("raining"));
    expect(heroGaps(raining)).toBeNull();
    const bare = structuredClone(raining);
    bare.hero_lead = { ...bare.hero_lead, usual_low: null, usual_median: null, usual_high: null };
    expect(heroGaps(bare)).toBe("usual range not compared");

    const panel = await boot(bare);
    expect(chips(panel)["hero-lead"]).toEqual(["usual range not compared"]);
  });

  it("names the calm lead's usual dry spell when it is not compared", () => {
    expect(heroGaps(parsed("dry"))).toBe("usual dry spell not compared");
  });
});

describe("the source line's words", () => {
  it("dates the archive, or says nothing is archived", () => {
    const snapshot = parsed("dry");
    expect(sourceLine(snapshot, "daily rain")).toMatch(/^Source · daily rain, archived Nov 18, /);
    expect(sourceLine({ ...snapshot, archive_newest_ms: null }, "daily rain")).toBe(
      "Source · daily rain, nothing archived yet",
    );
  });

  it("counts what a gap hides, and says nothing for none", () => {
    expect(gapReasons([[1, "window", "windows"]])).toBe("1 window not compared");
    expect(gapReasons([[3, "window", "windows"]])).toBe("3 windows not compared");
    expect(gapReasons([[0, "window", "windows"]])).toBeNull();
  });
});

describe("the header", () => {
  it("keeps the station selector with two or more stations, the tabs beside it", async () => {
    await boot(parsed("raining"), true);

    const masthead = document.getElementById("masthead")!;
    expect(masthead.querySelector("select.station-select")).not.toBeNull();
    // The tab bar sits between the header and the panels, so both grids set it
    // on the header's row.
    const bar = document.querySelector("nav.tabs")!;
    expect(bar.previousElementSibling).toBe(masthead);
    expect(bar.nextElementSibling).toBe(document.getElementById("sections"));
    expect([...bar.querySelectorAll("[role=tab]")].map(words)).toEqual([
      "Overview",
      "Rain",
      "Rain Events",
      "Solar",
    ]);
  });

  it("sets the page layout switch from the start", async () => {
    await boot(parsed("dry"));

    expect(document.documentElement.getAttribute("data-layout")).toMatch(/^(phone|two|four)$/);
  });
});
