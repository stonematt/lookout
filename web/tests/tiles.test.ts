// @vitest-environment happy-dom
/**
 * The Overview as tiles (#334), through `startPage` on the goldens: each
 * tile opens with its kicker and finding, folds its source (the feature and
 * mosaic tiles, one link each, fold nothing), carries a hatched chip only
 * when a gap hides part of its chart (never the archive lag on the Overview,
 * #458), and states every number in the live value. The chip rule is also
 * held on the sections that left the Overview for the zooms (#472).
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { setLayout } from "../src/layout.js";
import { almanacGrid, rollingRain } from "../src/overview/almanac.js";
import { renderOverview } from "../src/overview/index.js";
import { againstNormal, temperaturePeriods, tenDays } from "../src/overview/temperature.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
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

/**
 * The sections that left the Overview with the accordion (#472), the zooms'
 * content, rendered as tiles in a panel of their own.
 */
function zoom(snapshot: DashboardSnapshot): HTMLElement {
  setLayout("four");
  const panel = document.createElement("div");
  document.body.append(panel);
  renderOverview(panel, snapshot, [
    temperaturePeriods,
    againstNormal,
    tenDays,
    almanacGrid,
    rollingRain,
  ]);
  return panel;
}

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
      "feature",
      // The mosaic (#472), in its fixed order.
      "temperature",
      "rain",
      "last-10-days",
      "dry-spell",
      "wind",
      "pressure",
      "air",
      "sun-moon",
    ]);
    for (const tile of tiles) {
      if (tile.dataset.section === "feature") {
        // The feature tile is one link holding its kicker first, then its finding.
        const link = tile.querySelector("a")!;
        expect([...tile.children]).toEqual([link]);
        expect(link.querySelector("[data-part]")!.getAttribute("data-part")).toBe("kicker");
        expect(words(link.querySelector("[data-part=finding]"))).not.toBe("");
        continue;
      }
      if (tile.dataset.mosaic) {
        // A mosaic tile is one link holding its title first (mosaic.test).
        const link = tile.querySelector("a")!;
        expect([...tile.children]).toEqual([link]);
        expect(link.firstElementChild!.getAttribute("data-part")).toBe("kicker");
        continue;
      }
      const [first, second] = [...tile.children];
      // A quick-fact tile's kicker sits over its one value (DESIGN.md "Quick-fact tiles").
      const next = tile.dataset.fact ? "fact" : "finding";
      expect(first!.getAttribute("data-part"), tile.dataset.section).toBe("kicker");
      expect(second!.getAttribute("data-part"), tile.dataset.section).toBe(next);
      expect(words(first)).not.toBe("");
      expect(words(second)).not.toBe("");
    }
  });

  it("folds no source on any tile: one footer lists them (prototype decision 22)", async () => {
    const panel = await boot(parsed(scenario));

    expect(panel.querySelector(".tile details, .tile .ov-source")).toBeNull();
    const footer = panel.querySelector("footer.ov-sources")!;
    expect(footer).not.toBeNull();
    expect(footer.closest("main")).not.toBeNull();
    // The tiles that draw lead the footer in their order (overview.test has the lines).
    expect(footer.querySelectorAll("dt").length).toBeGreaterThanOrEqual(5);
  });

  it("holds the temperature, then today against the usual band, in the now tile", async () => {
    const panel = await boot(parsed(scenario));
    const now = panel.querySelector<HTMLElement>("[data-section=first-screen]")!;

    expect(now.dataset.surface).toBe("now");
    expect(now.querySelector("[data-part=right-now], [data-part=ribbon]")).toBeNull();
    const temperature = now.querySelector("[data-part=temperature]")!;
    const chart = now.querySelector("[data-part=today-chart]")!;
    expect(
      temperature.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("marks the now and feature tiles as the phone's fixed top, and no other", async () => {
    const panel = await boot(parsed(scenario));
    const top = [...panel.querySelectorAll<HTMLElement>(".tile[data-phone-top]")];

    expect(top.map((t) => t.dataset.section)).toEqual(["first-screen", "feature"]);
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
    expect(new Set(nows)).toEqual(new Set(nows.length ? [t] : []));
    // The hero and the Temperature tile print the one live reading.
    expect(words(panel.querySelector("[data-section=first-screen] [data-part=temperature]"))).toBe(
      `${t}°F`,
    );
    expect(words(panel.querySelector("[data-section=temperature] [data-part=value]"))).toBe(
      `${t}°`,
    );
    // Every "<x> in so far today" is the live total.
    const today = snapshot.current.rain_today_in;
    for (const m of text.matchAll(/(\d+\.\d+) in so far today/g))
      expect(m[1]).toBe(today?.toFixed(2));
  });
});

describe("the hatched chip on a tile's closed source line", () => {
  it("names the archive lag on no Overview tile, though a golden lags past 15 minutes", async () => {
    // Every golden lags: dry by 3 h, raining and seasoned by 30 m. The header
    // states freshness once instead (#458); the other tabs keep the chip (tabs-tiles).
    for (const scenario of SCENARIOS) {
      const snapshot = parsed(scenario);
      expect(archiveLagMs(snapshot)).toBeGreaterThan(15 * 60_000);
      const panel = await boot(snapshot);
      const lag = lagCaveat(snapshot)!;
      for (const tile of panel.querySelectorAll<HTMLElement>(".tile")) {
        expect(chips(panel)[tile.dataset.section!] ?? [], tile.dataset.section).not.toContain(lag);
      }
      page!.stop();
    }
    expect(lagCaveat(parsed("dry"))).toBe("archive 3 h behind");
  });

  it("names a gap that hides part of a chart, on that tile alone", () => {
    // The raining golden, archive level with its reading: only gaps remain.
    const panel = zoom(level(parsed("raining")));

    // The windows and days before its record began are not drawn, so not
    // counted (#470); the strips with no earlier year still are.
    expect(chips(panel)).toEqual({
      "against-normal": ["2 strips not compared"],
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

    expect(chips(await boot(snapshot))).toEqual({});
    expect(chips(zoom(snapshot))).toEqual({});
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
    const panel = zoom(snapshot);

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
