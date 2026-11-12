// @vitest-environment happy-dom
/**
 * Quiet tiles (#458, Mosaic Overview release 1), through `startPage` on the
 * goldens at phone and desktop width: no Overview tile carries a standfirst
 * or an archive lag chip, each tile's accessible name is its finding, and
 * the header states freshness once. The Rain, Solar and Rain events tabs
 * keep their standfirsts and lag chips until a later release.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { SECTIONS as RAIN_SECTIONS } from "../src/rain/index.js";
import { lagCaveat } from "../src/tiles.js";
import { SCENARIOS, golden } from "./golden.js";
import { viewport } from "./viewport.js";

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

beforeEach(() => {
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
});

afterEach(() => {
  page?.stop();
  page = null;
  window.matchMedia = realMatchMedia;
  document.documentElement.removeAttribute("data-layout");
});

/** Boot the page at a tab and width; `/current` and `/event` never answer. */
async function boot(snapshot: DashboardSnapshot, width: number, tab = "overview"): Promise<void> {
  viewport(width).install();
  history.replaceState(null, "", `/#${tab}`);
  globalThis.fetch = (() => new Promise<Response>(() => {})) as typeof fetch;
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
  });
  await page.booted;
}

const panel = (tab: string) => document.getElementById(`panel-${tab}`)!;
const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();

/** Every section host on the Overview: a tile, or on a phone a finding row. */
const sections = () => [...panel("overview").querySelectorAll<HTMLElement>("[data-section]")];

const WIDTHS = [390, 1440] as const;

describe.each(SCENARIOS)("the %s golden's Overview", (scenario) => {
  const snapshot = parseDashboard(golden(scenario, "dashboard"));

  it.each(WIDTHS)("renders no standfirst at %ipx", async (width) => {
    await boot(snapshot, width);

    expect(sections().length).toBeGreaterThan(5);
    expect(panel("overview").querySelectorAll("[data-part=standfirst], .standfirst")).toHaveLength(
      0,
    );
  });

  it.each(WIDTHS)("renders no archive lag chip at %ipx", async (width) => {
    // Every golden lags past 15 minutes, so every tile would have one.
    const lag = lagCaveat(snapshot);
    expect(lag).not.toBeNull();
    await boot(snapshot, width);

    const chips = [...panel("overview").querySelectorAll("[data-part=caveat]")].map(words);
    expect(chips).not.toContain(lag);
    for (const chip of chips) expect(chip).not.toMatch(/behind/);
    expect(words(panel("overview"))).not.toContain(lag!);
  });

  it.each(WIDTHS)("keeps the gap chips, which are not the lag (at %ipx)", async (width) => {
    await boot(snapshot, width);

    const chips = [...panel("overview").querySelectorAll("[data-part=caveat]")].map(words);
    expect(chips.some((chip) => /not compared|missing/.test(chip!))).toBe(true);
  });

  it.each(WIDTHS)("names each tile by its finding at %ipx", async (width) => {
    await boot(snapshot, width);

    for (const host of sections()) {
      const name = host.getAttribute("aria-label");
      const id = host.dataset.section;
      expect(name, id).toBeTruthy();
      const finding = host.querySelector("h2.tile-finding");
      if (finding) expect(name, id).toBe(words(finding));
    }
    const named = (id: string) =>
      panel("overview").querySelector(`[data-section="${id}"]`)!.getAttribute("aria-label");
    const c = snapshot.current;
    expect(named("first-screen")).toBe(`${c.temperature_f!.toFixed(1)}°F now.`);
    expect(named("wind")).toMatch(new RegExp(`^Wind ${c.wind_speed_mph!.toFixed(1)} mph, `));
    expect(named("pressure")).toBe(
      `Pressure ${c.barometer_inhg!.toFixed(2)} inHg, ${c.barometer_trend} over the last 3 hours.`,
    );
    expect(named("air")).toMatch(
      new RegExp(`^Humidity ${Math.round(c.humidity_pct!)}%; dew point`),
    );
    expect(named("sun-moon")).toMatch(/^Sun up \d+:\d\d [ap]m to \d+:\d\d [ap]m; .+ of daylight/);
  });
});

describe("the other tabs, until their own release", () => {
  const snapshot = parseDashboard(golden("raining", "dashboard"));

  it("keep their standfirsts and lag chips", async () => {
    await boot(snapshot, 1440, "rain");
    const lag = lagCaveat(snapshot)!;

    expect(panel("rain").querySelectorAll("[data-part=standfirst]").length).toBeGreaterThan(0);
    const chips = [...panel("rain").querySelectorAll("[data-part=caveat]")].map(words);
    expect(chips).toContain(lag);
  });

  it("keep their tiles' names from their registries", async () => {
    await boot(snapshot, 1440, "rain");

    const names = [...panel("rain").querySelectorAll<HTMLElement>(".tile")].map((t) =>
      t.getAttribute("aria-label"),
    );
    expect(names).toEqual(RAIN_SECTIONS.map((s) => s.label));
  });
});

describe("the header's one freshness line", () => {
  it.each(WIDTHS)("is one line on a fresh golden at %ipx", async (width) => {
    const snapshot = parseDashboard(golden("raining", "dashboard"));
    // Fresh: observed a minute before the browser's clock.
    const fresh = { ...snapshot, observed_at_ms: Date.now() - 60_000 };
    await boot(fresh, width);

    const lines = [...document.querySelectorAll<HTMLElement>("#masthead .masthead-line")].filter(
      (line) => !line.hidden,
    );
    expect(lines).toHaveLength(1);
    expect(lines[0]!.dataset.stale).toBe("false");
  });

  it.each(WIDTHS)("still reads as stale on a stale golden at %ipx", async (width) => {
    // The goldens served as recorded are long stale.
    await boot(parseDashboard(golden("dry", "dashboard")), width);

    const line = document.querySelector<HTMLElement>("#masthead .masthead-reading")!;
    expect(line.dataset.stale).toBe("true");
    expect(words(line.querySelector(".masthead-age"))).toMatch(/Stale/);
  });
});
