// @vitest-environment happy-dom
/**
 * The rain gauge (#619): rain over each window in the rain zoom, in its own
 * panel next to the water year, in place of the window strip. Through the
 * page on the goldens: three panels, each on one shared axis; this year's
 * bar in the rain colour, earlier years (`almanac.rolling`) as thin grey
 * bars to its left with a gap for a null year; a window the record does not
 * cover hides; a young station draws plain bars.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";
import type { Scenario } from "./golden.js";

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

afterEach(() => {
  page?.stop();
  page = null;
  window.matchMedia = realMatchMedia;
  vi.restoreAllMocks();
});

const parsed = (scenario: Scenario): DashboardSnapshot =>
  parseDashboard(structuredClone(golden(scenario, "dashboard")));

async function boot(hash: string, snapshot: DashboardSnapshot): Promise<void> {
  window.matchMedia = ((query: string) => {
    const found = /min-width:\s*(\d+)px/.exec(query);
    return {
      matches: found ? 1440 >= Number(found[1]) : false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaQueryList;
  }) as typeof window.matchMedia;
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
  history.replaceState(null, "", `/${hash}`);
  globalThis.fetch = (() => new Promise<Response>(() => {})) as typeof fetch;
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
  });
  await page.booted;
}

const zoom = () => document.querySelector<HTMLElement>("#panel-overview .zoom")!;
const panelIds = () =>
  [...zoom().querySelectorAll<HTMLElement>(".zoom-panel")].map((p) => p.dataset.panel);
const gauge = () => zoom().querySelector<HTMLElement>("[data-panel=rain-windows]");
const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const groups = () => [...gauge()!.querySelectorAll<SVGGElement>("g.gauge-panel")];
const column = (key: string) => gauge()!.querySelector<SVGGElement>(`g[data-window="${key}"]`);
const texts = (el: Element | null, cls: string) =>
  [...(el?.querySelectorAll(`text.${cls}`) ?? [])].map((t) => t.textContent);

describe("the rain gauge on the seasoned golden", () => {
  it("is its own panel, next to the water year, and the zoom drops the window strip", async () => {
    await boot("#overview/rain", parsed("seasoned"));
    expect(panelIds().slice(0, 2)).toEqual(["water-year", "rain-windows"]);
    expect(words(gauge()!.querySelector(".zoom-panel-title"))).toBe("Rain over each window");
    expect(zoom().querySelector(".rain-windows")).toBeNull();
  });

  it("draws three panels, each on one shared axis topped by its largest value", async () => {
    await boot("#overview/rain", parsed("seasoned"));
    expect(groups().map((g) => words(g.querySelector("text.gauge-head")))).toEqual([
      "Today and yesterday",
      "Last 7 and 30 days",
      // 365 days hides, so the heading names only what is drawn.
      "Last 90 days",
    ]);
    // 30 days' 0.44 tops the middle panel, over 7 days' 0.24.
    expect(texts(groups()[1]!, "gauge-tick")).toEqual(["0.44", "0"]);
    // Each panel draws its bars from one baseline.
    const bases = groups().map((g) =>
      [...g.querySelectorAll("rect.gauge-bar")].map(
        (r) => Number(r.getAttribute("y")) + Number(r.getAttribute("height")),
      ),
    );
    for (const panel of bases) expect(new Set(panel).size).toBe(1);
  });

  it("hides a window whose status is not ok", async () => {
    await boot("#overview/rain", parsed("seasoned"));
    expect(column("365d")).toBeNull();
    expect(column("90d")).not.toBeNull();
  });

  it("draws earlier years as grey bars, oldest first, with a gap for a null year", async () => {
    await boot("#overview/rain", parsed("seasoned"));
    const seven = column("7d")!;
    expect(texts(seven, "gauge-year")).toEqual(["'24", "'25", "'26"]);
    expect(seven.querySelectorAll("rect.gauge-prior")).toHaveLength(2);
    expect(seven.querySelector("text.gauge-year.gauge-this")?.textContent).toBe("'26");
    expect(texts(seven, "gauge-total")).toEqual(["0.24"]);

    // 30 days in 2024 is null: its label stays, its bar is a gap.
    const thirty = column("30d")!;
    expect(texts(thirty, "gauge-year")).toEqual(["'24", "'25", "'26"]);
    expect(thirty.querySelectorAll("rect.gauge-prior")).toHaveLength(1);
    expect(thirty.querySelector('rect.gauge-prior[data-year="2024"]')).toBeNull();
    expect(thirty.querySelector('rect.gauge-prior[data-year="2025"]')).not.toBeNull();
  });

  it("draws Today and Yesterday as plain bars", async () => {
    await boot("#overview/rain", parsed("seasoned"));
    for (const key of ["today", "yesterday"]) {
      expect(column(key)!.querySelector("rect.gauge-prior, text.gauge-year")).toBeNull();
      expect(column(key)!.querySelectorAll("rect.gauge-bar")).toHaveLength(1);
    }
  });

  it("keys the grey bars in one line", async () => {
    await boot("#overview/rain", parsed("seasoned"));
    expect(words(gauge()!.querySelector("[data-part=key]"))).toBe(
      "Grey: the same window in earlier years",
    );
  });

  it("titles the finding with its window and the earlier years' numbers", async () => {
    await boot("#overview/rain", parsed("seasoned"));
    expect(words(gauge()!.querySelector("[data-part=finding]"))).toBe(
      "0.44 in over the last 90 days, against ≈ 0.00 in (2025) over the same days",
    );
  });
});

describe("the rain gauge on other goldens", () => {
  it("draws plain bars for the raining golden's two ok windows, and hides the rest", async () => {
    await boot("#overview/rain", parsed("raining"));
    expect(groups()).toHaveLength(1);
    expect(texts(gauge(), "gauge-total")).toEqual(["0.30", "0.24"]);
    expect(gauge()!.querySelector("rect.gauge-prior, [data-part=key]")).toBeNull();
    expect(words(gauge()!.querySelector("[data-part=finding]"))).toBe("0.24 in yesterday");
  });

  it("is left out when the record covers no window (dry)", async () => {
    await boot("#overview/rain", parsed("dry"));
    expect(gauge()).toBeNull();
    expect(panelIds()[0]).toBe("water-year");
  });

  it("draws plain bars for a young station with no earlier years", async () => {
    const snapshot = parsed("seasoned");
    for (const row of snapshot.almanac!.rolling) row.years = row.years.slice(0, 1);
    await boot("#overview/rain", snapshot);
    expect(gauge()!.querySelector("rect.gauge-prior, text.gauge-year, [data-part=key]")).toBeNull();
    expect(texts(gauge(), "gauge-total")).toEqual(["0.00", "0.00", "0.24", "0.44", "0.44"]);
    expect(words(gauge()!.querySelector("[data-part=finding]"))).toBe(
      "0.44 in over the last 90 days",
    );
  });

  it("keeps the window strip on the Rain tab", async () => {
    await boot("#rain", parsed("seasoned"));
    expect(document.querySelector("#panel-rain .rain-windows")).not.toBeNull();
  });
});
