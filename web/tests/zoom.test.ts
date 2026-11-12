// @vitest-environment happy-dom
/**
 * The zoom shell and the temperature zoom (#474, spec #460 stories 33–35,
 * 44–45, 54–57, 59), through the page: `#overview/<slug>` opens a zoom and
 * leaving the route closes it; a shared link opens it directly; Back, Close
 * and Escape close it; focus moves to the zoom's heading and back to the
 * link that opened it; on the grids the zoom is a full-width row under the
 * tapped tile's row (under the feature block for a tile beside it) and the
 * tile carries `aria-expanded` and `aria-controls`; on a phone it takes the
 * screen. Reduced motion opens it without animation. A slug with no zoom yet falls back
 * to the Overview.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { ZOOMS } from "../src/overview/zooms/index.js";
import { isZoomRoute, zoomFor } from "../src/overview/zoom.js";
import { golden } from "./golden.js";

const parsed = (): DashboardSnapshot =>
  parseDashboard(structuredClone(golden("seasoned", "dashboard")));

/** A stand-in `matchMedia`: the viewport's width, and reduced motion or not. */
function media(width: number, reduce = false): void {
  window.matchMedia = ((query: string) => {
    const found = /min-width:\s*(\d+)px/.exec(query);
    const matches = found
      ? width >= Number(found[1])
      : query.includes("prefers-reduced-motion: reduce")
        ? reduce
        : false;
    return {
      matches,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaQueryList;
  }) as typeof window.matchMedia;
}

/**
 * Follow a link as a browser does: its click, then (unless a handler took
 * the click) a new history entry and a hash change.
 */
function follow(event: Event): void {
  if (event.defaultPrevented) return;
  const link = (event.target as Element).closest?.("a[href^='#']");
  if (!link) return;
  event.preventDefault();
  history.pushState(null, "", link.getAttribute("href")!);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

/** Go back `n` history entries, firing the hash change a browser fires. */
function back(n = 1): void {
  history.go(-n);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

let page: Page | null = null;
const realMatchMedia = window.matchMedia;

beforeEach(() => {
  document.addEventListener("click", follow);
  // Step back through the test's own stack: happy-dom's `go` is a no-op.
  stack = [location.href];
});

afterEach(() => {
  page?.stop();
  page = null;
  document.removeEventListener("click", follow);
  window.matchMedia = realMatchMedia;
  vi.restoreAllMocks();
});

let stack: string[] = [];

async function boot(hash: string, width = 1440, reduce = false): Promise<DashboardSnapshot> {
  media(width, reduce);
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
  history.replaceState(null, "", `/${hash}`);
  stack = [location.href];
  vi.spyOn(history, "pushState").mockImplementation(function (this: History, data, _t, url) {
    History.prototype.replaceState.call(history, data, "", url);
    stack.push(location.href);
  });
  vi.spyOn(history, "go").mockImplementation((delta = 0) => {
    stack = stack.slice(0, Math.max(1, stack.length + delta));
    History.prototype.replaceState.call(history, null, "", stack.at(-1)!);
  });
  const snapshot = parsed();
  globalThis.fetch = (() => new Promise<Response>(() => {})) as typeof fetch;
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
  });
  await page.booted;
  return snapshot;
}

const overview = () => document.getElementById("panel-overview")!;
const zoom = () => overview().querySelector<HTMLElement>(".zoom");
const tileLink = (slug = "temperature") =>
  overview().querySelector<HTMLAnchorElement>(`[data-section="${slug}"] a.mosaic-link`)!;
const key = (target: Element, name: string) =>
  target.dispatchEvent(new KeyboardEvent("keydown", { key: name, bubbles: true }));

describe("the zoom registry", () => {
  it("registers the temperature zoom by its tile's slug, one zoom per slug", () => {
    expect(ZOOMS.map((z) => z.slug)).toContain("temperature");
    expect(new Set(ZOOMS.map((z) => z.slug)).size).toBe(ZOOMS.length);
    expect(zoomFor("#overview/temperature")?.title).toBe("Temperature");
    expect(isZoomRoute("overview/temperature")).toBe(true);
  });

  it("claims no route a zoom has not registered", () => {
    expect(isZoomRoute("overview/wind")).toBe(false);
    expect(isZoomRoute("overview")).toBe(false);
    expect(isZoomRoute("rain/temperature")).toBe(false);
  });
});

describe("a zoom on the four-column grid", () => {
  it("opens from its tile: a full-width row, focus on its heading, the tile expanded", async () => {
    await boot("#overview");
    expect(zoom()).toBeNull();
    expect(tileLink().getAttribute("aria-expanded")).toBe("false");
    expect(tileLink().getAttribute("aria-controls")).toBe("zoom-temperature");

    tileLink().click();

    expect(location.hash).toBe("#overview/temperature");
    const el = zoom()!;
    expect(el.dataset.mode).toBe("row");
    expect(el.id).toBe("zoom-temperature");
    expect(el.parentElement!.classList.contains("mosaic-grid")).toBe(true);
    expect(tileLink().getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById(tileLink().getAttribute("aria-controls")!)).toBe(el);
    expect(document.activeElement).toBe(el.querySelector(".zoom-heading"));
    expect(el.getAttribute("aria-labelledby")).toBe("zoom-temperature-heading");
    expect(el.querySelector(".zoom-close")?.textContent).toContain("Close");
  });

  it("closes on Close: back to the Overview's entry, focus on the tile", async () => {
    await boot("#overview");
    tileLink().click();
    zoom()!.querySelector<HTMLButtonElement>(".zoom-close")!.click();
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    expect(history.go).toHaveBeenCalledWith(-1);
    expect(location.hash).toBe("#overview");
    expect(zoom()).toBeNull();
    expect(tileLink().getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(tileLink());
  });

  it("closes on Escape from inside it, focus back on the tile", async () => {
    await boot("#overview");
    tileLink().click();
    key(zoom()!.querySelector(".zoom-heading")!, "Escape");
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    expect(zoom()).toBeNull();
    expect(location.hash).toBe("#overview");
    expect(document.activeElement).toBe(tileLink());
  });

  it("closes on Escape from the tile that controls it", async () => {
    await boot("#overview");
    tileLink().click();
    tileLink().focus();
    key(tileLink(), "Escape");
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    expect(zoom()).toBeNull();
    expect(document.activeElement).toBe(tileLink());
  });

  it("replaces the route on Close when another part of the page pushed the entry", async () => {
    await boot("#overview");
    tileLink().click();
    // A station chosen in the zoom pushes its own entry, with no zoom stamp.
    history.pushState(null, "", location.href);
    zoom()!.querySelector<HTMLButtonElement>(".zoom-close")!.click();

    expect(history.go).not.toHaveBeenCalled();
    expect(location.hash).toBe("#overview");
    expect(zoom()).toBeNull();
    expect(history.state).toBeNull();
  });

  it("closes on the browser's Back", async () => {
    await boot("#overview");
    tileLink().click();
    back();

    expect(location.hash).toBe("#overview");
    expect(zoom()).toBeNull();
    expect(document.activeElement).toBe(tileLink());
  });

  it("closes when its own tile is followed again, as aria-expanded says", async () => {
    await boot("#overview");
    tileLink().click();
    tileLink().click();
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    expect(zoom()).toBeNull();
    expect(location.hash).toBe("#overview");
  });

  it("opens directly from a shared link, and Close replaces the route", async () => {
    await boot("#overview/temperature");
    expect(zoom()).not.toBeNull();
    expect(tileLink().getAttribute("aria-expanded")).toBe("true");

    zoom()!.querySelector<HTMLButtonElement>(".zoom-close")!.click();

    // Nothing behind a shared link is the site's to step back to.
    expect(history.go).not.toHaveBeenCalled();
    expect(location.hash).toBe("#overview");
    expect(zoom()).toBeNull();
    expect(document.activeElement).toBe(tileLink());
  });

  it("sits under the feature block for a tile beside it", async () => {
    // Lay the grid out as the four-column grid does: the now tile, the
    // feature two rows deep with the first four tiles beside it, then rows of four.
    const rows: Record<string, [number, number]> = {
      now: [0, 200],
      feature: [216, 616],
      temperature: [216, 408],
      rain: [216, 408],
      "last-10-days": [424, 616],
      "dry-spell": [424, 616],
    };
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (
      this: Element,
    ) {
      const section = (this as HTMLElement).dataset?.section ?? "";
      const [top, bottom] = rows[section] ?? [632, 800];
      return { top, bottom, left: 0, right: 0, width: 0, height: bottom - top } as DOMRect;
    });
    await boot("#overview");
    tileLink().click();

    const before = zoom()!.previousElementSibling as HTMLElement;
    expect(before.dataset.section).toBe("dry-spell");
    expect((zoom()!.nextElementSibling as HTMLElement).dataset.section).toBe("wind");
  });

  it("draws the temperature zoom: the number, the window chart with Usual on top, the strips", async () => {
    const snapshot = await boot("#overview/temperature");
    const el = zoom()!;

    expect(el.querySelector(".zoom-heading")?.textContent).toBe("Temperature");
    expect(el.querySelector("[data-part=value]")?.textContent).toBe(
      `${snapshot.current.temperature_f!.toFixed(1)}°`,
    );
    const panels = [...el.querySelectorAll<HTMLElement>(".zoom-panel")];
    expect(panels.map((p) => p.dataset.panel)).toEqual(["periods", "strips"]);
    expect(panels.map((p) => p.querySelector("h3")?.textContent)).toEqual([
      "Temperature, in context",
      "Now against this date",
    ]);

    const chart = panels[0]!.querySelector("svg[role=img]")!;
    const rows = [...chart.querySelectorAll(".tc-row")].map((t) => t.textContent);
    expect(rows[0]).toBe("Usual");
    expect(rows).toContain("Full history");
    expect(chart.querySelector("[data-row=usual]")).not.toBeNull();
    expect(panels[0]!.querySelector(".ov-sr")?.textContent).toMatch(/^Usual \(/);
    expect(panels[1]!.querySelectorAll("figure.tc-strip").length).toBeGreaterThan(0);
  });

  it("carries no source note or method fold (prototype decision 22)", async () => {
    await boot("#overview/temperature");
    expect(zoom()!.querySelector("details, .tile-folds")).toBeNull();
    // The sources footer still holds the tile's line.
    expect(
      overview().querySelector("[data-part=sources] [data-source=Temperature]"),
    ).not.toBeNull();
  });

  it("closes when another tab opens", async () => {
    await boot("#overview");
    tileLink().click();
    document.querySelector<HTMLButtonElement>("#tab-rain")!.click();

    expect(location.hash).toBe("#rain");
    expect(zoom()).toBeNull();
  });

  it("stays open, without taking focus, when a new snapshot redraws the Overview", async () => {
    await boot("#overview/temperature");
    const outside = document.querySelector<HTMLButtonElement>("#tab-overview")!;
    outside.focus();
    const { renderOverview } = await import("../src/overview/index.js");
    renderOverview(overview(), parsed());

    expect(zoom()).not.toBeNull();
    expect(document.activeElement).toBe(outside);
  });

  it("opens from the feature tile and returns focus to it", async () => {
    await boot("#overview");
    const feature = overview().querySelector<HTMLAnchorElement>(
      '[data-section=feature] a[href="#overview/temperature"]',
    );
    expect(feature).not.toBeNull();
    feature!.click();
    expect(zoom()).not.toBeNull();
    key(zoom()!, "Escape");
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    expect(document.activeElement).toBe(feature);
  });
});

describe("a zoom on a phone", () => {
  it("takes the screen: the mosaic and the tab bar step aside, focus on the heading", async () => {
    await boot("#overview", 390);
    expect(tileLink().hasAttribute("aria-expanded")).toBe(false);

    tileLink().click();

    const el = zoom()!;
    expect(el.dataset.mode).toBe("screen");
    expect(overview().hasAttribute("data-zoomed")).toBe(true);
    expect(document.documentElement.hasAttribute("data-zoom-screen")).toBe(true);
    expect(el.parentElement).toBe(overview());
    expect(document.activeElement).toBe(el.querySelector(".zoom-heading"));
    expect(el.querySelector(".zoom-close")).toBeNull();
  });

  it("goes back on its Overview button, focus on the tile", async () => {
    await boot("#overview", 390);
    tileLink().click();
    const button = zoom()!.querySelector<HTMLButtonElement>(".zoom-back")!;
    expect(button.textContent).toBe("‹Back to the Overview");
    button.click();
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    expect(zoom()).toBeNull();
    expect(overview().hasAttribute("data-zoomed")).toBe(false);
    expect(document.documentElement.hasAttribute("data-zoom-screen")).toBe(false);
    expect(document.activeElement).toBe(tileLink());
  });

  it("closes on the browser's Back", async () => {
    await boot("#overview", 390);
    tileLink().click();
    back();

    expect(zoom()).toBeNull();
    expect(document.activeElement).toBe(tileLink());
  });
});

describe("a tile with no zoom yet", () => {
  it("keeps its link, and its route falls back to the Overview", async () => {
    await boot("#overview");
    const wind = tileLink("wind");
    expect(wind.getAttribute("href")).toBe("#overview/wind");
    expect(wind.hasAttribute("data-zoom")).toBe(false);
    expect(wind.hasAttribute("aria-expanded")).toBe(false);

    wind.click();

    expect(location.hash).toBe("#overview");
    expect(overview().hidden).toBe(false);
    expect(zoom()).toBeNull();
  });
});

describe("reduced motion", () => {
  it("scrolls to the zoom without smoothing", async () => {
    const scrolled = vi.fn();
    Element.prototype.scrollIntoView = scrolled;
    await boot("#overview", 1440, true);
    tileLink().click();

    expect(scrolled).toHaveBeenCalledWith({ behavior: "auto", block: "start" });
  });

  it("smooths the scroll otherwise", async () => {
    const scrolled = vi.fn();
    Element.prototype.scrollIntoView = scrolled;
    await boot("#overview", 1440, false);
    tileLink().click();

    expect(scrolled).toHaveBeenCalledWith({ behavior: "smooth", block: "start" });
  });

  it("animates the opening only under no-preference, in the stylesheet", () => {
    const css = readFileSync(resolve(import.meta.dirname, "../src/overview/zoom.css"), "utf8");
    const noPref = /@media \(prefers-reduced-motion: no-preference\) \{([\s\S]*?)\n\}/.exec(css);
    expect(noPref?.[1]).toContain("animation: zoom-open");
    // No animation or transition outside that block, so reduce gets none.
    const rest = css.replace(noPref![0], "");
    expect(rest).not.toMatch(/animation:|transition:/);
  });
});
