// @vitest-environment happy-dom
/**
 * The page as a visitor sees it: the header and the tabs, fed by the two
 * read functions (faked here), on the 5-minute refresh cycle. No network.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Page, PageReads } from "../src/page.js";
import { CYCLE_MS, startPage as start } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const LOADING = "Loading the dashboard";
const FAILED = "The dashboard could not be loaded.";

let masthead: HTMLElement;
let sections: HTMLElement;

beforeEach(() => {
  vi.useFakeTimers();
  history.replaceState(null, "", "/");
  document.body.innerHTML = "<div id=masthead></div><main id=sections></main>";
  masthead = document.getElementById("masthead")!;
  sections = document.getElementById("sections")!;
});

let started: Page[] = [];

/** Start a page, stopped again after the test so its cycle cannot leak. */
function startPage(host: HTMLElement, tabs: HTMLElement, reads: PageReads): Page {
  const page = start(host, tabs, reads);
  started.push(page);
  return page;
}

afterEach(() => {
  for (const page of started) page.stop();
  started = [];
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** A fresh copy of the raining snapshot, built at `builtAtMs` if given. */
function snapshot(builtAtMs?: number): DashboardSnapshot {
  const doc = golden("raining", "dashboard") as Record<string, unknown>;
  if (builtAtMs !== undefined) doc.built_at_ms = builtAtMs;
  return parseDashboard(doc);
}

/** A `/current` that never answers: these tests are about `/dashboard`. */
const quietCurrent = () => new Promise<CurrentReading>(() => {});

const panel = (id: string) => sections.querySelector<HTMLElement>(`#panel-${id}`)!;
const open = (id: string) =>
  sections.querySelector<HTMLButtonElement>(`[role=tab][data-tab=${id}]`)!.click();

function becomeVisible(): void {
  Object.defineProperty(document, "visibilityState", { value: "visible", configurable: true });
  document.dispatchEvent(new Event("visibilitychange"));
}

describe("the tabs before and without a snapshot", () => {
  it("say the dashboard is loading while /dashboard is being read", () => {
    startPage(masthead, sections, {
      dashboard: () => new Promise(() => {}),
      current: quietCurrent,
    });

    expect(panel("overview").textContent).toContain(LOADING);
    open("rain");
    expect(panel("rain").textContent).toContain(LOADING);
    expect(panel("rain").textContent).not.toContain("still being built");
  });

  it("say the dashboard could not be loaded, as the header does", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const page = startPage(masthead, sections, {
      dashboard: () => Promise.reject(new Error("502")),
      current: quietCurrent,
    });
    await page.booted;

    expect(masthead.textContent).toContain(FAILED);
    expect(panel("overview").textContent).toContain(FAILED);
    expect(panel("overview").textContent).not.toContain(LOADING);
    open("solar");
    expect(panel("solar").textContent).toContain(FAILED);
  });
});

describe("the refresh cycle", () => {
  it("reads /dashboard again every 5 minutes, with /current", async () => {
    const dashboard = vi.fn(() => Promise.resolve(snapshot()));
    const current = vi.fn(quietCurrent);
    await startPage(masthead, sections, { dashboard, current }).booted;
    expect(dashboard).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(CYCLE_MS);
    expect(dashboard).toHaveBeenCalledTimes(2);
    expect(current).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(CYCLE_MS);
    expect(dashboard).toHaveBeenCalledTimes(3);
  });

  it("reads /dashboard again when the page comes back into view", async () => {
    const dashboard = vi.fn(() => Promise.resolve(snapshot()));
    await startPage(masthead, sections, { dashboard, current: quietCurrent }).booted;

    becomeVisible();
    await vi.advanceTimersByTimeAsync(0);

    expect(dashboard).toHaveBeenCalledTimes(2);
  });

  it("redraws the page only when the snapshot's built_at changes", async () => {
    const first = snapshot();
    const builds = [first, snapshot(first.built_at_ms), snapshot(first.built_at_ms + 300_000)];
    const dashboard = vi.fn(() => Promise.resolve(builds.shift() ?? snapshot()));
    await startPage(masthead, sections, { dashboard, current: quietCurrent }).booted;
    const drawn = panel("overview").firstElementChild;
    expect(drawn).not.toBeNull();

    await vi.advanceTimersByTimeAsync(CYCLE_MS);
    expect(panel("overview").firstElementChild).toBe(drawn);

    await vi.advanceTimersByTimeAsync(CYCLE_MS);
    expect(panel("overview").firstElementChild).not.toBe(drawn);
    expect(panel("overview").firstElementChild).not.toBeNull();
  });

  it("keeps the Events tab's selection and Show more across a new snapshot", async () => {
    // No network: the picked event's curve read never answers.
    vi.stubGlobal("fetch", () => new Promise(() => {}));
    // Enough events past the golden's three to need Show more; no open curve,
    // so the live event draws no chart.
    const build = (builtAtMs: number) => {
      const doc = snapshot(builtAtMs);
      doc.rain_events.open_curve = null;
      const oldest = doc.rain_events.events.at(-1)!;
      for (let i = 1; i <= 30; i += 1) {
        const start = oldest.start_ms - i * 86_400_000;
        doc.rain_events.events.push({ ...oldest, start_ms: start, end_ms: start + 3_600_000 });
      }
      return doc;
    };
    const first = build(1);
    const builds = [first, build(2)];
    const dashboard = vi.fn(() => Promise.resolve(builds.shift() ?? build(2)));
    await startPage(masthead, sections, { dashboard, current: quietCurrent }).booted;
    open("events");
    const events = panel("events");
    const tap = (selector: string) =>
      events.querySelector<HTMLElement>(selector)!.dispatchEvent(
        new MouseEvent("click", { bubbles: true, cancelable: true }),
      );
    const picked = first.rain_events.events[1]!.start_ms;
    tap(".ev-more");
    tap(`.ev-open[data-start='${picked}']`);
    const shown = () => events.querySelectorAll(".ev-table tbody tr").length;
    expect(shown()).toBe(33);
    const table = events.querySelector(".ev-table");

    await vi.advanceTimersByTimeAsync(CYCLE_MS);

    expect(events.querySelector(".ev-table")).not.toBe(table); // redrawn
    expect(shown()).toBe(33);
    const current = events.querySelector<HTMLElement>("tr[aria-current=true] .ev-open")!;
    expect(Number(current.dataset.start)).toBe(picked);
  });

  it("keeps the page it has when a later read fails", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const dashboard = vi
      .fn<() => Promise<DashboardSnapshot>>()
      .mockResolvedValueOnce(snapshot())
      .mockRejectedValue(new Error("502"));
    await startPage(masthead, sections, { dashboard, current: quietCurrent }).booted;
    const drawn = panel("overview").firstElementChild;

    await vi.advanceTimersByTimeAsync(CYCLE_MS);

    expect(dashboard).toHaveBeenCalledTimes(2);
    expect(panel("overview").firstElementChild).toBe(drawn);
    expect(masthead.textContent).not.toContain(FAILED);
  });

  it("draws the page on a later cycle after the first read failed", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const dashboard = vi
      .fn<() => Promise<DashboardSnapshot>>()
      .mockRejectedValueOnce(new Error("502"))
      .mockResolvedValue(snapshot());
    await startPage(masthead, sections, { dashboard, current: quietCurrent }).booted;
    expect(panel("overview").textContent).toContain(FAILED);

    await vi.advanceTimersByTimeAsync(CYCLE_MS);

    expect(panel("overview").textContent).not.toContain(FAILED);
    expect(masthead.textContent).not.toContain(FAILED);
  });
});
