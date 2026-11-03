// @vitest-environment happy-dom
/**
 * The page as a visitor sees it: the header and the tabs, fed by the two
 * read functions (faked here), on the 5-minute refresh cycle. No network.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ReadError } from "../src/load.js";
import type { Page, PageReads } from "../src/page.js";
import { CYCLE_MS, startPage as start } from "../src/page.js";
import type { CurrentReading, DashboardSnapshot } from "../src/snapshot.js";
import { SCHEMA_VERSION, parseCurrent, parseDashboard } from "../src/snapshot.js";
import type { StationIndex } from "../src/stations.js";
import { parseStationIndex } from "../src/stations.js";
import { golden, goldenCurveText } from "./golden.js";

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
      events
        .querySelector<HTMLElement>(selector)!
        .dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
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

const FIXTURE_MAC = "00:11:22:33:44:55";
const SALMON_MAC = "00:11:22:33:44:66";
const SALMON_QUERY = "?station=00%3A11%3A22%3A33%3A44%3A66";

/** The golden station index: Fixture Station, then Salmon Run. */
function twoStations(): StationIndex {
  const path = resolve(import.meta.dirname, "../../tests/golden/stations/raining.stations.json");
  return parseStationIndex(JSON.parse(readFileSync(path, "utf8")));
}

/** The live inventory today: one station. */
function oneStation(): StationIndex {
  const index = twoStations();
  return { ...index, stations: index.stations.slice(0, 1) };
}

/** The page's `?station=`, as a read function would be asked it. */
const asked = () => new URLSearchParams(location.search).get("station") ?? FIXTURE_MAC;

/**
 * Salmon Run's snapshot. A WS-2902 lacks soil moisture, PM2.5 and the WH31E
 * sensors; the snapshot reads none of them (the refresh test
 * `test_a_station_without_soil_pm25_or_wh31e_builds_its_snapshot` builds
 * one without them), so its shape is the fixture station's.
 */
function salmon(): DashboardSnapshot {
  const doc = snapshot(1);
  doc.station = { mac: SALMON_MAC, name: "Salmon Run", tz: "America/Los_Angeles" };
  return doc;
}

/** A `/current` for the station asked, observed `observedAtMs`. */
function liveFor(mac: string, observedAtMs: number): CurrentReading {
  return parseCurrent({
    schema_version: SCHEMA_VERSION,
    kind: "current",
    station: { mac, name: mac === SALMON_MAC ? "Salmon Run" : "Fixture Station" },
    observed_at_ms: observedAtMs,
    built_at_ms: null,
    last_data: { dateutc: observedAtMs },
    context: null,
  });
}

/** Read functions that serve whichever station the page's URL asks for. */
function byStation(index: StationIndex = twoStations()) {
  return {
    stations: vi.fn(() => Promise.resolve(index)),
    dashboard: vi.fn(() => Promise.resolve(asked() === SALMON_MAC ? salmon() : snapshot())),
    current: quietCurrent,
  };
}

const picker = () => masthead.querySelector<HTMLSelectElement>("select.station-select");

function choose(mac: string): void {
  picker()!.value = mac;
  picker()!.dispatchEvent(new Event("change"));
}

describe("the station selector", () => {
  it("lists the station index in order, the default station selected", async () => {
    await startPage(masthead, sections, byStation()).booted;
    await vi.advanceTimersByTimeAsync(0);

    expect([...picker()!.options].map((o) => o.textContent)).toEqual([
      "Fixture Station",
      "Salmon Run",
    ]);
    expect(picker()!.value).toBe(FIXTURE_MAC);
    expect(masthead.querySelector(".masthead-archive")?.textContent).toBe(
      "Archive from May 4, 2022",
    );
  });

  it("is not drawn for a one-station index; the details strip still is", async () => {
    await startPage(masthead, sections, byStation(oneStation())).booted;
    await vi.advanceTimersByTimeAsync(0);

    expect(picker()).toBeNull();
    expect(masthead.querySelector("h1")?.textContent).toBe("Fixture Station");
    expect(masthead.querySelector(".masthead-reading .masthead-when")?.textContent).toBe(
      "Reading Jan 8, 4:25 pm",
    );
    expect(masthead.querySelector(".masthead-archive")?.textContent).toBe(
      "Archive from May 4, 2022",
    );
  });

  it("writes ?station= to the URL and redraws every tab for the chosen station", async () => {
    history.replaceState(null, "", "/?event=1#rain");
    const reads = byStation();
    await startPage(masthead, sections, reads).booted;
    await vi.advanceTimersByTimeAsync(0);
    const before = new Map(
      ["overview", "rain", "events", "solar"].map((id) => [id, panel(id).firstElementChild]),
    );

    choose(SALMON_MAC);
    await vi.advanceTimersByTimeAsync(0);

    // The other station's event is not carried over; the open tab is.
    expect(location.search).toBe(SALMON_QUERY);
    expect(location.hash).toBe("#rain");
    expect(reads.dashboard).toHaveBeenCalledTimes(2);
    expect(document.title).toBe("Salmon Run · Lookout");
    expect(picker()!.value).toBe(SALMON_MAC);
    expect(masthead.querySelector(".masthead-archive")?.textContent).toBe(
      "Archive start not known yet",
    );
    for (const id of ["overview", "rain", "events", "solar"]) {
      open(id);
      expect(panel(id).firstElementChild).not.toBe(before.get(id));
      expect(panel(id).textContent).not.toContain(LOADING);
    }
  });

  it("moves the live reading to the chosen station, dropping the last one's", async () => {
    const observed = snapshot().observed_at_ms!;
    vi.setSystemTime(observed + 3 * 60_000);
    // Fixture Station's /current answers only after Salmon Run was chosen.
    let late: (reading: CurrentReading) => void = () => undefined;
    const reads = {
      ...byStation(),
      current: vi.fn(() =>
        asked() === SALMON_MAC
          ? Promise.resolve(liveFor(SALMON_MAC, observed - 60 * 60_000))
          : new Promise<CurrentReading>((answer) => (late = answer)),
      ),
    };
    await startPage(masthead, sections, reads).booted;
    await vi.advanceTimersByTimeAsync(0);

    choose(SALMON_MAC);
    await vi.advanceTimersByTimeAsync(0);
    late(liveFor(FIXTURE_MAC, observed + 2 * 60_000));
    await vi.advanceTimersByTimeAsync(0);

    expect(reads.current).toHaveBeenCalledTimes(2);
    // Salmon Run's snapshot (Jan 8, 4:25 pm) is its newest; the fixture
    // station's fresher live reading is not shown for it.
    expect(masthead.querySelector(".masthead-reading .masthead-age")?.textContent).toBe("3 m ago");
    expect(masthead.textContent).not.toContain("1 m ago");
  });

  it("opens on the station a ?station= link names", async () => {
    history.replaceState(null, "", `/${SALMON_QUERY}`);
    const reads = byStation();
    await startPage(masthead, sections, reads).booted;
    await vi.advanceTimersByTimeAsync(0);

    expect(picker()!.value).toBe(SALMON_MAC);
    expect(document.title).toBe("Salmon Run · Lookout");
  });

  it("follows the back button to the station it returns to", async () => {
    const reads = byStation();
    await startPage(masthead, sections, reads).booted;
    await vi.advanceTimersByTimeAsync(0);
    choose(SALMON_MAC);
    await vi.advanceTimersByTimeAsync(0);

    history.replaceState(null, "", "/");
    window.dispatchEvent(new PopStateEvent("popstate"));
    await vi.advanceTimersByTimeAsync(0);

    expect(picker()!.value).toBe(FIXTURE_MAC);
    expect(document.title).toBe("Fixture Station · Lookout");
    expect(reads.dashboard).toHaveBeenCalledTimes(3);
  });

  it("says a station listed with no snapshot yet has none, not the last station's page", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const reads = {
      ...byStation(),
      dashboard: vi.fn(() =>
        asked() === SALMON_MAC
          ? Promise.reject(new ReadError("/dashboard", 404))
          : Promise.resolve(snapshot()),
      ),
    };
    await startPage(masthead, sections, reads).booted;
    await vi.advanceTimersByTimeAsync(0);

    choose(SALMON_MAC);
    await vi.advanceTimersByTimeAsync(0);

    const missing = "Salmon Run has no dashboard yet";
    expect(panel("overview").textContent).toContain(missing);
    expect(masthead.textContent).toContain(missing);
    expect(masthead.querySelector(".masthead-retry")).not.toBeNull();
    open("solar");
    expect(panel("solar").textContent).toContain(missing);
  });

  it("draws every tab for a station that lacks soil, PM2.5 and WH31E sensors", async () => {
    history.replaceState(null, "", `/${SALMON_QUERY}`);
    const errors = vi.spyOn(console, "error");
    await startPage(masthead, sections, byStation()).booted;

    for (const id of ["overview", "rain", "events", "solar"]) {
      open(id);
      expect(panel(id).textContent).not.toContain(LOADING);
      expect(panel(id).textContent).not.toContain(FAILED);
      expect(panel(id).querySelector("h2")).not.toBeNull();
    }
    expect(errors).not.toHaveBeenCalled();
  });

  it("reads a chosen station's event curve again, not the last station's", async () => {
    // Two stations near each other can catch one storm on the same 5-minute
    // reading: the same start_ms is a different event at each.
    const start = 1767758400000;
    const urls: string[] = [];
    vi.stubGlobal("fetch", (url: string) => {
      urls.push(url);
      return Promise.resolve(new Response(goldenCurveText("raining", start)));
    });
    history.replaceState(null, "", `/?event=${start}`);
    await startPage(masthead, sections, byStation()).booted;
    open("events");
    await vi.advanceTimersByTimeAsync(0);
    expect(urls).toContain(`/event?start=${start}`);

    choose(SALMON_MAC);
    await vi.advanceTimersByTimeAsync(0);
    const events = panel("events");
    events
      .querySelector<HTMLElement>(`.ev-open[data-start='${start}']`)!
      .dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    await vi.advanceTimersByTimeAsync(0);

    // (Pages left by earlier tests also hear the tab's hash change; only
    // reads for the chosen station count here.)
    expect(urls.filter((url) => url.includes("station="))).toEqual([
      `/event?start=${start}&station=00%3A11%3A22%3A33%3A44%3A66`,
    ]);
  });

  it("keeps the default station's page when the index cannot be read", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const reads = { ...byStation(), stations: () => Promise.reject(new Error("502")) };
    await startPage(masthead, sections, reads).booted;
    await vi.advanceTimersByTimeAsync(0);

    expect(picker()).toBeNull();
    expect(masthead.querySelector("h1")?.textContent).toBe("Fixture Station");
    expect(panel("overview").textContent).not.toContain(FAILED);
  });
});
