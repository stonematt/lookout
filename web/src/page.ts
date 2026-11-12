/**
 * The page: the header and the tab shell, fed by `/dashboard` and then
 * `/current`, on one refresh cycle.
 *
 * Both are read at boot, every 5 minutes and whenever the page comes back
 * into view. A new snapshot (a different `built_at_ms`) redraws the
 * Overview and the open tab, and the other tabs when next opened; the same
 * snapshot again redraws nothing. `/current` only moves the header's reading
 * forward; the page's figures, and their own age, stay the snapshot's. The
 * data age ticks every 15 seconds from the browser clock.
 *
 * Until the first snapshot arrives a tab says it is loading; when it cannot
 * be read, a tab says so in the header's words. A later read that fails
 * keeps the page already drawn.
 *
 * The page layout switch (`layout.ts`) runs from the start, setting the
 * root's `data-layout` (phone, two- or four-column) that every page layout
 * rule keys off; the tab bar sits just before the tabs' host, so on both
 * grids it shares the header's row, and on a phone it is the bottom tab bar
 * (CSS places it). A phone draws each tab as a fixed top and finding rows,
 * the grids as tiles (`tiles.ts`), so crossing 760px either way redraws
 * the tabs drawn.
 *
 * The station index (`/stations`, read on the same cycle) feeds the header's
 * station selector (#273). Choosing a station writes `?station=<MAC>` to the
 * URL (a new history entry, so Back returns to the last station), and the
 * page starts over for it: every tab and the header's reading wait for the
 * new station's reads, and a read still under way for the last station is
 * dropped when it answers, so the page never mixes two stations. Every read
 * function takes the station from the page's URL.
 *
 * The tab shell keeps a zoom's route (`#overview/<slug>`, #474) in the URL,
 * and each time a tab opens the Overview's zoom shell follows the route
 * (`overview/zoom.ts`), opening or closing a zoom.
 */
import { mountEvents } from "./events/tab.js";
import { setStationZone } from "./format.js";
import { mountHeader } from "./header.js";
import { mountLayout } from "./layout.js";
import { ReadError } from "./load.js";
import { renderOverview } from "./overview/index.js";
import { drawZoom, followZoom, isZoomRoute } from "./overview/zoom.js";
import { renderRain } from "./rain/index.js";
import type { CurrentReading, DashboardSnapshot } from "./snapshot.js";
import { renderSolar } from "./solar/index.js";
import type { StationIndex } from "./stations.js";
import { shownStation } from "./stations.js";
import type { TabId } from "./tabs.js";
import { TABS, mountTabs } from "./tabs.js";

/** How often both read functions are read again. */
export const CYCLE_MS = 5 * 60_000;

/** How often the data age ticks. */
const TICK_MS = 15_000;

/**
 * The read functions, as the page calls them. Each reads the station the
 * page's URL names at the time of the call.
 */
export interface PageReads {
  dashboard: () => Promise<DashboardSnapshot>;
  current: () => Promise<CurrentReading>;
  /** The station index; without it the page shows one station, no selector. */
  stations?: () => Promise<StationIndex>;
}

/** A started page. */
export interface Page {
  /** Settles once the first `/dashboard` read has drawn or failed. */
  booted: Promise<void>;
  /** Stop the refresh cycle and the age tick. */
  stop(): void;
}

/** Each tab's renderer: replaces the panel's content from the snapshot. */
const RENDER: Record<TabId, (panel: HTMLElement, snapshot: DashboardSnapshot) => void> = {
  overview: (panel, snapshot) => renderOverview(panel, snapshot),
  rain: renderRain,
  events: (panel, snapshot) => mountEvents(panel, snapshot),
  solar: (panel, snapshot) => renderSolar(panel, snapshot),
};

/** What a tab says while it has no snapshot to draw. */
const STATE_TEXT = {
  loading: "Loading the dashboard…",
  // The header's own words (header.ts `fail`).
  failed: "The dashboard could not be loaded.",
} as const;

/**
 * What the page says for a station the index lists whose snapshot is not
 * there (`/dashboard` answers 404): the index lists only stations that have
 * published, so the snapshot is lost or not yet written. The words are
 * neutral about why; the header says it short, the tabs add a retry.
 */
function missingText(name: string | null): { header: string; tab: string } {
  const header = `${name ?? "This station"} has no dashboard right now.`;
  return { header, tab: `${header} Try again shortly.` };
}

/** The station the page's URL names, or null for the default station. */
function askedStation(): string | null {
  return new URLSearchParams(location.search).get("station");
}

/**
 * Start the page: mount the header and tabs, read the functions, and keep
 * reading them on the refresh cycle.
 *
 * @param masthead - The header's host.
 * @param sections - The tabs' host.
 * @param reads - The read functions.
 * @returns The started page.
 */
export function startPage(masthead: HTMLElement, sections: HTMLElement, reads: PageReads): Page {
  // Every page layout rule keys off the attribute this sets (`layout.ts`).
  let dashboard: DashboardSnapshot | null = null;
  let rendered = new WeakMap<HTMLElement, DashboardSnapshot>();
  let wasPhone: boolean;
  const layout = mountLayout(undefined, undefined, (next) => {
    // A phone draws rows, the grids tiles: redraw only across that line.
    const phone = next === "phone";
    if (phone === wasPhone) {
      // Two and four columns break the rows differently: the zoom moves.
      if (dashboard) drawZoom(tabs.panel("overview"), dashboard);
      return;
    }
    wasPhone = phone;
    rendered = new WeakMap();
    if (!dashboard) return;
    renderTab("overview", tabs.panel("overview"));
    renderTab(tabs.current(), tabs.panel(tabs.current()));
  });
  wasPhone = layout.layout() === "phone";
  const header = mountHeader(masthead);
  let stateText: string = STATE_TEXT.loading;
  let index: StationIndex | null = null;
  /** The station shown, as the URL names it (null: the default station). */
  let showing = askedStation();
  /** Counts station choices: a read begun for an earlier one is dropped. */
  let generation = 0;
  /** The read of every function under way, if any. */
  let running: Promise<void> | null = null;
  /** The station index read under way, if any. */
  let indexRead: Promise<void> = Promise.resolve();

  /** Draw a tab's panel from the snapshot, once per snapshot. */
  function renderTab(id: TabId, panel: HTMLElement): void {
    if (!dashboard) {
      const note = document.createElement("p");
      note.className = "standfirst panel-state";
      note.setAttribute("role", "status");
      note.textContent = stateText;
      panel.replaceChildren(note);
      return;
    }
    if (rendered.get(panel) === dashboard) return;
    RENDER[id](panel, dashboard);
    rendered.set(panel, dashboard);
  }

  const tabs = mountTabs(
    sections,
    (id, panel) => {
      renderTab(id, panel);
      // Every hash change opens a tab: the Overview's zoom follows the route.
      const overview = sections.querySelector<HTMLElement>("#panel-overview");
      if (overview) followZoom(overview);
    },
    { barBeforeHost: true, keeps: isZoomRoute },
  );

  /** The index's entry for the station shown, when the index is known. */
  function shownEntry() {
    return index ? shownStation(index, showing) : undefined;
  }

  /** Show the station index in the header, for the station shown. */
  function showIndex(): void {
    if (!index) return;
    const entry = shownEntry();
    // Before its snapshot names the zone, the index names the same one.
    if (entry && !dashboard) setStationZone(entry.tz);
    header.showStations(index, showing, choose);
  }

  /** A visitor chose a station: put it in the URL and start over for it. */
  function choose(mac: string): void {
    const url = new URL(location.href);
    url.searchParams.set("station", mac);
    // An event is one station's: the chosen station's list opens on its newest.
    url.searchParams.delete("event");
    history.pushState(null, "", `${url.pathname}${url.search}${url.hash}`);
    startOver();
  }

  /** Drop the station shown and read the one the URL now names. */
  function startOver(): void {
    showing = askedStation();
    generation += 1;
    running = null;
    dashboard = null;
    stateText = STATE_TEXT.loading;
    document.title = "Lookout";
    header.reset();
    showIndex();
    for (const { id } of TABS) renderTab(id, tabs.panel(id));
    void refresh();
  }

  function show(snapshot: DashboardSnapshot): void {
    // Every station-local label is in the zone this snapshot names (ADR-0005).
    setStationZone(snapshot.station.tz);
    header.showSnapshot(snapshot);
    document.title = snapshot.station.name ? `${snapshot.station.name} · Lookout` : "Lookout";
    dashboard = snapshot;
    renderTab("overview", tabs.panel("overview"));
    renderTab(tabs.current(), tabs.panel(tabs.current()));
  }

  async function readDashboard(): Promise<void> {
    const asked = generation;
    try {
      const snapshot = await reads.dashboard();
      if (asked !== generation) return;
      if (dashboard && snapshot.built_at_ms === dashboard.built_at_ms) return;
      show(snapshot);
    } catch (error) {
      if (asked !== generation) return;
      if (dashboard) {
        // The page already drawn stays; its figures age on their own clock.
        console.warn("Lookout: /dashboard could not be read again", error);
        return;
      }
      console.error("Lookout: /dashboard could not be read", error);
      const unbuilt = error instanceof ReadError && error.status === 404;
      // Whether the index lists the station decides the words: wait for it.
      if (unbuilt && !index) await indexRead;
      if (asked !== generation) return;
      const listed = unbuilt ? shownEntry() : undefined;
      const missing = listed ? missingText(listed.name) : null;
      stateText = missing?.tab ?? STATE_TEXT.failed;
      header.fail(() => void refresh(), missing?.header);
      renderTab(tabs.current(), tabs.panel(tabs.current()));
      if (tabs.current() !== "overview") renderTab("overview", tabs.panel("overview"));
    }
  }

  async function readStations(): Promise<void> {
    if (!reads.stations) return;
    try {
      index = await reads.stations();
      showIndex();
    } catch (error) {
      // Without the index the page still shows the station its URL names.
      console.warn("Lookout: /stations could not be read", error);
    }
  }

  async function readCurrent(): Promise<void> {
    const asked = generation;
    try {
      const current = await reads.current();
      if (asked === generation) header.showCurrent(current);
    } catch (error) {
      // The snapshot's own observation time stays on screen and keeps ageing,
      // stale on its own clock.
      console.warn("Lookout: /current could not be read", error);
    }
  }

  /** Read both functions, unless a read of both is already under way. */
  function refresh(): Promise<void> {
    if (running) return running;
    const asked = generation;
    indexRead = readStations();
    const mine: Promise<void> = (async () => {
      await readDashboard();
      // A station chosen meanwhile has its own refresh and its own /current.
      if (asked === generation) void readCurrent();
      await indexRead;
    })().finally(() => {
      if (running === mine) running = null;
    });
    running = mine;
    return mine;
  }

  const booted = refresh();
  const ticking = setInterval(() => header.tick(), TICK_MS);
  const cycling = setInterval(() => void refresh(), CYCLE_MS);
  const listening = new AbortController();
  document.addEventListener(
    "visibilitychange",
    () => {
      if (document.visibilityState !== "visible") return;
      header.tick();
      void refresh();
    },
    { signal: listening.signal },
  );
  window.addEventListener(
    "popstate",
    () => {
      // Back or Forward to another station; a tab's own hash is not one.
      if (askedStation() !== showing) startOver();
    },
    { signal: listening.signal },
  );
  return {
    booted,
    stop() {
      layout.stop();
      clearInterval(ticking);
      clearInterval(cycling);
      listening.abort();
    },
  };
}
