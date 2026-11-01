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
 */
import { mountEvents } from "./events/tab.js";
import { setStationZone } from "./format.js";
import { mountHeader } from "./header.js";
import { renderOverview } from "./overview/index.js";
import { renderRain } from "./rain/index.js";
import type { CurrentReading, DashboardSnapshot } from "./snapshot.js";
import { renderSolar } from "./solar/index.js";
import type { TabId } from "./tabs.js";
import { mountTabs } from "./tabs.js";

/** How often both read functions are read again. */
export const CYCLE_MS = 5 * 60_000;

/** How often the data age ticks. */
const TICK_MS = 15_000;

/** The two read functions, as the page calls them. */
export interface PageReads {
  dashboard: () => Promise<DashboardSnapshot>;
  current: () => Promise<CurrentReading>;
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
 * Start the page: mount the header and tabs, read both functions, and keep
 * reading them on the refresh cycle.
 *
 * @param masthead - The header's host.
 * @param sections - The tabs' host.
 * @param reads - The read functions.
 * @returns The started page.
 */
export function startPage(
  masthead: HTMLElement,
  sections: HTMLElement,
  reads: PageReads,
): Page {
  const header = mountHeader(masthead);
  let dashboard: DashboardSnapshot | null = null;
  let state: keyof typeof STATE_TEXT = "loading";
  const rendered = new WeakMap<HTMLElement, DashboardSnapshot>();

  /** Draw a tab's panel from the snapshot, once per snapshot. */
  function renderTab(id: TabId, panel: HTMLElement): void {
    if (!dashboard) {
      const note = document.createElement("p");
      note.className = "standfirst panel-state";
      note.setAttribute("role", "status");
      note.textContent = STATE_TEXT[state];
      panel.replaceChildren(note);
      return;
    }
    if (rendered.get(panel) === dashboard) return;
    RENDER[id](panel, dashboard);
    rendered.set(panel, dashboard);
  }

  const tabs = mountTabs(sections, renderTab);

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
    try {
      const snapshot = await reads.dashboard();
      if (dashboard && snapshot.built_at_ms === dashboard.built_at_ms) return;
      show(snapshot);
    } catch (error) {
      if (dashboard) {
        // The page already drawn stays; its figures age on their own clock.
        console.warn("Lookout: /dashboard could not be read again", error);
        return;
      }
      console.error("Lookout: /dashboard could not be read", error);
      state = "failed";
      header.fail(() => void refresh());
      renderTab(tabs.current(), tabs.panel(tabs.current()));
      if (tabs.current() !== "overview") renderTab("overview", tabs.panel("overview"));
    }
  }

  async function readCurrent(): Promise<void> {
    try {
      header.showCurrent(await reads.current());
    } catch (error) {
      // The snapshot's own observation time stays on screen and keeps ageing,
      // stale on its own clock.
      console.warn("Lookout: /current could not be read", error);
    }
  }

  let running: Promise<void> | null = null;

  /** Read both functions, unless a read of both is already under way. */
  function refresh(): Promise<void> {
    running ??= (async () => {
      await readDashboard();
      void readCurrent();
    })().finally(() => {
      running = null;
    });
    return running;
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
  return {
    booted,
    stop() {
      clearInterval(ticking);
      clearInterval(cycling);
      listening.abort();
    },
  };
}
