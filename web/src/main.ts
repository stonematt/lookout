/**
 * Boot the page: the header and the tab shell, fed by `/dashboard` and then
 * `/current`. The data age ticks every 15 seconds from the browser clock,
 * and `/current` is re-read every 5 minutes and whenever the page comes back
 * into view.
 */
import "@fontsource-variable/atkinson-hyperlegible-next/wght.css";
import "./styles/tokens.css";
import "./styles/base.css";

import { mountEvents } from "./events/tab.js";
import { mountHeader } from "./header.js";
import { fetchCurrent, fetchDashboard } from "./load.js";
import { renderOverview } from "./overview/index.js";
import { renderRain } from "./rain/index.js";
import { renderSolar } from "./solar/index.js";
import type { DashboardSnapshot } from "./snapshot.js";
import type { TabId } from "./tabs.js";
import { mountTabs } from "./tabs.js";
import { applyThemeOverride } from "./theme.js";

const TICK_MS = 15_000;
const CURRENT_EVERY_MS = 5 * 60_000;

applyThemeOverride(location.search);

const header = mountHeader(document.getElementById("masthead")!);
let dashboard: DashboardSnapshot | null = null;
const rendered = new WeakMap<HTMLElement, DashboardSnapshot>();

/** Render a tab's panel from the snapshot, once per snapshot. */
function renderTab(id: TabId, panel: HTMLElement): void {
  if (id === "rain" && dashboard) {
    if (rendered.get(panel) !== dashboard) renderRain(panel, dashboard);
    rendered.set(panel, dashboard);
    return;
  }
  if (id === "events" && dashboard) {
    if (rendered.get(panel) !== dashboard) mountEvents(panel, dashboard);
    rendered.set(panel, dashboard);
    return;
  }
  if (id === "solar" && dashboard) {
    if (rendered.get(panel) !== dashboard) renderSolar(panel, dashboard);
    rendered.set(panel, dashboard);
    return;
  }
  if (!panel.childElementCount) {
    panel.innerHTML =
      '<p class="standfirst">This section is still being built.</p>';
  }
}

const tabs = mountTabs(document.getElementById("sections")!, renderTab);

function title(name: string | null): void {
  document.title = name ? `${name} · Lookout` : "Lookout";
}

async function loadCurrent(): Promise<void> {
  try {
    const current = await fetchCurrent(location.search);
    header.show({ station: current.station, observedAtMs: current.observed_at_ms });
  } catch (error) {
    // The snapshot's own observation time stays on screen and keeps ageing.
    console.warn("Lookout: /current could not be read", error);
  }
}

async function boot(): Promise<void> {
  try {
    const snapshot = await fetchDashboard(location.search);
    header.show({ station: snapshot.station, observedAtMs: snapshot.observed_at_ms });
    title(snapshot.station.name);
    renderOverview(tabs.panel("overview"), snapshot);
    dashboard = snapshot;
    renderTab(tabs.current(), tabs.panel(tabs.current()));
  } catch (error) {
    console.error("Lookout: /dashboard could not be read", error);
    header.fail(() => void boot());
  }
  await loadCurrent();
}

void boot();
setInterval(() => header.tick(), TICK_MS);
setInterval(() => void loadCurrent(), CURRENT_EVERY_MS);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;
  header.tick();
  void loadCurrent();
});
