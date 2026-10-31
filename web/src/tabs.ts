/**
 * The four sticky tabs: Overview, Rain, Rain Events, Solar.
 *
 * A WAI-ARIA tablist with a roving tabindex: only the open tab is a tab
 * stop; ArrowLeft and ArrowRight move (and wrap), Home and End jump to the
 * ends. Moving opens the tab. The open tab is tracked in the URL hash, so a
 * link or bookmark opens the same tab, and a hash change (back button, a
 * pasted link) opens the tab it names.
 */

/** One tab: its id is its URL hash. */
export interface TabSpec {
  id: TabId;
  label: string;
}

/** The tab ids, in order. */
export type TabId = "overview" | "rain" | "events" | "solar";

/** The tabs, in order. */
export const TABS: readonly TabSpec[] = [
  { id: "overview", label: "Overview" },
  { id: "rain", label: "Rain" },
  { id: "events", label: "Rain Events" },
  { id: "solar", label: "Solar" },
];

/** Mounted tabs. */
export interface Tabs {
  /** Open a tab; an unknown id opens Overview. */
  select(id: string, focus?: boolean): void;
  /** A tab's panel, to render that tab's content into. */
  panel(id: TabId): HTMLElement;
  /** The open tab. */
  current(): TabId;
}

function isTabId(id: string): id is TabId {
  return TABS.some((tab) => tab.id === id);
}

/**
 * Render the tab bar and its four empty panels into `host`.
 *
 * @param host - The element to render into; its content is replaced.
 * @param onSelect - Called with the tab and its panel each time one opens,
 *   so a tab can render its content the first time it is shown.
 * @returns The tabs' controls.
 */
export function mountTabs(
  host: HTMLElement,
  onSelect?: (id: TabId, panel: HTMLElement) => void,
): Tabs {
  const bar = document.createElement("nav");
  bar.className = "tabs";
  bar.setAttribute("aria-label", "Dashboard sections");
  const list = document.createElement("div");
  list.className = "tabs-list";
  list.setAttribute("role", "tablist");
  bar.append(list);

  const buttons = new Map<TabId, HTMLButtonElement>();
  const panels = new Map<TabId, HTMLElement>();
  const panelHost = document.createElement("div");
  panelHost.className = "panels";

  for (const { id, label } of TABS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tab";
    button.id = `tab-${id}`;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", `panel-${id}`);
    button.dataset.tab = id;
    button.textContent = label;
    list.append(button);
    buttons.set(id, button);

    const panel = document.createElement("section");
    panel.className = "panel";
    panel.id = `panel-${id}`;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `tab-${id}`);
    panel.tabIndex = 0;
    panelHost.append(panel);
    panels.set(id, panel);
  }
  host.replaceChildren(bar, panelHost);

  let open: TabId = "overview";

  function select(id: string, focus = false): void {
    open = isTabId(id) ? id : "overview";
    for (const { id: tab } of TABS) {
      const on = tab === open;
      const button = buttons.get(tab)!;
      button.setAttribute("aria-selected", String(on));
      button.tabIndex = on ? 0 : -1;
      panels.get(tab)!.hidden = !on;
    }
    if (focus) buttons.get(open)!.focus();
    if (location.hash !== `#${open}`) {
      history.replaceState(history.state, "", `#${open}`);
    }
    onSelect?.(open, panels.get(open)!);
  }

  list.addEventListener("click", (event) => {
    const button = (event.target as Element).closest<HTMLElement>("[role=tab]");
    if (button?.dataset.tab) select(button.dataset.tab);
  });

  list.addEventListener("keydown", (event) => {
    const index = TABS.findIndex((tab) => tab.id === open);
    const last = TABS.length - 1;
    const next: Record<string, number> = {
      ArrowRight: index === last ? 0 : index + 1,
      ArrowLeft: index === 0 ? last : index - 1,
      Home: 0,
      End: last,
    };
    const target = next[event.key];
    if (target === undefined) return;
    event.preventDefault();
    select(TABS[target]!.id, true);
  });

  window.addEventListener("hashchange", () => select(location.hash.slice(1)));
  select(location.hash.slice(1));

  return {
    select,
    panel: (id) => panels.get(id)!,
    current: () => open,
  };
}
