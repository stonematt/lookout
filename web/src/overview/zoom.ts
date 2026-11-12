/**
 * The zoom shell (#474, spec #460 stories 33–35, 44–45, 54–57, 59): a
 * mosaic tile's full view at its own URL, `#overview/<slug>`.
 *
 * A zoom is a definition, not markup: `ZoomDefinition` names its slug (the
 * mosaic tile's), its heading, its lead (the number and delta) and its
 * panels, each a heading and a `render` that fills it from the snapshot.
 * The definitions are registered by slug in `zooms/index.ts`, one file per
 * zoom under `zooms/`. This shell owns everything else: the route, where
 * the zoom sits, its heading, Close and Back, focus, `aria-expanded` and
 * `aria-controls`, Escape and motion. A zoom's content never touches any of
 * them.
 *
 * The route. `#overview/<slug>` opens the zoom for a registered slug whose
 * tile is drawn; any other route closes it. The tab shell keeps such a hash
 * (`TabsOptions.keeps`, `isZoomRoute`) and opens the Overview; a slug with
 * no zoom yet falls back to `#overview`, as before #474, so its tile's link
 * still lands on the Overview and never breaks. A shared zoom URL opens the
 * zoom directly.
 *
 * Where it sits, by the page layout (`layout.ts`):
 *
 * - phone: full screen. The zoom takes the Overview's place (the mosaic,
 *   the sources footer and the bottom tab bar step aside; the header stays,
 *   with its one freshness line), with "Overview" at its top to go back.
 * - two and four columns: in place, a full-width row under the tapped
 *   tile's row (under the feature block for a tile beside it), its panels
 *   two by two, with Close. Each tile with a zoom carries `aria-expanded`
 *   and `aria-controls`.
 *
 * Focus moves to the zoom's heading when it opens, and back to the link
 * that opened it (else its tile) when it closes. Close, Back and Escape
 * close it: they step back through the history entries the zooms pushed
 * (so the browser's Back and the zoom's own agree), or, for a zoom opened
 * from a shared link, replace the route with `#overview`. Under
 * `prefers-reduced-motion` the zoom opens with no animation and scrolls
 * into view without smoothing.
 *
 * The shell's markup and CSS hooks (`zoom.css`):
 *
 * - `section.zoom#zoom-<slug>[data-zoom=<slug>][data-mode=screen|row]`,
 *   named by its heading;
 * - `.zoom-bar` with `button.zoom-back` (phone) or `button.zoom-close`
 *   (grids);
 * - `.zoom-head`: `h2.zoom-heading` (focus target, `tabindex=-1`),
 *   `p.zoom-value` with `span.zoom-unit`, `p.zoom-delta`;
 * - `.zoom-panels`, the two-by-two grid on the grids, one column on a
 *   phone, of `section.zoom-panel[data-panel=<id>]`, each an
 *   `h3.zoom-panel-title` then the panel's content; a `wide` panel
 *   (`[data-wide]`) takes the whole row. Headings inside a panel are
 *   `h4` (a shared section's finding, `SectionOptions.zoom` in
 *   `tiles.ts`), so the zoom's headings nest.
 *
 * Offers `ZoomDefinition`, `ZoomPanel`, `zoomFor`, `isZoomRoute`,
 * `drawZoom` (the Overview calls it after each render) and `followZoom`
 * (the page calls it each time a tab opens or the hash moves).
 */
import "./zoom.css";

import { spoken } from "../dom.js";
import { currentLayout } from "../layout.js";
import type { DashboardSnapshot } from "../snapshot.js";
import type { TileReading } from "./mosaic.js";
import { ZOOMS } from "./zooms/index.js";

/** One panel of a zoom: a heading and what sits under it. */
export interface ZoomPanel {
  /** Unique in its zoom: the panel's `data-panel`. */
  id: string;
  /** Its heading, an `h3`: "Now against this date". */
  title: string;
  /**
   * Fill the panel under its heading. The panel is in the page when this
   * runs, so a chart can measure its width.
   */
  render(host: HTMLElement, snapshot: DashboardSnapshot): void;
  /** Take both columns of the two-by-two; one column by default. */
  wide?: boolean;
  /** Whether the snapshot holds what the panel draws; shown by default. */
  shown?(snapshot: DashboardSnapshot): boolean;
}

/** One zoom, declared once in its own file under `zooms/`. */
export interface ZoomDefinition {
  /** The mosaic tile's slug: the zoom opens at `#overview/<slug>`. */
  slug: string;
  /** The zoom's heading, an `h2`: "Temperature". */
  title: string;
  /** The number and delta under the heading; the tile's `read` fits. */
  lead?(snapshot: DashboardSnapshot): TileReading;
  /** The panels, in reading order: two by two on the grids. */
  panels: readonly ZoomPanel[];
}

/** The route prefix a zoom's hash starts with. */
const PREFIX = "overview/";

/**
 * The registered zoom a hash names.
 *
 * @param hash - A URL hash, with or without its `#`.
 * @param zooms - The registry.
 * @returns The zoom, or null for any other hash.
 */
export function zoomFor(
  hash: string,
  zooms: readonly ZoomDefinition[] = ZOOMS,
): ZoomDefinition | null {
  const route = hash.replace(/^#/, "");
  if (!route.startsWith(PREFIX)) return null;
  const slug = route.slice(PREFIX.length);
  return zooms.find((zoom) => zoom.slug === slug) ?? null;
}

/**
 * Whether a hash is a zoom route the tab shell should keep, not rewrite.
 *
 * @param hash - A URL hash, with or without its `#`.
 * @returns True for `overview/<slug>` with a registered zoom.
 */
export function isZoomRoute(hash: string): boolean {
  return zoomFor(hash) !== null;
}

/** What one Overview panel's zoom shell knows. */
interface ZoomState {
  snapshot: DashboardSnapshot | null;
  zooms: readonly ZoomDefinition[];
  /** The slug open now, or null. */
  open: string | null;
  /** The link that opened the zoom, to return focus to. */
  origin: HTMLElement | null;
  /** History entries back to the closed Overview; 0 when there is none. */
  back: number;
  /** Whether the route has been followed once (the page's first view). */
  started: boolean;
  /** The page's scroll before a phone zoom took the screen. */
  scrollY: number;
}

const states = new WeakMap<HTMLElement, ZoomState>();

/** Whether the visitor asked for reduced motion. */
function reducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** The history entry's own count of entries back to the closed Overview. */
function stampedBack(): number | null {
  const state: unknown = history.state;
  if (state && typeof state === "object" && "zoomBack" in state) {
    const back = (state as { zoomBack: unknown }).zoomBack;
    if (typeof back === "number") return back;
  }
  return null;
}

/** Record on this history entry how far back the closed Overview is. */
function stampBack(back: number): void {
  const state: unknown = history.state;
  const base = state && typeof state === "object" ? state : {};
  history.replaceState({ ...base, zoomBack: back }, "");
}

/** The shell's state for a panel, made the first time it is asked for. */
function stateOf(panel: HTMLElement, zooms: readonly ZoomDefinition[]): ZoomState {
  let state = states.get(panel);
  if (!state) {
    state = {
      snapshot: null,
      zooms,
      open: null,
      origin: null,
      back: 0,
      started: false,
      scrollY: 0,
    };
    states.set(panel, state);
    listen(panel, state);
  }
  state.zooms = zooms;
  return state;
}

/** A tile's own link, by slug. */
function tileLink(panel: HTMLElement, slug: string): HTMLElement | null {
  return panel.querySelector<HTMLElement>(`a.mosaic-link[data-zoom="${CSS.escape(slug)}"]`);
}

/** The panel's listeners: the origin of a zoom, the tile toggle, Escape. */
function listen(panel: HTMLElement, state: ZoomState): void {
  panel.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const link = target.closest<HTMLAnchorElement>(`a[href^="#${PREFIX}"]`);
    if (!link) return;
    const zoom = zoomFor(link.getAttribute("href") ?? "", state.zooms);
    if (!zoom) return;
    if (zoom.slug === state.open && link.closest(".zoom") === null) {
      // The open zoom's own tile closes it again (aria-expanded true).
      event.preventDefault();
      closeZoom(panel);
      return;
    }
    state.origin = link;
  });
  panel.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || state.open === null) return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    // From inside the zoom, or from the tile that controls it.
    const control = target.closest(`a[href="#${PREFIX}${state.open}"]`);
    if (!target.closest(".zoom") && !control) return;
    event.preventDefault();
    closeZoom(panel);
  });
}

/**
 * Close the open zoom, as Close, Back and Escape do: back through the
 * entries the zooms pushed, or, opened from a shared link, by replacing the
 * route with `#overview`.
 *
 * @param panel - The Overview's panel.
 */
function closeZoom(panel: HTMLElement): void {
  const state = states.get(panel);
  if (!state || state.open === null) return;
  // The entry's own stamp, not a count kept here: an entry another part of
  // the page pushed (a station chosen) carries none, and is replaced.
  const back = stampedBack() ?? 0;
  if (back > 0) {
    history.go(-back);
    return;
  }
  history.replaceState(unstamped(), "", `#overview`);
  followZoom(panel);
}

/** This entry's state without the zoom's stamp, for a route that is closed. */
function unstamped(): unknown {
  const state: unknown = history.state;
  if (!state || typeof state !== "object" || !("zoomBack" in state)) return state;
  const { zoomBack: _, ...rest } = state as Record<string, unknown>;
  return Object.keys(rest).length ? rest : null;
}

/** Whether a grid layout draws the zoom in place. */
const inPlace = () => currentLayout() !== "phone";

/** The zoom's bar: Back to the Overview on a phone, Close on the grids. */
function bar(panel: HTMLElement): HTMLElement {
  const el = document.createElement("div");
  el.className = "zoom-bar";
  const button = document.createElement("button");
  button.type = "button";
  if (inPlace()) {
    button.className = "zoom-close";
    button.append("Close", chevron("×"));
  } else {
    button.className = "zoom-back";
    button.append(chevron("‹"), spoken("Back to the "), "Overview");
  }
  button.addEventListener("click", () => closeZoom(panel));
  el.append(button);
  return el;
}

/** A glyph beside a button's words, hidden from screen readers. */
function chevron(glyph: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.className = "zoom-glyph";
  span.setAttribute("aria-hidden", "true");
  span.textContent = glyph;
  return span;
}

/** The zoom's head: its heading, then the number and delta. */
function head(zoom: ZoomDefinition, snapshot: DashboardSnapshot): HTMLElement {
  const el = document.createElement("header");
  el.className = "zoom-head";
  const heading = document.createElement("h2");
  heading.className = "zoom-heading";
  heading.id = `zoom-${zoom.slug}-heading`;
  heading.tabIndex = -1;
  heading.textContent = zoom.title;
  el.append(heading);
  const reading = zoom.lead?.(snapshot);
  if (reading?.value != null) {
    const value = document.createElement("p");
    value.className = "zoom-value";
    value.dataset.part = "value";
    value.append(reading.value);
    if (reading.unit) {
      const unit = document.createElement("span");
      unit.className = "zoom-unit";
      unit.textContent = `${String.fromCharCode(160)}${reading.unit}`;
      value.append(unit);
    }
    el.append(value);
  }
  if (reading?.delta) {
    const delta = document.createElement("p");
    delta.className = "zoom-delta";
    delta.dataset.part = "delta";
    delta.textContent = reading.delta;
    el.append(delta);
  }
  return el;
}

/** The zoom's element, empty of panels until it is in the page. */
function shell(panel: HTMLElement, zoom: ZoomDefinition, snapshot: DashboardSnapshot): HTMLElement {
  const el = document.createElement("section");
  el.className = "zoom";
  el.id = `zoom-${zoom.slug}`;
  el.dataset.zoom = zoom.slug;
  el.dataset.mode = inPlace() ? "row" : "screen";
  el.setAttribute("aria-labelledby", `zoom-${zoom.slug}-heading`);
  const panels = document.createElement("div");
  panels.className = "zoom-panels";
  el.append(bar(panel), head(zoom, snapshot), panels);
  return el;
}

/** Fill the zoom's panels, now that it is in the page and has a width. */
function fill(el: HTMLElement, zoom: ZoomDefinition, snapshot: DashboardSnapshot): void {
  const panels = el.querySelector<HTMLElement>(".zoom-panels")!;
  for (const spec of zoom.panels) {
    if (spec.shown && !spec.shown(snapshot)) continue;
    const host = document.createElement("section");
    host.className = "zoom-panel";
    host.dataset.panel = spec.id;
    if (spec.wide) host.dataset.wide = "true";
    const title = document.createElement("h3");
    title.className = "zoom-panel-title";
    title.textContent = spec.title;
    host.append(title);
    panels.append(host);
    spec.render(host, snapshot);
  }
}

/**
 * Put an in-place zoom under the tapped tile's row: after the last tile
 * that starts above the row's foot, where the row runs as deep as any tile
 * beside it (the feature's two rows, for a tile in its 2×2). With no
 * layout to measure (a test), it goes straight after the tile.
 */
function place(grid: HTMLElement, tile: HTMLElement, el: HTMLElement): void {
  el.remove();
  const tiles = [...grid.children].filter((child): child is HTMLElement => child !== el);
  const top = tile.getBoundingClientRect().top;
  let foot = tile.getBoundingClientRect().bottom;
  for (let grew = true; grew;) {
    grew = false;
    for (const other of tiles) {
      const box = other.getBoundingClientRect();
      if (box.top < foot - 1 && box.bottom > top + 1 && box.bottom > foot + 1) {
        foot = box.bottom;
        grew = true;
      }
    }
  }
  let after = tile;
  for (const other of tiles.slice(tiles.indexOf(tile) + 1)) {
    if (other.getBoundingClientRect().top < foot - 1) after = other;
  }
  after.after(el);
}

/** Mark each tile with a zoom: `data-zoom`, and on the grids its state. */
function markTiles(panel: HTMLElement, state: ZoomState): void {
  for (const link of panel.querySelectorAll<HTMLAnchorElement>("a.mosaic-link")) {
    const zoom = zoomFor(link.getAttribute("href") ?? "", state.zooms);
    if (!zoom) continue;
    link.dataset.zoom = zoom.slug;
    if (inPlace()) {
      link.setAttribute("aria-controls", `zoom-${zoom.slug}`);
      link.setAttribute("aria-expanded", String(zoom.slug === state.open));
    } else {
      link.removeAttribute("aria-controls");
      link.removeAttribute("aria-expanded");
    }
  }
}

/**
 * Draw the zoom the route names into the Overview's panel, and mark its
 * tiles. The Overview calls it after every render; it opens, keeps or
 * closes the zoom without moving focus, except back into a redrawn zoom
 * that held it.
 *
 * @param panel - The Overview's panel, its tiles already drawn.
 * @param snapshot - The dashboard snapshot.
 * @param zooms - The registry.
 */
export function drawZoom(
  panel: HTMLElement,
  snapshot: DashboardSnapshot,
  zooms: readonly ZoomDefinition[] = ZOOMS,
): void {
  const state = stateOf(panel, zooms);
  state.snapshot = snapshot;
  const was = state.open;
  const held = was !== null && document.activeElement?.closest(".zoom") != null;
  render(panel, state);
  // A zoom the page opens on its own (a shared link, a reload) steps back
  // as far as its history entry says, else replaces the route.
  if (state.open !== was) state.back = state.open === null ? 0 : (stampedBack() ?? 0);
  if (held) panel.querySelector<HTMLElement>(".zoom-heading")?.focus({ preventScroll: true });
}

/** A drawn tile, by slug, or null. */
function tileOf(panel: HTMLElement, slug: string): HTMLElement | null {
  return panel.querySelector<HTMLElement>(`[data-section="${CSS.escape(slug)}"]`);
}

/** The zoom the route names whose tile is drawn, or null. */
function routed(panel: HTMLElement, state: ZoomState): ZoomDefinition | null {
  const zoom = zoomFor(location.hash, state.zooms);
  return zoom && tileOf(panel, zoom.slug) ? zoom : null;
}

/** Draw the routed zoom (or none) from the state's snapshot. */
function render(panel: HTMLElement, state: ZoomState): void {
  panel.querySelector(":scope .zoom")?.remove();
  const zoom = state.snapshot ? routed(panel, state) : null;
  state.open = zoom?.slug ?? null;
  const screen = zoom !== null && !inPlace();
  panel.toggleAttribute("data-zoomed", screen);
  document.documentElement.toggleAttribute("data-zoom-screen", screen && !panel.hidden);
  markTiles(panel, state);
  if (!zoom || !state.snapshot) return;
  const el = shell(panel, zoom, state.snapshot);
  const tile = tileOf(panel, zoom.slug)!;
  const grid = tile.parentElement!;
  if (screen) grid.before(el);
  else place(grid, tile, el);
  fill(el, zoom, state.snapshot);
}

/**
 * Follow the route: the page calls this each time a tab opens, which the
 * tab shell does on every hash change. Opening a zoom moves focus to its
 * heading; closing one returns focus to the link that opened it, else its
 * tile. A zoom opened by following a link records how far back the closed
 * Overview is, so Close steps back to it.
 *
 * @param panel - The Overview's panel.
 * @param zooms - The registry.
 */
export function followZoom(panel: HTMLElement, zooms: readonly ZoomDefinition[] = ZOOMS): void {
  const state = stateOf(panel, zooms);
  const was = state.open;
  const wasBack = state.back;
  const first = !state.started;
  state.started = true;
  if (!state.snapshot) return;
  const next = routed(panel, state)?.slug ?? null;
  if (next === was) return;
  if (next !== null) {
    const stamped = stampedBack();
    if (stamped !== null) state.back = stamped;
    else {
      // Opened by following a link: one entry past the last zoom's, or
      // past the closed Overview; a shared link has none behind it.
      state.back = first ? 0 : was === null ? 1 : wasBack > 0 ? wasBack + 1 : 0;
      stampBack(state.back);
    }
  }
  if (next === null) state.back = 0;
  if (was === null && next !== null && !inPlace()) state.scrollY = window.scrollY;
  render(panel, state);
  if (panel.hidden) return;
  if (next !== null) {
    const heading = panel.querySelector<HTMLElement>(".zoom-heading");
    heading?.focus({ preventScroll: true });
    const el = panel.querySelector<HTMLElement>(".zoom");
    if (inPlace())
      el?.scrollIntoView?.({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
    else window.scrollTo?.(0, 0);
    return;
  }
  if (was !== null) {
    if (!inPlace()) window.scrollTo?.(0, state.scrollY);
    const origin = state.origin?.isConnected ? state.origin : tileLink(panel, was);
    state.origin = null;
    origin?.focus({ preventScroll: !inPlace() });
  }
}
