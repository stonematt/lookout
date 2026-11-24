/**
 * The Rain Events tab: the Event Catalog's events from the snapshot as one
 * exclusive accordion, each row opening its own curve, and the open event in
 * the URL (`?event=<start_ms>`), so a link opens the same row (#339).
 *
 * The tab holds the state and the wiring; each part's markup is its own
 * module (#261): `catalog.ts` the top tile, `list.ts` the list tile with its
 * filter bar and rows, `detail.ts` the open row's body. All of it is built
 * with the escaping `html` template (`web/src/dom.ts`).
 *
 * - The tab is two tiles (`web/src/tiles.ts`, spec #332): the catalog
 *   (kicker, headline, scatter, folded source; #666), the phone's fixed
 *   top; under it, the event list, which the filter line alone drives (DESIGN.md "Event list", #254): one
 *   filter line pinned under any sticky header while the list scrolls
 *   (water year, sort and size, three selects: "2026–27", "Newest",
 *   "Showers and up"; the size kept in the browser), a count line with the
 *   days missing, the rows, "Show N more" and the folded method.
 * - The list opens on the newest water year, every event of it of the size
 *   picked (sprinkles hidden at first, #644).
 * - A closed row is a link to its event (#370: ctrl-click opens a new
 *   tab): its start, the four-step category glyph and word, length, peak
 *   and quality chip, its total, and a size bar (its total over the
 *   catalog's largest, hidden from assistive tech). The open row adds the
 *   event's finding, its stats, one chart of its rain (columns) and its
 *   running total (a line, #419), its quality and its folded source.
 *   Opening another row closes it; opening writes `?event=`.
 * - With no `?event=`, the live event is open, or with none live, the newest
 *   event in the water year shown. With none live the newest row carries
 *   the "latest" chip. A linked event shows in its own water year, the size
 *   widened to list it.
 * - The live event's curve is the snapshot's `open_curve`, which the
 *   snapshot holds only for the live event (`active_rain_event`, the hero's
 *   decision); without one it says its readings are "not in the snapshot
 *   yet". Any other event's is its curve file (`/event`), read when its row
 *   opens: "Loading the curve…" while it reads, and on until the chart is
 *   drawn (Plotly loads on first use), so an empty chart never reads as no
 *   rain; "Couldn't load this event's curve." with Try again if the read
 *   fails. An event the catalog still holds open after the rain has stopped
 *   is not live and has no file yet, so it reads "not published yet" until
 *   then. An event with no archive data shows its totals only, with "n/a:
 *   gap in record".
 * - The event is keyed by its start, which is stable across refresh runs,
 *   so a link opened while it is live still opens it once it has completed.
 * - A newer snapshot mounts the tab again over the same panel; it keeps the
 *   water year, the order, the rows "Show more" opened and a row closed by
 *   hand, and redraws a completed event from the curve it already read (the
 *   per-panel `Kept` record). Keyboard focus goes back to the element that
 *   had it, and an open event already shown is not announced again (#239).
 * - A `?event=` the catalog does not hold opens the default row with a note
 *   saying why: the link names no event in the catalog, or the event the
 *   visitor had open here has left it. The stale `?event=` is cleared, so
 *   the note shows once.
 */
import "./events.css";

import { html } from "../dom.js";
import { ReadError, fetchEventCurve } from "../load.js";
import type { DashboardSnapshot, EventCurve, RainEventRow } from "../snapshot.js";
import { renderTiles, sourceNote } from "../tiles.js";
import { renderCatalogHead, renderCatalogTile } from "./catalog.js";
import type { EventChartOptions } from "./charts.js";
import type { MissingEvent } from "./detail.js";
import { chartLabel, eventBody, eventSource } from "./detail.js";
import {
  PAGE,
  emptyListLine,
  eventRows,
  listFoot,
  renderListTile,
  storeSize,
  storedSize,
  syncListHead,
} from "./list.js";
import type { EventSize, EventSort, Season } from "./model.js";
import {
  SIZES,
  SORTS,
  currentSeason,
  defaultOpen,
  inSize,
  latestStart,
  sizeFor,
  selectedStart,
  statsFor,
  visibleEvents,
  withEvent,
} from "./model.js";

/** What the tab reads and draws with; tests replace these. */
export interface EventsDeps {
  /** A completed event's curve; rejects with a `ReadError` 404 if unwritten. */
  fetchCurve(startMs: number): Promise<EventCurve>;
  /** Draw the open row's chart into its host. */
  draw(
    host: HTMLElement,
    curve: EventCurve,
    row: RainEventRow,
    options: EventChartOptions,
  ): Promise<void>;
  /** The browser's clock. */
  nowMs(): number;
}

/** The real sources: `/event` and Plotly, imported when first needed. */
export function browserDeps(): EventsDeps {
  return {
    fetchCurve: (startMs) => fetchEventCurve(startMs, location.search),
    draw: async (host, curve, row, options) => {
      const { drawEventChart } = await import("./charts.js");
      await drawEventChart(host, curve, row, options);
    },
    nowMs: () => Date.now(),
  };
}

/** The list's shape the visitor picked. */
interface View {
  season: Season;
  sort: EventSort;
  /** The smallest category listed, or all; kept in the browser too. */
  size: EventSize;
  limit: number;
}

/** What a panel keeps from its last mount for the next one. */
interface Kept {
  /** The station the panel was drawn for: another station keeps nothing. */
  station: string;
  /** The last mount's listeners, dropped by the next mount. */
  listening: AbortController;
  /** The view, so a newer snapshot redraws the tab as the visitor left it. */
  view: View;
  /** The completed events' curves read, redrawn without a second read. */
  curves: Map<number, EventCurve>;
  /** The visitor closed the open row, and the URL names none: keep it shut. */
  shut: boolean;
  /**
   * The event whose row body was last shown, or null: a newer snapshot
   * does not announce it again, and if it has left the catalog the note
   * says it has gone rather than that a link was wrong.
   */
  shown: number | null;
}

/**
 * Each panel's kept state, across mounts. The open event is not in it: it
 * is in the URL (`?event=`), which every mount reads.
 */
const panels = new WeakMap<HTMLElement, Kept>();

/**
 * The view a mount starts from: the panel's last one, or the first page of
 * the newest water year, newest first, in the size kept in the browser. A
 * water year the snapshot no longer lists goes to the newest, with the
 * first page, the order and the size kept.
 */
function keptView(last: View | undefined, section: DashboardSnapshot["rain_events"]): View {
  if (!last) {
    return { season: currentSeason(section), sort: "newest", size: storedSize(), limit: PAGE };
  }
  const { season } = last;
  if (season === "all" || section.seasons.some((s) => s.water_year === season)) return last;
  return { ...last, season: currentSeason(section), limit: PAGE };
}

/** The page address that opens an event on this tab (#370). */
function eventHref(startMs: number): string {
  return `${location.pathname}${withEvent(location.search, startMs)}#events`;
}

/** The controls a newer snapshot can find again by name, to keep focus on. */
const FOCUS_NAMES = ["ev-year-select", "ev-sort-select", "ev-size-select", "ev-more", "ev-retry"];

/**
 * Where keyboard focus sits in the panel, as something the next mount can
 * find again: a row's header by its event, a control by its class, any
 * other element by its place among the panel's elements of its tag.
 */
type FocusMark = { selector: string } | { tag: string; index: number };

/** The panel's focused element as a mark, or null when focus is elsewhere. */
function focusMark(panel: HTMLElement): FocusMark | null {
  const el = document.activeElement;
  if (!(el instanceof HTMLElement) || el === panel || !panel.contains(el)) return null;
  const start = el.closest<HTMLElement>(".ev-open")?.dataset.start;
  if (start) return { selector: `.ev-open[data-start='${start}']` };
  const name = FOCUS_NAMES.find((n) => el.classList.contains(n));
  if (name) return { selector: `.${name}` };
  const tag = el.tagName.toLowerCase();
  return { tag, index: [...panel.querySelectorAll(tag)].indexOf(el) };
}

/** Put focus back on the marked element, if the new markup still has it. */
function refocus(panel: HTMLElement, mark: FocusMark | null): void {
  if (!mark) return;
  const el =
    "selector" in mark
      ? panel.querySelector<HTMLElement>(mark.selector)
      : panel.querySelectorAll<HTMLElement>(mark.tag)[mark.index];
  el?.focus({ preventScroll: true });
}

/** Whether a click asks the browser to open the link elsewhere. */
function opensElsewhere(event: MouseEvent): boolean {
  return event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey;
}

/**
 * The chart area's loading note, "Loading the curve…" (DESIGN.md "Loading
 * a curve"). The read's note shows at once and is a status for assistive
 * tech; the drawing's (`later`) shows only after a moment and is not
 * announced, so a quick redraw at each newer snapshot neither flashes nor
 * speaks.
 *
 * @param later - The note waits for the chart's drawing, not a read.
 * @returns The note.
 */
function loadingNote(later = false): HTMLElement {
  const note = document.createElement("p");
  note.className = later ? "ev-wait ev-loading ev-later" : "ev-wait ev-loading";
  if (!later) note.setAttribute("role", "status");
  note.textContent = "Loading the curve…";
  return note;
}

/**
 * How far below the top of the viewport the filter bar pins: under every
 * header part held at the top (a sticky or fixed element in the top half of
 * the viewport; a bar fixed at the bottom, the phone's tabs, is not one).
 */
function stickyTop(): number {
  let bottom = 0;
  for (const el of document.querySelectorAll<HTMLElement>(
    "#masthead, #masthead > *, .page > *, .tabs",
  )) {
    if (el.closest("[role=tabpanel], #sections")) continue;
    const style = getComputedStyle(el);
    if (style.position !== "sticky" && style.position !== "fixed") continue;
    const top = parseFloat(style.top);
    const box = el.getBoundingClientRect();
    if (Number.isNaN(top) || box.top > window.innerHeight / 2) continue;
    bottom = Math.max(bottom, top + box.height);
  }
  return Math.round(bottom);
}

/**
 * Render the Rain Events tab into its panel. Mounted again over the same
 * panel (a newer snapshot), it keeps the open event, the water year, the
 * order and the rows "Show more" opened; for another station it keeps
 * nothing.
 *
 * @param panel - The tab's panel; its content is replaced.
 * @param snapshot - The dashboard snapshot.
 * @param deps - Curve source, chart drawing and clock.
 */
export function mountEvents(
  panel: HTMLElement,
  snapshot: DashboardSnapshot,
  deps: EventsDeps = browserDeps(),
): void {
  const before = panels.get(panel);
  before?.listening.abort();
  // The markup is rebuilt below; keyboard focus goes back where it was.
  const focused = focusMark(panel);
  // Two stations can share an event start; a curve is one station's.
  const last = before?.station === snapshot.station.mac ? before : undefined;
  const section = snapshot.rain_events;
  const all = section.events;
  const asked = selectedStart(location.search);
  const kept: Kept = {
    station: snapshot.station.mac,
    listening: new AbortController(),
    view: keptView(last?.view, section),
    curves: last?.curves ?? new Map(),
    shut: asked === null && (last?.shut ?? false),
    shown: last?.shown ?? null,
  };
  panels.set(panel, kept);
  const { listening, view } = kept;
  const linked = asked === null ? null : (all.find((e) => e.start_ms === asked) ?? null);
  // A linked event outside the water year kept shows in its own (under All
  // when the snapshot lists no such year).
  if (linked && view.season !== "all" && linked.water_year !== view.season) {
    const own = section.seasons.some((s) => s.water_year === linked.water_year);
    view.season = own ? linked.water_year : "all";
    view.limit = PAGE;
  }
  // A linked event smaller than the size kept widens the size to list it.
  if (linked?.category && !inSize(linked, view.size)) {
    view.size = sizeFor(linked.category);
  }
  let selected: RainEventRow | null = kept.shut
    ? null
    : (linked ?? defaultOpen(section, view.season, view.size));
  // The URL names an event the catalog does not hold: the one the visitor
  // had open here has gone, or the link was never good. Say so once, and
  // clear the stale `?event=` so the next snapshot does not say it again.
  let missing: MissingEvent =
    asked === null || linked !== null ? null : last?.shown === asked ? "vanished" : "linked";
  if (missing) writeUrl(null);
  const latest = latestStart(section);
  let drawn = 0;

  // Two tiles (spec #332, DESIGN.md "Layout"): the catalog, its finding and
  // stats, the phone's fixed top; under it, the event list.
  renderTiles(panel, snapshot, [
    {
      id: "catalog",
      label: "Rain events",
      kicker: "Rain events",
      span: 4,
      phoneTop: true,
      render: (host) => renderCatalogTile(host, snapshot),
    },
    {
      id: "events",
      label: "The list of rain events",
      kicker: "The event list",
      span: 4,
      // The event list is itself an accordion: open on a phone, not a row.
      phoneOpen: true,
      render: (host) => renderListTile(host, section),
    },
  ]);
  const $ = <T extends HTMLElement>(selector: string) => panel.querySelector<T>(selector)!;
  const listTile = $("[data-section=events]");
  const pin = () => listTile.style.setProperty("--ev-bar-top", `${stickyTop()}px`);
  pin();
  window.addEventListener("resize", pin, { signal: listening.signal, passive: true });

  /** The catalog tile's finding and stats, and the filter line, for the pick. */
  function renderHead(): void {
    const rows = visibleEvents(section, view.season, view.sort, view.size);
    const hidden = visibleEvents(section, view.season, view.sort).length - rows.length;
    renderCatalogHead(panel, section, view.season, rows);
    syncListHead(panel, section, view, { shown: rows.length, hidden });
  }

  /** The rows shown and the foot under them, then the open row's body. */
  function renderList(): void {
    const rows = visibleEvents(section, view.season, view.sort, view.size);
    // The catalog's largest event total, the size bars' full track.
    const now = deps.nowMs();
    const most = Math.max(0, ...all.map((e) => e.total_in));
    const list = $(".ev-list");
    if (!rows.length) {
      const hidden = visibleEvents(section, view.season, view.sort).length;
      // Markup from the list module is already `html`; the outer tag is
      // what the no-unsanitized rule can see, and passes it as is.
      list.innerHTML = html`${emptyListLine(all.length ? view.season : "all", hidden)}`;
      $(".ev-foot").replaceChildren();
      return;
    }
    const context = {
      openStart: selected?.start_ms ?? null,
      latest,
      now,
      most,
      href: eventHref,
    };
    list.innerHTML = html`${eventRows(rows.slice(0, view.limit), context)}`;
    $(".ev-foot").innerHTML = html`${listFoot(rows.length - view.limit, most)}`;
    renderBody();
  }

  /** The open row's body, then its chart: drawn, read, or waiting. */
  function renderBody(): void {
    const row = selected;
    const host = row ? panel.querySelector<HTMLElement>(".ev-row[data-open] .ev-body") : null;
    if (!row || !host) {
      kept.shown = null;
      return;
    }
    const ticket = ++drawn;
    const announce = kept.shown !== row.start_ms;
    kept.shown = row.start_ms;
    host.innerHTML = html`${eventBody(row, all.length, missing, announce)}`;
    const charts = host.querySelector<HTMLElement>(".ev-charts")!;
    host.append(sourceNote(snapshot, eventSource(row)));

    if (!row.has_curve) {
      charts.innerHTML = html`<p class="ev-gap gap-hatch" role="note">n/a: gap in record<span class="sr-only">. The archive has no readings for this event, so only its catalog totals are shown.</span></p>`;
      return;
    }
    const curve = curveOf(row);
    if (curve) {
      show(charts, curve, row, ticket);
      return;
    }
    if (row.live) {
      charts.innerHTML = html`<p class="ev-note ev-wait">This event's readings are not in the snapshot yet.</p>`;
      return;
    }
    charts.replaceChildren(loadingNote());
    deps.fetchCurve(row.start_ms).then(
      (curve) => {
        kept.curves.set(row.start_ms, curve);
        show(charts, curve, row, ticket);
      },
      (error: unknown) => {
        if (ticket !== drawn) return;
        if (error instanceof ReadError && error.status === 404) {
          charts.innerHTML = html`<p class="ev-note ev-wait">This event's readings are not published yet. The refresh run adds them within the hour.</p>`;
          return;
        }
        console.error("Lookout: event curve could not be read", error);
        charts.innerHTML = html`<div class="ev-wait ev-failed" role="alert"><p>Couldn't load this event's curve.</p><button type="button" class="ev-retry">Try again</button></div>`;
      },
    );
  }

  /** An event's curve already to hand: the snapshot's open curve, or one read. */
  function curveOf(row: RainEventRow): EventCurve | null {
    const open = section.open_curve;
    if (open && open.start_ms === row.start_ms) return open;
    return kept.curves.get(row.start_ms) ?? null;
  }

  /**
   * The open row's chart from its curve, against the median event of the
   * water year shown (the catalog tile's). Until it is drawn the area says
   * it is loading: the note the read put there, or one that shows only if
   * the drawing takes a moment (a redraw with Plotly at hand does not).
   */
  function show(charts: HTMLElement, curve: EventCurve, row: RainEventRow, ticket: number): void {
    if (ticket !== drawn) return;
    const median = statsFor(section, view.season).median_in;
    const chart = document.createElement("div");
    chart.className = "ev-chart";
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", chartLabel(row, median));
    const reading = charts.querySelector<HTMLElement>(":scope > .ev-loading");
    for (const child of [...charts.children]) if (child !== reading) child.remove();
    const note = reading ?? loadingNote(true);
    charts.prepend(chart);
    if (!reading) charts.append(note);
    charts.setAttribute("aria-busy", "true");
    deps.draw(chart, curve, row, { medianIn: median }).then(
      () => {
        note.remove();
        charts.removeAttribute("aria-busy");
      },
      (error: unknown) => {
        charts.removeAttribute("aria-busy");
        if (ticket !== drawn) return;
        console.error("Lookout: event chart failed", error);
        charts.innerHTML = html`<p class="ev-note ev-wait">The chart could not be drawn.</p>`;
      },
    );
  }

  /** Make sure the open row is on the rows shown. */
  function reach(): void {
    if (!selected) return;
    const at = visibleEvents(section, view.season, view.sort, view.size).indexOf(selected);
    if (at >= view.limit) view.limit = at + 1;
  }

  /** Write the open event to the URL, or clear it. */
  function writeUrl(startMs: number | null): void {
    history.replaceState(
      history.state,
      "",
      `${location.pathname}${withEvent(location.search, startMs)}${location.hash}`,
    );
  }

  /** Open a row, closing the open one, keeping its header where it was. */
  function toggle(startMs: number): void {
    const button = () =>
      panel.querySelector<HTMLElement>(`.ev-open[data-start='${startMs}']`) ?? null;
    const top = button()?.getBoundingClientRect().top ?? 0;
    if (selected?.start_ms === startMs) {
      selected = null;
      kept.shut = true;
      writeUrl(null);
    } else {
      selected = all.find((e) => e.start_ms === startMs) ?? null;
      if (!selected) return;
      kept.shut = false;
      missing = null;
      writeUrl(startMs);
    }
    renderList();
    const after = button();
    if (!after) return;
    // The row above closing moves this one: keep its header under the finger.
    const shift = after.getBoundingClientRect().top - top;
    if (shift) window.scrollBy(0, shift);
    after.focus({ preventScroll: true });
  }

  panel.addEventListener(
    "click",
    (event) => {
      const target = event.target as Element;
      const head = target.closest<HTMLElement>(".ev-open");
      if (head) {
        // A row is a link (#370): ctrl-, cmd-, shift- or middle-click
        // opens its event elsewhere, as the browser does; a plain click
        // opens or closes it here.
        if (opensElsewhere(event)) return;
        event.preventDefault();
        toggle(Number(head.dataset.start));
        return;
      }
      if (target.closest(".ev-retry")) {
        renderBody();
        return;
      }
      if (target.closest(".ev-more")) {
        view.limit += PAGE * 2;
        renderList();
      }
    },
    { signal: listening.signal },
  );

  panel.addEventListener(
    "change",
    (event) => {
      const select = (event.target as Element).closest<HTMLSelectElement>(".ev-select");
      if (!select) return;
      view.limit = PAGE;
      if (select.classList.contains("ev-sort-select")) {
        view.sort = SORTS.find((s) => s.sort === select.value)?.sort ?? view.sort;
      } else if (select.classList.contains("ev-size-select")) {
        view.size = SIZES.find((s) => s.size === select.value)?.size ?? view.size;
        storeSize(view.size);
        // An open row the size hides gives way to the default rule.
        if (selected && !inSize(selected, view.size)) {
          selected = kept.shut ? null : defaultOpen(section, view.season, view.size);
          missing = null;
          writeUrl(null);
        }
      } else {
        view.season = select.value === "all" ? "all" : Number(select.value);
        // The open row follows the default rule within the water year
        // picked: the live event, or the newest. A row closed by hand stays
        // closed.
        if (!kept.shut) {
          selected = defaultOpen(section, view.season, view.size);
          missing = null;
          writeUrl(null);
        }
      }
      reach();
      renderHead();
      renderList();
    },
    { signal: listening.signal },
  );

  reach();
  renderHead();
  renderList();
  refocus(panel, focused);
}
