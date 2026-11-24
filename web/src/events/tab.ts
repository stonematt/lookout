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
 *   (kicker, finding, standfirst, stats, folded source), the phone's fixed
 *   top; under it, the event list (DESIGN.md "Event list", #254): one
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
 *   event's finding, its stats, a Total / Rate switch over one chart, its
 *   quality and its folded source. Opening another row closes it; opening
 *   writes `?event=`.
 * - With no `?event=`, the live event is open, or with none live, the newest
 *   event in the water year shown. With none live the newest row carries
 *   the "latest" chip. A linked event shows in its own water year, the size
 *   widened to list it.
 * - The live event's curve is the snapshot's `open_curve`, which the
 *   snapshot holds only for the live event (`active_rain_event`, the hero's
 *   decision); without one it says its readings are "not in the snapshot
 *   yet". Any other event's is its curve file (`/event`), read when its row
 *   opens: "Loading the curve…" while it reads, "Couldn't load this event's
 *   curve." with Try again if the read fails. An event the catalog still
 *   holds open after the rain has stopped is not live and has no file yet,
 *   so it reads "not published yet" until then. An event with no archive
 *   data shows its totals only, with "n/a: gap in record".
 * - The event is keyed by its start, which is stable across refresh runs,
 *   so a link opened while it is live still opens it once it has completed.
 * - A newer snapshot mounts the tab again over the same panel; it keeps the
 *   water year, the order, the rows "Show more" opened, the Total / Rate
 *   pick and a row closed by hand, and redraws a completed event from the
 *   curve it already read (the per-panel `Kept` record).
 */
import "./events.css";

import { html } from "../dom.js";
import { ReadError, fetchEventCurve } from "../load.js";
import type { DashboardSnapshot, EventCurve, RainEventRow } from "../snapshot.js";
import { renderTiles, sourceNote } from "../tiles.js";
import { renderCatalogHead, renderCatalogTile } from "./catalog.js";
import type { EventChart, EventChartOptions } from "./charts.js";
import { chartLabel, eventBody, eventSource } from "./detail.js";
import {
  PAGE,
  emptyListLine,
  eventRows,
  listFoot,
  renderListTile,
  storeSize,
  storedSize,
  syncFilterBar,
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
  /** Draw the open row's chart, Total or Rate, into its host. */
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
  /** The open row's chart. */
  chart: EventChart;
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
}

/**
 * Each panel's kept state, across mounts. The open event is not in it: it
 * is in the URL (`?event=`), which every mount reads.
 */
const panels = new WeakMap<HTMLElement, Kept>();

/**
 * The view a mount starts from: the panel's last one, or the first page of
 * the newest water year, newest first, in the size kept in the browser, on
 * Total. A water year the snapshot no longer lists goes to the newest, with
 * the first page, the order, the size and the chart kept.
 */
function keptView(last: View | undefined, section: DashboardSnapshot["rain_events"]): View {
  if (!last) {
    return {
      season: currentSeason(section),
      sort: "newest",
      size: storedSize(),
      limit: PAGE,
      chart: "total",
    };
  }
  const { season } = last;
  if (season === "all" || section.seasons.some((s) => s.water_year === season)) return last;
  return { ...last, season: currentSeason(section), limit: PAGE };
}

/** The page address that opens an event on this tab (#370). */
function eventHref(startMs: number): string {
  return `${location.pathname}${withEvent(location.search, startMs)}#events`;
}

/** Whether a click asks the browser to open the link elsewhere. */
function opensElsewhere(event: MouseEvent): boolean {
  return event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey;
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
 * order, the chart and the rows "Show more" opened; for another station it
 * keeps nothing.
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
  let missing = asked !== null && linked === null;
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
    syncFilterBar(panel, section, view, { shown: rows.length, hidden });
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
    if (!row || !host) return;
    const ticket = ++drawn;
    host.innerHTML = html`${eventBody(row, all.length, view.chart, missing)}`;
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
    charts.innerHTML = html`<p class="ev-wait" role="status">Loading the curve…</p>`;
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
   * The open row's chart, Total or Rate as the switch says, from its curve,
   * against the median event of the water year shown (the catalog tile's).
   */
  function show(charts: HTMLElement, curve: EventCurve, row: RainEventRow, ticket: number): void {
    if (ticket !== drawn) return;
    const median = statsFor(section, view.season).median_in;
    const chart = document.createElement("div");
    chart.className = "ev-chart";
    chart.dataset.chart = view.chart;
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", chartLabel(row, view.chart, median));
    charts.replaceChildren(chart);
    deps
      .draw(chart, curve, row, { chart: view.chart, medianIn: median })
      .catch((error: unknown) => {
        if (ticket !== drawn) return;
        console.error("Lookout: event chart failed", error);
        charts.innerHTML = html`<p class="ev-note ev-wait">The chart could not be drawn.</p>`;
      });
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
      missing = false;
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
      const chartButton = target.closest<HTMLButtonElement>("[data-chart]");
      if (chartButton) {
        view.chart = chartButton.dataset.chart as EventChart;
        for (const b of panel.querySelectorAll<HTMLButtonElement>("[data-chart]")) {
          b.setAttribute("aria-pressed", String(b === chartButton));
        }
        // Only the chart changes: the row, its switch and its fold stay. A
        // curve still loading draws on the new pick when it arrives.
        const charts = panel.querySelector<HTMLElement>(".ev-row[data-open] .ev-charts");
        const curve = selected && curveOf(selected);
        if (charts && curve && selected) show(charts, curve, selected, drawn);
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
          missing = false;
          writeUrl(null);
        }
      } else {
        view.season = select.value === "all" ? "all" : Number(select.value);
        // The open row follows the default rule within the water year
        // picked: the live event, or the newest. A row closed by hand stays
        // closed.
        if (!kept.shut) {
          selected = defaultOpen(section, view.season, view.size);
          missing = false;
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
}
