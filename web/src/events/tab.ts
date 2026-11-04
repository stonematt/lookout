/**
 * The Rain Events tab: the Event Catalog's events from the snapshot as one
 * exclusive accordion, each row opening its own curve, and the open event in
 * the URL (`?event=<start_ms>`), so a link opens the same row (#339).
 *
 * - The tab is two tiles (`web/src/tiles.ts`, spec #332): the catalog
 *   (kicker, finding, standfirst, stats, folded source), the phone's fixed
 *   top; under it, the event list (DESIGN.md "Event list"): a filter bar
 *   pinned under any sticky header while the list scrolls (a water-year
 *   select and a Newest / Wettest / Longest sort), the rows, "Show N more".
 * - A closed row is the event's start, length, peak and quality chip, its
 *   total, and a size bar (its total over the catalog's largest, hidden
 *   from assistive tech). The open row adds the event's finding, its stats,
 *   a Total / Rate switch over one chart, its quality and its folded source.
 *   Opening another row closes it; opening writes `?event=`.
 * - With no `?event=`, the live event is open, or with none live, the newest
 *   event in the water year shown. With none live the newest row carries
 *   the "latest" chip.
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

import { inches, inchesPerHour, waterYearSpan } from "../format.js";
import { ReadError, fetchEventCurve } from "../load.js";
import { escapeHtml as esc } from "../dom.js";
import { gapElement, gapLabel, isComparable } from "../gaps.js";
import { staleCatalogNote } from "../stale.js";
import { notCompared, renderTiles, sourceNote, tileFolds } from "../tiles.js";
import type { DashboardSnapshot, EventCurve, EventSeason, RainEventRow } from "../snapshot.js";
import type { EventChart, EventChartOptions } from "./charts.js";
import type { EventSort, Season } from "./model.js";
import {
  SORTS,
  defaultOpen,
  emptyList,
  eventFinding,
  latestStart,
  listFinding,
  listHeading,
  lowQuality,
  minutes,
  qualityLabel,
  seasonNote,
  selectedStart,
  sizeShare,
  startLabel,
  statsFor,
  visibleEvents,
  withEvent,
} from "./model.js";

/** Rows shown before "Show more". */
const PAGE = 20;

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

/**
 * An event's start as two unbreakable parts, date and time, so a narrow
 * row breaks it between them: `Jan 8, 2025,` and `2:00 pm`.
 */
function when(label: string): string {
  const cut = label.lastIndexOf(", ") + 1;
  if (cut === 0) return `<span class="ev-when">${esc(label)}</span>`;
  return `<span><span class="ev-when">${esc(label.slice(0, cut))}</span> <span class="ev-when">${esc(label.slice(cut + 1))}</span></span>`;
}

/** A one-row stat strip cell. */
function stat(value: string, label: string): string {
  return `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`;
}

/**
 * A water year's option in the select. A year the record does not cover
 * reads "<year>, partial" ("not archived" for one the archive does not
 * reach yet), short so the select stays beside the sort on a
 * phone, with its reason in the title; a ≈ one is the year alone (a ≈ is
 * never marked as a gap). The catalog tile's note says why once picked.
 */
function yearOption(season: EventSeason): string {
  const { status } = season;
  const year = esc(season.water_year);
  const span = esc(waterYearSpan(season.water_year));
  if (isComparable(status)) return `<option value="${year}" title="${span}">${year}</option>`;
  const reason = gapLabel(status)!;
  const word = status.reason === "not_archived" ? "not archived" : "partial";
  return `<option value="${year}" title="${esc(`${waterYearSpan(season.water_year)}: ${reason}`)}" data-gap="${esc(status.reason ?? "gap")}">${year}, ${word}</option>`;
}

/** The list's shape the visitor picked. */
interface View {
  season: Season;
  sort: EventSort;
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
 * every event, newest first, on Total. A water year the snapshot no longer
 * lists goes back to All, with the first page, the order and the chart kept.
 */
function keptView(last: View | undefined, section: DashboardSnapshot["rain_events"]): View {
  if (!last) return { season: "all", sort: "newest", limit: PAGE, chart: "total" };
  const { season } = last;
  if (season === "all" || section.seasons.some((s) => s.water_year === season)) return last;
  return { ...last, season: "all", limit: PAGE };
}

/**
 * How far below the top of the viewport the filter bar pins: under every
 * header part held at the top (a sticky or fixed element with a `top`).
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
    if (Number.isNaN(top)) continue;
    bottom = Math.max(bottom, top + el.getBoundingClientRect().height);
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
  // A linked event outside the water year kept shows under All.
  if (linked && view.season !== "all" && linked.water_year !== view.season) {
    view.season = "all";
  }
  let selected: RainEventRow | null = kept.shut
    ? null
    : (linked ?? defaultOpen(section, view.season));
  let missing = asked !== null && linked === null;
  const latest = latestStart(section);
  let drawn = 0;

  // Two tiles (spec #332, DESIGN.md "Layout"): the catalog, its finding and
  // stats, the phone's fixed top; under it, the event list.
  renderTiles(panel, snapshot, [
    {
      id: "catalog",
      label: "The Event Catalog",
      kicker: "The Event Catalog",
      span: 4,
      phoneTop: true,
      render: (host) => {
        host.classList.add("ev-tile");
        host.insertAdjacentHTML(
          "beforeend",
          `
<h2 class="tile-finding ev-title" data-part="finding"></h2>
<p class="standfirst" data-part="standfirst">Each storm the station recorded, from the Event Catalog. Open one to see how its rain built up and how hard it fell.</p>
<p class="ev-note ev-stale" role="status" hidden></p>
<dl class="ev-stats"></dl>
<p class="ev-note ev-season-gap" role="note" hidden></p>`,
        );
        const uncovered = section.seasons.filter((s) => !isComparable(s.status)).length;
        host.append(
          tileFolds(
            sourceNote(snapshot, {
              what: `Event Catalog, ${all.length} event${all.length === 1 ? "" : "s"}`,
              method: `The Event Catalog, ${all.length} event${all.length === 1 ? "" : "s"}, each storm the station recorded. Peak is the highest rate between 5-minute readings. A dashed quality chip marks a poor event or one with no archive data.`,
              gaps: notCompared(uncovered, "water year", "water years"),
            }),
          ),
        );
      },
    },
    {
      id: "events",
      label: "The list of rain events",
      kicker: "Rain events",
      span: 4,
      // The event list is itself an accordion: open on a phone, not a row.
      phoneOpen: true,
      render: (host) => {
        host.classList.add("ev-tile", "ev-list-tile");
        // The filter bar heads the list; the kicker and finding are named
        // for assistive tech, the catalog tile above says it on screen.
        host.querySelector("[data-part=kicker]")!.classList.add("ov-sr");
        const sorts = SORTS.map(
          ({ sort, label }) => `<button type="button" data-sort="${sort}">${label}</button>`,
        ).join("");
        // Literal markup; the select's options are escaped in yearOption.
        // eslint-disable-next-line no-unsanitized/method
        host.insertAdjacentHTML(
          "beforeend",
          `
<h2 class="tile-finding ev-list-finding ov-sr" data-part="finding"></h2>
<div class="ev-bar" role="group" aria-label="Filter and sort the events">
  <label class="ev-year"><span class="ov-sr">Water year</span><select class="ev-select">
    <option value="all">All years</option>${section.seasons.map(yearOption).join("")}
  </select></label>
  <div class="ev-seg ev-sort" role="group" aria-label="Order">${sorts}</div>
</div>
<div class="ev-list"></div>
<div class="ev-foot"></div>`,
        );
      },
    },
  ]);
  const $ = <T extends HTMLElement>(selector: string) => panel.querySelector<T>(selector)!;
  const stale = staleCatalogNote(snapshot, "events");
  if (stale) {
    $(".ev-stale").textContent = stale;
    $(".ev-stale").hidden = false;
  }
  const listTile = $("[data-section=events]");
  const pin = () => listTile.style.setProperty("--ev-bar-top", `${stickyTop()}px`);
  pin();
  window.addEventListener("resize", pin, { signal: listening.signal, passive: true });

  function renderHead(): void {
    const stats = statsFor(section, view.season);
    const oldest = all.length ? all[all.length - 1]!.start_ms : null;
    const rows = visibleEvents(section, view.season, view.sort);
    $(".ev-title").textContent = all.length
      ? listFinding(stats, view.season, oldest, rows)
      : "No rain events in the catalog yet";
    $(".ev-list-finding").textContent = listHeading(rows.length, view.season, view.sort);
    // Each cell is escaped in stat; the linter cannot follow it (#261).
    // eslint-disable-next-line no-unsanitized/property
    $(".ev-stats").innerHTML = all.length
      ? [
          stat(String(stats.count), "events"),
          stat(inches(stats.total_in, { digits: 2 }), "rain in events"),
          stat(
            stats.median_in === null ? "n/a" : inches(stats.median_in, { digits: 2 }),
            "median event",
          ),
          stat(
            stats.mean_duration_minutes === null ? "n/a" : minutes(stats.mean_duration_minutes),
            "mean length",
          ),
        ].join("")
      : "";
    $(".ev-stats").hidden = !all.length;
    const note = $(".ev-season-gap");
    const text = seasonNote(section, view.season);
    note.hidden = text === null;
    note.replaceChildren();
    const year =
      view.season === "all" ? undefined : section.seasons.find((s) => s.water_year === view.season);
    const status = year?.status;
    if (status && !isComparable(status)) note.append(gapElement(status), " ");
    if (text !== null) note.append(text);
    $<HTMLSelectElement>(".ev-select").value = String(view.season);
    for (const b of panel.querySelectorAll<HTMLButtonElement>("[data-sort]")) {
      b.setAttribute("aria-pressed", String(b.dataset.sort === view.sort));
    }
  }

  /** The catalog's largest event total, the size bars' full track. */
  function largest(): number {
    return Math.max(0, ...all.map((e) => e.total_in));
  }

  /** A closed row's header: the button that opens and closes it. */
  function rowHead(e: RainEventRow, open: boolean, now: number, most: number): string {
    const mark = e.live
      ? '<span class="ev-live">Live</span>'
      : e.start_ms === latest
        ? '<span class="ev-latest">latest</span>'
        : "";
    const peak =
      e.peak_rate_in_hr === null || !e.has_curve ? "n/a" : inchesPerHour(e.peak_rate_in_hr);
    const chip = `<span class="ev-chip${lowQuality(e) ? " low" : ""}">${esc(qualityLabel(e))}</span>`;
    const share = (sizeShare(e, most) * 100).toFixed(1);
    return `<h3 class="ev-row-head"><button type="button" class="ev-open" data-start="${esc(e.start_ms)}" aria-expanded="${open}" aria-controls="ev-body-${esc(e.start_ms)}">
  <span class="ev-date">${when(startLabel(e.start_ms, now))}${mark}</span>
  <span class="ev-total">${e.total_in.toFixed(2)}<small> in</small></span>
  <span class="ev-meta">${esc(minutes(e.duration_minutes))} · peak ${esc(peak)} ${chip}</span>
  <span class="ev-size" aria-hidden="true"><span style="width: ${share}%"></span></span>
</button></h3>`;
  }

  function renderList(): void {
    const rows = visibleEvents(section, view.season, view.sort);
    const shown = rows.slice(0, view.limit);
    const now = deps.nowMs();
    const most = largest();
    const list = $(".ev-list");
    if (!rows.length) {
      list.innerHTML = `<p class="ev-empty" role="status">${esc(emptyList(all.length ? view.season : "all"))}</p>`;
      $(".ev-foot").replaceChildren();
      return;
    }
    const body = shown
      .map((e) => {
        const open = selected?.start_ms === e.start_ms;
        return `<article class="ev-row"${open ? " data-open" : ""}${e.live ? " data-live" : ""} data-start="${esc(e.start_ms)}">
${rowHead(e, open, now, most)}
${open ? `<div class="ev-body" id="ev-body-${esc(e.start_ms)}" role="region" aria-label="${esc(`The event of ${startLabel(e.start_ms, now)}`)}"></div>` : ""}
</article>`;
      })
      .join("");
    // Rows escape their text with esc and when; the rest is numbers (#261).
    // eslint-disable-next-line no-unsanitized/property
    list.innerHTML = body;
    const more = rows.length - view.limit;
    // The button and the key are literals and numbers (#261).
    // eslint-disable-next-line no-unsanitized/property
    $(".ev-foot").innerHTML =
      `${more > 0 ? `<button type="button" class="ev-more">Show ${Math.min(PAGE * 2, more)} more</button>` : ""}
<p class="ev-key">Each bar under a row is that event's total against the wettest on record, ${esc(inches(most, { digits: 2 }))}. From the Event Catalog.</p>`;
    renderBody();
  }

  function renderBody(): void {
    const row = selected;
    const host = row ? panel.querySelector<HTMLElement>(".ev-row[data-open] .ev-body") : null;
    if (!row || !host) return;
    const ticket = ++drawn;
    const quality =
      row.quality === "invalid"
        ? "No archive readings."
        : `Quality ${row.quality}${row.completeness_pct === null ? "" : `, ${Math.round(row.completeness_pct)}% of readings`}${row.max_gap_minutes ? `, longest gap ${minutes(row.max_gap_minutes)}` : ""}.`;
    const note = missing
      ? '<p class="ev-note">The linked event is not in the catalog; showing the newest one instead.</p>'
      : "";
    const switcher = row.has_curve
      ? `<div class="ev-seg ev-mode" role="group" aria-label="Chart">
  <button type="button" data-chart="total" aria-pressed="${view.chart === "total"}">Total</button>
  <button type="button" data-chart="rate" aria-pressed="${view.chart === "rate"}">Rate</button>
</div>`
      : "";
    // Snapshot text goes through esc; the note is a literal (#261).
    // eslint-disable-next-line no-unsanitized/property
    host.innerHTML = `
${note}
<p class="ev-finding" aria-live="polite">${esc(eventFinding(row, all.length))}</p>
<dl class="ev-facts">
  ${stat(inches(row.total_in, { digits: 2 }), row.live ? "so far (live)" : "total")}
  ${stat(minutes(row.duration_minutes), "length")}
  ${stat(row.peak_rate_in_hr === null || !row.has_curve ? "n/a" : inchesPerHour(row.peak_rate_in_hr), "peak rate")}
</dl>
${switcher}
<div class="ev-charts"></div>
<p class="ev-quality">${esc(quality)}</p>`;
    const charts = host.querySelector<HTMLElement>(".ev-charts")!;
    const source = (what: string, method: string, gaps: string | null = null) =>
      host.append(sourceNote(snapshot, { what, method, gaps }));

    if (!row.has_curve) {
      charts.innerHTML =
        '<p class="ev-gap gap-hatch" role="note">n/a: gap in record<span class="sr-only">. The archive has no readings for this event, so only its catalog totals are shown.</span></p>';
      source(
        "Event Catalog",
        "The Event Catalog. The archive holds no readings in this event's time range.",
        "no archive readings",
      );
      return;
    }
    if (row.live) {
      source(
        "live station and archive, 5-minute readings",
        "The live station for the newest reading and the total; the station archive for the rest. 5-minute readings, station time. Total is the event's rain as it built up, against the median event's total, dashed. Rate is the rise in daily rain between readings.",
      );
    } else {
      source(
        "station archive, 5-minute readings",
        "The station archive, 5-minute readings, station time. Total is the event's rain as it built up, against the median event's total, dashed. Rate is the rise in daily rain between readings.",
      );
    }

    const curve = curveOf(row);
    if (curve) {
      show(charts, curve, row, ticket);
      return;
    }
    if (row.live) {
      charts.innerHTML =
        '<p class="ev-note ev-wait">This event\'s readings are not in the snapshot yet.</p>';
      return;
    }
    charts.innerHTML = '<p class="ev-wait" role="status">Loading the curve…</p>';
    deps.fetchCurve(row.start_ms).then(
      (curve) => {
        kept.curves.set(row.start_ms, curve);
        show(charts, curve, row, ticket);
      },
      (error: unknown) => {
        if (ticket !== drawn) return;
        if (error instanceof ReadError && error.status === 404) {
          charts.innerHTML =
            '<p class="ev-note ev-wait">This event\'s readings are not published yet. The refresh run adds them within the hour.</p>';
          return;
        }
        console.error("Lookout: event curve could not be read", error);
        charts.innerHTML =
          '<div class="ev-wait ev-failed" role="alert"><p>Couldn\'t load this event\'s curve.</p><button type="button" class="ev-retry">Try again</button></div>';
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
    const label =
      view.chart === "total"
        ? `Rain so far in this event, rising to ${inches(row.total_in, { digits: 2 })}${median === null ? "" : `, against ${inches(median, { digits: 2 })} for the median event`}`
        : row.peak_rate_in_hr === null
          ? "Rain rate between readings"
          : `Rain rate between readings, peaking at ${inchesPerHour(row.peak_rate_in_hr)}`;
    const chart = document.createElement("div");
    chart.className = "ev-chart";
    chart.dataset.chart = view.chart;
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", label);
    charts.replaceChildren(chart);
    deps
      .draw(chart, curve, row, { chart: view.chart, medianIn: median })
      .catch((error: unknown) => {
        if (ticket !== drawn) return;
        console.error("Lookout: event chart failed", error);
        charts.innerHTML = '<p class="ev-note ev-wait">The chart could not be drawn.</p>';
      });
  }

  /** Make sure the open row is on the rows shown. */
  function reach(): void {
    if (!selected) return;
    const at = visibleEvents(section, view.season, view.sort).indexOf(selected);
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
      const sortButton = target.closest<HTMLButtonElement>("[data-sort]");
      if (sortButton) {
        view.sort = sortButton.dataset.sort as EventSort;
        view.limit = PAGE;
        reach();
        renderHead();
        renderList();
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
      view.season = select.value === "all" ? "all" : Number(select.value);
      view.limit = PAGE;
      // The open row follows the default rule within the water year picked:
      // the live event, or the newest. A row closed by hand stays closed.
      if (!kept.shut) {
        selected = defaultOpen(section, view.season);
        missing = false;
        writeUrl(null);
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
