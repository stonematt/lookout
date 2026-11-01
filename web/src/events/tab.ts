/**
 * The Rain Events tab: the Event Catalog's events from the snapshot, one
 * opened with its accumulation and rate charts, and the selected event in
 * the URL (`?event=<start_ms>`), so a link opens the same event.
 *
 * - The list renders from the snapshot's `rain_events`; a water-year picker
 *   and a sort reshape it in the browser. A water year the record does not
 *   cover is hatched in the picker, and a note beside the summary says the
 *   list can miss its events.
 * - The live event's detail renders from the snapshot's `open_curve`, which
 *   the snapshot holds only for the live event (`active_rain_event`, the
 *   hero's decision); without one it says its readings are "not in the
 *   snapshot yet". Any other event's renders from its curve file (`/event`).
 *   An event the catalog still holds open after the rain has stopped is not
 *   live and has no file yet (the file is written once the catalog closes
 *   it), so it reads "not published yet" until then. An event with no
 *   archive data shows its totals only, with "n/a: gap in record".
 * - The event is keyed by its start, which is stable across refresh runs,
 *   so a link opened while it is live still opens it once it has completed.
 */
import "./events.css";

import { formatStamp, inches, inchesPerHour, waterYearSpan } from "../format.js";
import { ReadError, fetchEventCurve } from "../load.js";
import { escapeHtml as esc } from "../dom.js";
import { gapElement, gapLabel, isComparable } from "../gaps.js";
import { staleCatalogNote } from "../stale.js";
import type { DashboardSnapshot, EventCurve, EventSeason, RainEventRow } from "../snapshot.js";
import type { EventSort, Season } from "./model.js";
import {
  eventFinding,
  listFinding,
  lowQuality,
  minutes,
  qualityLabel,
  scrollEdges,
  seasonNote,
  selectedStart,
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
  /** Draw the two charts. */
  draw(
    hosts: { accumulation: HTMLElement; rate: HTMLElement },
    curve: EventCurve,
    row: RainEventRow,
  ): Promise<void>;
  /** The browser's clock. */
  nowMs(): number;
}

/** The real sources: `/event` and Plotly, imported when first needed. */
export function browserDeps(): EventsDeps {
  return {
    fetchCurve: (startMs) => fetchEventCurve(startMs, location.search),
    draw: async (hosts, curve, row) => {
      const { drawEventCharts } = await import("./charts.js");
      await drawEventCharts(hosts, curve, row);
    },
    nowMs: () => Date.now(),
  };
}

/**
 * An event's start as two unbreakable parts, date and time, so a narrow
 * column breaks it between them: `Jan 8, 2025,` and `2:00 pm`.
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
 * A water year's picker button. One the record does not cover carries the
 * gap hatch along its foot and its reason in the title and spoken; a ≈ one
 * says so in its title only (a ≈ is never hatched).
 */
function yearButton(season: EventSeason): string {
  const { status } = season;
  const year = esc(season.water_year);
  const span = waterYearSpan(season.water_year);
  if (status.status === "valid") {
    return `<button type="button" data-season="${year}" title="${esc(span)}">${year}</button>`;
  }
  if (isComparable(status)) {
    const title = `${span}: ${status.missing_days} of ${status.days} days missing from the record`;
    return `<button type="button" data-season="${year}" title="${esc(title)}">${year}</button>`;
  }
  const reason = gapLabel(status)!;
  return `<button type="button" data-season="${year}" data-gap="${esc(status.reason ?? "gap")}" title="${esc(`${span}: ${reason}`)}">${year}<span class="ev-seg-gap gap-hatch" aria-hidden="true"></span><span class="sr-only">, ${esc(reason)}</span></button>`;
}

/** Each panel's listeners from its last mount, dropped by the next one. */
const mounted = new WeakMap<HTMLElement, AbortController>();

/**
 * Render the Rain Events tab into its panel.
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
  mounted.get(panel)?.abort();
  const listening = new AbortController();
  mounted.set(panel, listening);
  const section = snapshot.rain_events;
  const all = section.events;
  const asked = selectedStart(location.search);
  let season: Season = "all";
  let sort: EventSort = "newest";
  let limit = PAGE;
  let selected: RainEventRow | null =
    all.find((e) => e.start_ms === asked) ?? all[0] ?? null;
  let missing = asked !== null && !all.some((e) => e.start_ms === asked);
  let drawn = 0;

  panel.innerHTML = `
<div class="ev">
  <header class="ev-head">
    <h2 class="ev-title"></h2>
    <p class="standfirst">Each storm the station recorded, from the Event Catalog. Open one to see how its rain built up and how hard it fell.</p>
    <p class="ev-note ev-stale" role="status" hidden></p>
  </header>
  <div class="ev-controls">
    <div class="ev-seg" role="group" aria-label="Water year"></div>
    <div class="ev-seg" role="group" aria-label="Order">
      <button type="button" data-sort="newest">Newest</button>
      <button type="button" data-sort="wettest">Wettest</button>
    </div>
  </div>
  <dl class="ev-stats"></dl>
  <p class="ev-note ev-season-gap" role="note" hidden></p>
  <div class="ev-grid">
    <article class="ev-detail" aria-live="polite"></article>
    <div class="ev-list"></div>
  </div>
</div>`;
  const $ = <T extends HTMLElement>(selector: string) => panel.querySelector<T>(selector)!;
  const seasons = $(".ev-seg[aria-label='Water year']");
  const stale = staleCatalogNote(snapshot, "events");
  if (stale) {
    $(".ev-stale").textContent = stale;
    $(".ev-stale").hidden = false;
  }

  if (!all.length) {
    $(".ev-title").textContent = "No rain events in the catalog yet";
    $(".ev-controls").hidden = true;
    $(".ev-grid").hidden = true;
    return;
  }

  seasons.innerHTML = [
    `<button type="button" data-season="all">All</button>`,
    ...section.seasons.map(yearButton),
  ].join("");
  // The control scrolls in place at phone width; the edge with years hidden
  // past it fades (events.css), so the scroll is not a secret.
  const cue = () => {
    seasons.dataset.more = scrollEdges(seasons.scrollLeft, seasons.scrollWidth, seasons.clientWidth);
  };
  seasons.addEventListener("scroll", cue, { passive: true });
  // Also fires once the tab is first shown, when the control gets its width.
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(cue).observe(seasons);
  cue();

  function renderHead(): void {
    const stats = statsFor(section, season);
    const oldest = all.length ? all[all.length - 1]!.start_ms : null;
    $(".ev-title").textContent = listFinding(stats, season, oldest, visibleEvents(section, season, sort));
    $(".ev-stats").innerHTML = [
      stat(String(stats.count), "events"),
      stat(inches(stats.total_in, { digits: 2 }), "rain in events"),
      stat(stats.median_in === null ? "n/a" : inches(stats.median_in, { digits: 2 }), "median event"),
      stat(
        stats.mean_duration_minutes === null ? "n/a" : minutes(stats.mean_duration_minutes),
        "mean length",
      ),
    ].join("");
    const note = $(".ev-season-gap");
    const text = seasonNote(section, season);
    note.hidden = text === null;
    note.replaceChildren();
    const year = season === "all" ? undefined : section.seasons.find((s) => s.water_year === season);
    const status = year?.status;
    if (status && !isComparable(status)) note.append(gapElement(status), " ");
    if (text !== null) note.append(text);
    for (const b of panel.querySelectorAll<HTMLButtonElement>("[data-season]")) {
      b.setAttribute("aria-pressed", String(b.dataset.season === String(season)));
    }
    for (const b of panel.querySelectorAll<HTMLButtonElement>("[data-sort]")) {
      b.setAttribute("aria-pressed", String(b.dataset.sort === sort));
    }
  }

  function renderList(): void {
    const rows = visibleEvents(section, season, sort);
    const shown = rows.slice(0, limit);
    const now = deps.nowMs();
    const body = shown
      .map((e) => {
        const current = selected?.start_ms === e.start_ms ? ' aria-current="true"' : "";
        const live = e.live ? '<span class="ev-live">Live</span> ' : "";
        const href = `${withEvent(location.search, e.start_ms)}#events`;
        const peak = e.peak_rate_in_hr === null || !e.has_curve ? "n/a" : e.peak_rate_in_hr.toFixed(2);
        const chip = `<span class="ev-chip${lowQuality(e) ? " low" : ""}">${esc(qualityLabel(e))}</span>`;
        // At phone width the Quality column drops; a low one folds into the row.
        const fold = lowQuality(e) ? ` <span class="ev-chip low ev-chip-fold">${esc(qualityLabel(e))}</span>` : "";
        return `<tr class="${e.live ? "pin" : ""}"${current}>
  <th scope="row"><a class="ev-open" href="${esc(href)}" data-start="${esc(e.start_ms)}">${live}${when(startLabel(e.start_ms, now))}</a>${fold}</th>
  <td>${esc(minutes(e.duration_minutes))}</td>
  <td><b>${e.total_in.toFixed(2)}</b></td>
  <td class="ev-peak">${peak}</td>
  <td class="ev-quality">${chip}</td>
</tr>`;
      })
      .join("");
    const more =
      rows.length > limit
        ? `<button type="button" class="ev-more">Show ${Math.min(PAGE * 2, rows.length - limit)} more of ${rows.length}</button>`
        : "";
    $(".ev-list").innerHTML = `
<table class="ev-table">
  <caption class="sr-only">Rain events, ${sort === "newest" ? "newest" : "wettest"} first${rows.some((e) => e.live) ? ", the live event pinned on top" : ""}</caption>
  <thead><tr><th scope="col">Start</th><th scope="col">Length</th><th scope="col">Total (in)</th><th scope="col" class="ev-peak">Peak (in/hr)</th><th scope="col" class="ev-quality">Quality</th></tr></thead>
  <tbody>${body}</tbody>
</table>
${more}
<p class="ev-source">Source: Event Catalog, ${all.length} event${all.length === 1 ? "" : "s"}. Peak is the highest rate between 5-minute readings. A dashed quality chip marks a poor event or one with no archive data.</p>`;
  }

  function renderDetail(): void {
    const host = $(".ev-detail");
    const row = selected;
    if (!row) {
      host.innerHTML = "";
      return;
    }
    const ticket = ++drawn;
    const span = row.live
      ? `Began ${formatStamp(row.start_ms)}; the newest reading is ${formatStamp(row.end_ms)}.`
      : `Began ${formatStamp(row.start_ms)}; its last reading with event rain was ${formatStamp(row.end_ms)}.`;
    const quality =
      row.quality === "invalid"
        ? "No archive readings."
        : `Quality: ${row.quality}${row.completeness_pct === null ? "" : `, ${Math.round(row.completeness_pct)}% of readings`}${row.max_gap_minutes ? `, longest gap ${minutes(row.max_gap_minutes)}` : ""}.`;
    const note = missing
      ? '<p class="ev-note">The linked event is not in the catalog; showing the newest one instead.</p>'
      : "";
    missing = false;
    host.innerHTML = `
${note}
<p class="ev-kicker">${row.live ? '<span class="ev-live">Live</span> ' : ""}Water year ${esc(row.water_year)}</p>
<h3 class="ev-finding">${esc(eventFinding(row, all.length))}</h3>
<p class="standfirst">${esc(span)} ${esc(quality)}</p>
<dl class="ev-facts">
  ${stat(inches(row.total_in, { digits: 2 }), row.live ? "so far (live)" : "total")}
  ${stat(minutes(row.duration_minutes), "length")}
  ${stat(row.peak_rate_in_hr === null || !row.has_curve ? "n/a" : inchesPerHour(row.peak_rate_in_hr), "peak rate")}
</dl>
<div class="ev-charts"></div>
<p class="ev-source"></p>`;
    const charts = host.querySelector<HTMLElement>(".ev-charts")!;
    const source = host.querySelector<HTMLElement>(".ev-source")!;

    if (!row.has_curve) {
      charts.innerHTML =
        '<p class="ev-gap gap-hatch" role="note">n/a: gap in record<span class="sr-only">. The archive has no readings for this event, so only its catalog totals are shown.</span></p>';
      source.textContent = "Source: Event Catalog. The archive holds no readings in this event's time range.";
      return;
    }
    source.textContent = row.live
      ? "Source: live station for the newest reading and the total; station archive for the rest. 5-minute readings, station time."
      : "Source: station archive, 5-minute readings, station time. Rate is the rise in daily rain between readings.";

    const show = (curve: EventCurve) => {
      if (ticket !== drawn) return;
      charts.innerHTML = `
<p class="ev-chart-label">Rain so far</p>
<div class="ev-chart ev-acc" role="img" aria-label="${esc(`Rain so far in this event, rising to ${inches(row.total_in, { digits: 2 })}`)}"></div>
<p class="ev-chart-label">Rate between readings</p>
<div class="ev-chart ev-rate" role="img" aria-label="${esc(row.peak_rate_in_hr === null ? "Rain rate between readings" : `Rain rate between readings, peaking at ${inchesPerHour(row.peak_rate_in_hr)}`)}"></div>`;
      deps
        .draw(
          {
            accumulation: charts.querySelector<HTMLElement>(".ev-acc")!,
            rate: charts.querySelector<HTMLElement>(".ev-rate")!,
          },
          curve,
          row,
        )
        .catch((error: unknown) => {
          console.error("Lookout: event charts failed", error);
          charts.innerHTML = '<p class="ev-note">The charts could not be drawn.</p>';
        });
    };

    const open = section.open_curve;
    if (open && open.start_ms === row.start_ms) {
      show(open);
      return;
    }
    if (row.live) {
      charts.innerHTML = '<p class="ev-note">This event\'s readings are not in the snapshot yet.</p>';
      return;
    }
    charts.innerHTML = '<p class="ev-note">Loading the event\'s readings…</p>';
    deps.fetchCurve(row.start_ms).then(show, (error: unknown) => {
      if (ticket !== drawn) return;
      if (error instanceof ReadError && error.status === 404) {
        charts.innerHTML =
          '<p class="ev-note">This event\'s readings are not published yet. The refresh run adds them within the hour.</p>';
        return;
      }
      console.error("Lookout: event curve could not be read", error);
      charts.innerHTML =
        '<p class="ev-note">The event\'s readings could not be loaded. <button type="button" class="ev-retry">Try again</button></p>';
      charts.querySelector(".ev-retry")?.addEventListener("click", renderDetail);
    });
  }

  function open(startMs: number): void {
    const row = all.find((e) => e.start_ms === startMs);
    if (!row) return;
    selected = row;
    history.replaceState(
      history.state,
      "",
      `${location.pathname}${withEvent(location.search, startMs)}${location.hash}`,
    );
    renderList();
    renderDetail();
    const detail = $(".ev-detail");
    if (detail.getBoundingClientRect().top < 0) detail.scrollIntoView({ block: "start" });
  }

  panel.addEventListener("click", (event) => {
    const target = event.target as Element;
    const link = target.closest<HTMLAnchorElement>("a.ev-open");
    if (link) {
      if ((event as MouseEvent).metaKey || (event as MouseEvent).ctrlKey) return;
      event.preventDefault();
      open(Number(link.dataset.start));
      return;
    }
    const seasonButton = target.closest<HTMLButtonElement>("[data-season]");
    if (seasonButton) {
      const value = seasonButton.dataset.season!;
      season = value === "all" ? "all" : Number(value);
      limit = PAGE;
      renderHead();
      renderList();
      // A year tapped half under the fade comes whole into view.
      seasonButton.scrollIntoView({ block: "nearest", inline: "nearest" });
      return;
    }
    const sortButton = target.closest<HTMLButtonElement>("[data-sort]");
    if (sortButton) {
      sort = sortButton.dataset.sort as EventSort;
      limit = PAGE;
      renderHead();
      renderList();
      return;
    }
    if (target.closest(".ev-more")) {
      limit += PAGE * 2;
      renderList();
    }
  }, { signal: listening.signal });

  renderHead();
  renderList();
  renderDetail();
}
