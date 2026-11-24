/**
 * The Rain Events tab's list tile (DESIGN.md "Event list"): the filter bar
 * (a water-year select, the order and the category toggles), the event rows
 * and the foot under them ("Show N more" and the size bars' key). Markup
 * goes through `html` (`web/src/dom.ts`), which escapes every value.
 *
 * - `PAGE`: rows shown before "Show more".
 * - `storedCategories`, `storeCategories`: the categories picked, kept in
 *   the browser across visits.
 * - `renderListTile`: the tile's markup, once per mount.
 * - `syncFilterBar`: the bar's pressed buttons, the select and the list's
 *   heading for assistive tech, after every pick.
 * - `eventRows`, `listFoot`, `emptyList`: the rows, the foot and the empty
 *   list's line, for `innerHTML`.
 */
import { html } from "../dom.js";
import type { Html } from "../dom.js";
import { inches, inchesPerHour, waterYearSpan } from "../format.js";
import { gapLabel, isComparable } from "../gaps.js";
import type { DashboardSnapshot, EventCategory, EventSeason, RainEventRow } from "../snapshot.js";
import type { EventSort, Season } from "./model.js";
import {
  CATEGORIES,
  DEFAULT_CATEGORIES,
  SORTS,
  categoryLabel,
  emptyList as emptyListText,
  listHeading,
  lowQuality,
  minutes,
  qualityLabel,
  sizeShare,
  startLabel,
} from "./model.js";

/** Rows shown before "Show more". */
export const PAGE = 20;

/** Where the browser keeps the categories picked, across visits. */
const CATEGORIES_KEY = "lookout.events.categories";

/** The categories picked on an earlier visit, or the default. */
export function storedCategories(): Set<EventCategory> {
  try {
    const raw = localStorage.getItem(CATEGORIES_KEY);
    if (raw !== null) {
      const known = new Set<string>(CATEGORIES.map((c) => c.category));
      const picked = (JSON.parse(raw) as unknown[]).filter(
        (c): c is EventCategory => typeof c === "string" && known.has(c),
      );
      return new Set(picked);
    }
  } catch {
    // No storage, or a value this version cannot read: the default.
  }
  return new Set(DEFAULT_CATEGORIES);
}

/** Keep the categories picked for the next visit, where the browser lets it. */
export function storeCategories(categories: ReadonlySet<EventCategory>): void {
  try {
    localStorage.setItem(CATEGORIES_KEY, JSON.stringify([...categories]));
  } catch {
    // Storage blocked: the pick lasts this page only.
  }
}

/**
 * A water year's option in the select. A year the record does not cover
 * reads "<year>, partial" ("not archived" for one the archive does not
 * reach yet), short so the select stays beside the sort on a
 * phone, with its reason in the title; a ≈ one is the year alone (a ≈ is
 * never marked as a gap). The catalog tile's note says why once picked.
 */
function yearOption(season: EventSeason): Html {
  const { status } = season;
  const year = season.water_year;
  const span = waterYearSpan(season.water_year);
  if (isComparable(status)) return html`<option value="${year}" title="${span}">${year}</option>`;
  const reason = gapLabel(status)!;
  const word = status.reason === "not_archived" ? "not archived" : "partial";
  return html`<option value="${year}" title="${`${span}: ${reason}`}" data-gap="${status.reason ?? "gap"}">${year}, ${word}</option>`;
}

/**
 * The list tile's markup under its kicker: the heading for assistive tech,
 * the filter bar, and the empty list and foot the renders fill.
 *
 * @param host - The tile's body.
 * @param section - The snapshot's Rain Events section.
 */
export function renderListTile(host: HTMLElement, section: DashboardSnapshot["rain_events"]): void {
  host.classList.add("ev-tile", "ev-list-tile");
  // The filter bar heads the list; the kicker and finding are named
  // for assistive tech, the catalog tile above says it on screen.
  host.querySelector("[data-part=kicker]")!.classList.add("ov-sr");
  const sorts = SORTS.map(
    ({ sort, label }) => html`<button type="button" data-sort="${sort}">${label}</button>`,
  );
  const kinds = CATEGORIES.map(
    ({ category, label }) =>
      html`<button type="button" data-category="${category}">${label}</button>`,
  );
  host.insertAdjacentHTML(
    "beforeend",
    html`
<h2 class="tile-finding ev-list-finding ov-sr" data-part="finding"></h2>
<div class="ev-bar" role="group" aria-label="Filter and sort the events">
  <label class="ev-year"><span class="ov-sr">Water year</span><select class="ev-select">
    <option value="all">All years</option>${section.seasons.map(yearOption)}
  </select></label>
  <div class="ev-seg ev-sort" role="group" aria-label="Order">${sorts}</div>
  <div class="ev-seg ev-kinds" role="group" aria-label="Categories">${kinds}</div>
</div>
<div class="ev-list"></div>
<div class="ev-foot"></div>`,
  );
}

/**
 * Set the filter bar to the visitor's pick: the water year, the pressed
 * order and categories, and the list's heading.
 *
 * @param panel - The tab's panel.
 * @param pick - The water year, order and categories picked.
 * @param shown - How many events the pick lists.
 */
export function syncFilterBar(
  panel: HTMLElement,
  pick: { season: Season; sort: EventSort; categories: ReadonlySet<EventCategory> },
  shown: number,
): void {
  panel.querySelector(".ev-list-finding")!.textContent = listHeading(shown, pick.season, pick.sort);
  panel.querySelector<HTMLSelectElement>(".ev-select")!.value = String(pick.season);
  for (const b of panel.querySelectorAll<HTMLButtonElement>("[data-sort]")) {
    b.setAttribute("aria-pressed", String(b.dataset.sort === pick.sort));
  }
  for (const b of panel.querySelectorAll<HTMLButtonElement>("[data-category]")) {
    const on = pick.categories.has(b.dataset.category as EventCategory);
    b.setAttribute("aria-pressed", String(on));
  }
}

/**
 * An event's start as two unbreakable parts, date and time, so a narrow
 * row breaks it between them: `Jan 8, 2025,` and `2:00 pm`.
 */
function when(label: string): Html {
  const cut = label.lastIndexOf(", ") + 1;
  if (cut === 0) return html`<span class="ev-when">${label}</span>`;
  return html`<span><span class="ev-when">${label.slice(0, cut)}</span> <span class="ev-when">${label.slice(cut + 1)}</span></span>`;
}

/** What the rows are drawn against. */
export interface RowContext {
  /** The open row's start, or null. */
  openStart: number | null;
  /** The newest event's start, which carries "latest" when none is live. */
  latest: number | null;
  /** The browser's clock. */
  now: number;
  /** The catalog's largest event total, the size bars' full track. */
  most: number;
}

/** A closed row's header: the button that opens and closes it. */
function rowHead(
  e: RainEventRow,
  open: boolean,
  now: number,
  most: number,
  latest: number | null,
): Html {
  const mark = e.live
    ? html`<span class="ev-live">Live</span>`
    : e.start_ms === latest
      ? html`<span class="ev-latest">latest</span>`
      : "";
  const peak =
    e.peak_rate_in_hr === null || !e.has_curve ? "n/a" : inchesPerHour(e.peak_rate_in_hr);
  const chip = html`<span class="ev-chip${lowQuality(e) ? " low" : ""}">${qualityLabel(e)}</span>`;
  const share = (sizeShare(e, most) * 100).toFixed(1);
  const kind = categoryLabel(e);
  return html`<h3 class="ev-row-head"><button type="button" class="ev-open" data-start="${e.start_ms}" aria-expanded="${open}" aria-controls="ev-body-${e.start_ms}">
  <span class="ev-date">${when(startLabel(e.start_ms, now))}${mark}</span>
  <span class="ev-total">${e.total_in.toFixed(2)}<small> in</small></span>
  <span class="ev-meta">${kind ? `${kind} · ` : ""}${minutes(e.duration_minutes)} · peak ${peak} ${chip}</span>
  <span class="ev-size" aria-hidden="true"><span style="width: ${share}%"></span></span>
</button></h3>`;
}

/**
 * The rows shown, each an `article` with its header button; the open one
 * also holds the empty region its event fills.
 *
 * @param rows - The events shown, in order.
 * @param context - The open row, the latest, the clock and the largest total.
 * @returns The rows' markup.
 */
export function eventRows(rows: readonly RainEventRow[], context: RowContext): Html {
  const { openStart, latest, now, most } = context;
  return html`${rows.map((e) => {
    const open = openStart === e.start_ms;
    return html`<article class="ev-row"${open ? html` data-open` : ""}${e.live ? html` data-live` : ""} data-start="${e.start_ms}">
${rowHead(e, open, now, most, latest)}
${open ? html`<div class="ev-body" id="ev-body-${e.start_ms}" role="region" aria-label="${`The event of ${startLabel(e.start_ms, now)}`}"></div>` : ""}
</article>`;
  })}`;
}

/**
 * The foot under the rows: "Show N more" while rows are left, and the key
 * to the size bars.
 *
 * @param more - Rows not shown yet.
 * @param most - The catalog's largest event total.
 * @returns The foot's markup.
 */
export function listFoot(more: number, most: number): Html {
  return html`${more > 0 ? html`<button type="button" class="ev-more">Show ${Math.min(PAGE * 2, more)} more</button>` : ""}
<p class="ev-key">Each bar under a row is that event's total against the wettest on record, ${inches(most, { digits: 2 })}. From the Event Catalog.</p>`;
}

/**
 * The line an empty list shows.
 *
 * @param season - The water year shown, or "all" for an empty catalog.
 * @param hidden - Events the category pick hides.
 * @returns Its markup.
 */
export function emptyList(season: Season, hidden: number): Html {
  return html`<p class="ev-empty" role="status">${emptyListText(season, hidden)}</p>`;
}
