/**
 * The Rain Events tab's list tile (DESIGN.md "Event list", board EventsBC):
 * the filter line (water year, sort, size: "2026–27", "Newest", "Showers and
 * up"), the line under it (how many events, which sizes are hidden, the days
 * missing), the list's heading, the event rows and the foot under them
 * ("Show N more", the size bars' key and the folded method). Markup goes
 * through `html` (`web/src/dom.ts`), which escapes every value.
 *
 * - `PAGE`: rows shown before "Show more".
 * - `storedSize`, `storeSize`: the size picked, kept in the browser across
 *   visits.
 * - `renderListTile`: the tile's markup, once per mount.
 * - `syncListHead`: the three selects, the count line, the missing-days
 *   note, the heading and the fold's sentence, after every pick.
 * - `eventRows` (drawn against a `RowContext`), `listFoot`, `emptyListLine`:
 *   the rows, the foot and the empty list's line, for `innerHTML`.
 */
import { html } from "../dom.js";
import type { Html } from "../dom.js";
import { inches, waterYearSpan } from "../format.js";
import { gapLabel, isComparable } from "../gaps.js";
import type { DashboardSnapshot, EventCategory, EventSeason, RainEventRow } from "../snapshot.js";
import { fold, hatchChip, tileFolds } from "../tiles.js";
import type { EventSize, EventSort, Season } from "./model.js";
import {
  DEFAULT_SIZE,
  SIZES,
  SORTS,
  categoryLabel,
  categoryStep,
  emptyList,
  listCount,
  listHeading,
  lowQuality,
  minutes,
  missingNote,
  peakText,
  qualityLabel,
  seasonLabel,
  seasonNote,
  sizeShare,
  startLabel,
} from "./model.js";

/** Rows shown before "Show more". */
export const PAGE = 20;

/** Where the browser keeps the size picked, across visits. */
const SIZE_KEY = "lookout.events.size";

/** The size picked on an earlier visit, or the default (showers and up). */
export function storedSize(): EventSize {
  try {
    const raw = localStorage.getItem(SIZE_KEY);
    const known = SIZES.find((s) => s.size === raw);
    if (known) return known.size;
  } catch {
    // No storage: the default.
  }
  return DEFAULT_SIZE;
}

/** Keep the size picked for the next visit, where the browser lets it. */
export function storeSize(size: EventSize): void {
  try {
    localStorage.setItem(SIZE_KEY, size);
  } catch {
    // Storage blocked: the pick lasts this page only.
  }
}

/**
 * A water year's option in the select: "2026–27", short so the three
 * filters share one line on a phone. A year the record does not cover
 * carries its reason in the title and `data-gap`; the count line under the
 * filter says it once picked ("partial year", "40 days missing").
 */
function yearOption(season: EventSeason): Html {
  const { status } = season;
  const year = season.water_year;
  const span = waterYearSpan(year);
  const label = seasonLabel(year);
  if (isComparable(status)) return html`<option value="${year}" title="${span}">${label}</option>`;
  const reason = gapLabel(status)!;
  return html`<option value="${year}" title="${`${span}: ${reason}`}" data-gap="${status.reason ?? "gap"}">${label}</option>`;
}

/**
 * One of the filter line's selects, styled as a pill, named for assistive
 * tech by its hidden label.
 *
 * @param name - The label, e.g. "Sort".
 * @param kind - Its class, `ev-<kind>-select`, which the tab listens on.
 * @param options - Its options' markup.
 * @returns The labelled select.
 */
function filterSelect(name: string, kind: string, options: readonly Html[]): Html {
  return html`<label class="ev-pill"><span class="ov-sr">${name}</span><select class="ev-select ev-${kind}-select">${options}</select></label>`;
}

/** A plain option: its value and its words. */
const option = (value: string, label: string): Html =>
  html`<option value="${value}">${label}</option>`;

/**
 * What the fold under the list says about how events are found and sized.
 * The category cutoffs are Python's and stay there: the fold names the
 * categories, not their numbers.
 */
const METHOD =
  "An event is a run of rain the station recorded, from the Event Catalog. Its size is the " +
  "category the refresh run gives it from its total and its peak rate: sprinkle, shower, " +
  "rain or storm. Peak is the highest rate between 5-minute readings. A dashed quality " +
  "chip marks a poor event or one with no archive data.";

/**
 * The list tile's markup under its kicker: the filter line, the count
 * line, the heading, the empty list and foot the renders fill, and the
 * fold with the method and the missing days' sentence.
 *
 * @param host - The tile's body.
 * @param section - The snapshot's Rain Events section.
 */
export function renderListTile(host: HTMLElement, section: DashboardSnapshot["rain_events"]): void {
  host.classList.add("ev-tile", "ev-list-tile");
  // The heading names the list, as every tile's finding follows its
  // kicker; the kicker is for assistive tech, the catalog tile above says
  // "Rain events" on screen.
  host.querySelector("[data-part=kicker]")!.classList.add("ov-sr");
  host.insertAdjacentHTML(
    "beforeend",
    html`
<h2 class="tile-finding ev-list-finding" data-part="finding"></h2>
<div class="ev-bar" role="group" aria-label="Filter the events">
  ${filterSelect("Water year", "year", [...section.seasons.map(yearOption), option("all", "All years")])}
  ${filterSelect(
    "Sort",
    "sort",
    SORTS.map((s) => option(s.sort, s.label)),
  )}
  ${filterSelect(
    "Size",
    "size",
    SIZES.map((s) => option(s.size, s.label)),
  )}
</div>
<p class="ev-count" role="status"><span class="ev-count-words"></span><span class="ev-missing" hidden></span></p>
<div class="ev-list"></div>
<div class="ev-foot"></div>`,
  );
  const method = document.createElement("p");
  method.className = "ov-source";
  method.textContent = METHOD;
  const gaps = document.createElement("p");
  gaps.className = "ov-source ev-season-gap";
  host.append(tileFolds(fold("How events are found", [method, gaps], "method")));
}

/** The pick the filter line shows. */
export interface FilterPick {
  season: Season;
  sort: EventSort;
  size: EventSize;
}

/**
 * Set the filter line to the visitor's pick, and the lines that follow it:
 * how many events, which sizes are hidden, the days missing, the heading,
 * and the fold's sentence on the days missing.
 *
 * @param panel - The tab's panel.
 * @param section - The snapshot's Rain Events section.
 * @param pick - The water year, order and size picked.
 * @param counts - Events listed, and those of the water year the size hides.
 */
export function syncListHead(
  panel: HTMLElement,
  section: DashboardSnapshot["rain_events"],
  pick: FilterPick,
  counts: { shown: number; hidden: number },
): void {
  const value = (selector: string, to: string) => {
    panel.querySelector<HTMLSelectElement>(selector)!.value = to;
  };
  value(".ev-year-select", String(pick.season));
  value(".ev-sort-select", pick.sort);
  value(".ev-size-select", pick.size);
  panel.querySelector(".ev-count-words")!.textContent = listCount(
    counts.shown,
    counts.hidden,
    pick.size,
  );
  const missing = panel.querySelector<HTMLElement>(".ev-missing")!;
  const note = missingNote(section, pick.season);
  missing.hidden = note === null;
  missing.replaceChildren();
  if (note) {
    missing.append(" · ");
    if (note.hatched) missing.append(hatchChip(note.text));
    else missing.append(note.text);
  }
  panel.querySelector(".ev-list-finding")!.textContent = listHeading(pick.sort);
  panel.querySelector(".ev-season-gap")!.textContent = seasonNote(section, pick.season) ?? "";
}

/**
 * The four-step glyph: four bars rising left to right, filled to the
 * event's category (one for a sprinkle, four for a storm). The word beside
 * it says the same, so the glyph is hidden from assistive tech.
 */
function glyph(category: EventCategory): Html {
  const step = categoryStep(category);
  const bars = [0, 1, 2, 3].map((i) => {
    const h = 5 + i * 3.67;
    return html`<rect x="${i * 6}" y="${(16 - h).toFixed(2)}" width="4" height="${h.toFixed(2)}" rx="1"${i < step ? html` class="on"` : ""}/>`;
  });
  return html`<svg class="ev-glyph" width="22" height="16" viewBox="0 0 22 16" aria-hidden="true" data-step="${step}">${bars}</svg>`;
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
  /** The page address that opens an event's row (#370). */
  href(startMs: number): string;
}

/**
 * A row's header: a link to the page with its event open (#370), so
 * ctrl-click opens it in a new tab; a plain click opens and closes it in
 * place.
 */
function rowHead(e: RainEventRow, open: boolean, context: RowContext): Html {
  const { latest, now, most } = context;
  const mark = e.live
    ? html`<span class="ev-live">Live</span>`
    : e.start_ms === latest
      ? html`<span class="ev-latest">latest</span>`
      : "";
  const peak = peakText(e);
  const chip = html`<span class="ev-chip${lowQuality(e) ? " low" : ""}">${qualityLabel(e)}</span>`;
  const share = (sizeShare(e, most) * 100).toFixed(1);
  const word = categoryLabel(e);
  const kind =
    e.category && word ? html`<span class="ev-kind">${glyph(e.category)}${word}</span> · ` : "";
  return html`<h3 class="ev-row-head"><a class="ev-open" href="${context.href(e.start_ms)}" data-start="${e.start_ms}" aria-expanded="${open}" aria-controls="ev-body-${e.start_ms}">
  <span class="ev-date">${when(startLabel(e.start_ms, now))}${mark}</span>
  <span class="ev-total">${e.total_in.toFixed(2)}<small> in</small></span>
  <span class="ev-meta">${kind}${minutes(e.duration_minutes)} · peak ${peak} ${chip}</span>
  <span class="ev-size" aria-hidden="true"><span style="width: ${share}%"></span></span>
</a></h3>`;
}

/**
 * The rows shown, each an `article` with its header link; the open one
 * also holds the empty region its event fills.
 *
 * @param rows - The events shown, in order.
 * @param context - The open row, the latest, the clock, the largest total
 *   and each row's address.
 * @returns The rows' markup.
 */
export function eventRows(rows: readonly RainEventRow[], context: RowContext): Html {
  const { openStart, now } = context;
  return html`${rows.map((e) => {
    const open = openStart === e.start_ms;
    return html`<article class="ev-row"${open ? html` data-open` : ""}${e.live ? html` data-live` : ""} data-start="${e.start_ms}">
${rowHead(e, open, context)}
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
 * @param hidden - Events the size filter hides.
 * @returns Its markup.
 */
export function emptyListLine(season: Season, hidden: number): Html {
  return html`<p class="ev-empty" role="status">${emptyList(season, hidden)}</p>`;
}
