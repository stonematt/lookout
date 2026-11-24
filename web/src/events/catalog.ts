/**
 * The Rain Events tab's top tile, the Event Catalog: its finding, the
 * standfirst, the stale-catalog note, the stat strip, the water year's gap
 * note and the folded source. On a phone it is the fixed top. Markup goes
 * through `html` (`web/src/dom.ts`). The page's headline and the scatter
 * (#666) open this tile, so they belong here or in a module it calls.
 *
 * - `renderCatalogTile`: the tile's markup and source fold, once per mount.
 * - `renderCatalogHead`: its finding, stats and gap note for the water year
 *   shown, after every pick.
 */
import { html } from "../dom.js";
import { inches } from "../format.js";
import { gapElement, isComparable } from "../gaps.js";
import type { DashboardSnapshot, RainEventRow } from "../snapshot.js";
import { staleCatalogNote } from "../stale.js";
import { gapReasons, sourceNote, tileFolds } from "../tiles.js";
import { stat } from "./detail.js";
import type { Season } from "./model.js";
import { listFinding, minutes, seasonNote, statsFor } from "./model.js";

/**
 * Fill the catalog tile: the finding and stats it renders later, the
 * standfirst, the stale-catalog note, and the source fold.
 *
 * @param host - The tile's body.
 * @param snapshot - The dashboard snapshot.
 */
export function renderCatalogTile(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const section = snapshot.rain_events;
  const count = section.events.length;
  const events = `${count} event${count === 1 ? "" : "s"}`;
  host.classList.add("ev-tile");
  host.insertAdjacentHTML(
    "beforeend",
    html`
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
        what: `Event Catalog, ${events}`,
        method: `The Event Catalog, ${events}, each storm the station recorded. Peak is the highest rate between 5-minute readings. A dashed quality chip marks a poor event or one with no archive data.`,
        gaps: gapReasons([[uncovered, "water year", "water years"]]),
      }),
    ),
  );
  const stale = staleCatalogNote(snapshot, "events");
  if (stale) {
    const note = host.querySelector<HTMLElement>(".ev-stale")!;
    note.textContent = stale;
    note.hidden = false;
  }
}

/**
 * The catalog tile's finding, stat strip and gap note for the water year
 * shown.
 *
 * @param panel - The tab's panel.
 * @param section - The snapshot's Rain Events section.
 * @param season - The water year shown, or "all".
 * @param rows - The events the list shows.
 */
export function renderCatalogHead(
  panel: HTMLElement,
  section: DashboardSnapshot["rain_events"],
  season: Season,
  rows: RainEventRow[],
): void {
  const all = section.events;
  const stats = statsFor(section, season);
  const oldest = all.length ? all[all.length - 1]!.start_ms : null;
  panel.querySelector(".ev-title")!.textContent = all.length
    ? listFinding(stats, season, oldest, rows)
    : "No rain events in the catalog yet";
  const strip = panel.querySelector<HTMLElement>(".ev-stats")!;
  const cells = [
    stat(String(stats.count), "events"),
    stat(inches(stats.total_in, { digits: 2 }), "rain in events"),
    stat(stats.median_in === null ? "n/a" : inches(stats.median_in, { digits: 2 }), "median event"),
    stat(
      stats.mean_duration_minutes === null ? "n/a" : minutes(stats.mean_duration_minutes),
      "mean length",
    ),
  ];
  strip.innerHTML = html`${all.length ? cells : ""}`;
  strip.hidden = !all.length;
  const note = panel.querySelector<HTMLElement>(".ev-season-gap")!;
  const text = seasonNote(section, season);
  note.hidden = text === null;
  note.replaceChildren();
  const year = season === "all" ? undefined : section.seasons.find((s) => s.water_year === season);
  const status = year?.status;
  if (status && !isComparable(status)) note.append(gapElement(status), " ");
  if (text !== null) note.append(text);
}
