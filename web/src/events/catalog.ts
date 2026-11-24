/**
 * The Rain Events tab's top tile, the Event Catalog. On a phone it is the
 * fixed top. Markup goes through `html` (`web/src/dom.ts`).
 *
 * The page opens with a picture (#666, board EventsBC): the headline, "10
 * events this water year · 3 in the last 7 days" (`RainEvents.counts`), is
 * the tile's finding, then the scatter of four water years' events by size
 * (`scatter.ts`), the stale-catalog note and the folded source. The filter
 * line drives only the list (#239, board EventsBC), so nothing here follows
 * the pick; the water year's missing days are the list's (`list.ts`). A
 * snapshot written before the counts (N−1) opens as before: the pick's
 * finding as the tile's finding, the standfirst, and the stat strip.
 *
 * - `renderCatalogTile`: the tile's markup, scatter and source fold, once
 *   per mount.
 * - `renderCatalogHead`: on an N−1 snapshot, the pick's finding and stats
 *   for the water year shown, after every pick.
 */
import { html } from "../dom.js";
import type { Html } from "../dom.js";
import { inches } from "../format.js";
import { isComparable } from "../gaps.js";
import type { DashboardSnapshot, EventCounts, RainEventRow } from "../snapshot.js";
import { staleCatalogNote } from "../stale.js";
import { gapReasons, sourceNote, statCell as stat, tileFolds } from "../tiles.js";
import type { Season } from "./model.js";
import { listFinding, minutes, statsFor } from "./model.js";
import { drawScatter, scatterInput, scatterMarkup } from "./scatter.js";

/**
 * The headline: the count in Fact type beside its two lines, read as one
 * sentence, "10 events this water year · 3 in the last 7 days".
 *
 * @param counts - The snapshot's counts.
 * @returns The tile's finding, an `h2`.
 */
export function headline(counts: EventCounts): Html {
  const n = counts.events;
  return html`<h2 class="tile-finding ev-headline" data-part="finding"><span class="ev-headline-n">${n}</span> <span class="ev-headline-words"><span>${n === 1 ? "event" : "events"} this water year</span><span class="sr-only"> · </span><span>${counts.last_7_days} in the last 7 days</span></span></h2>`;
}

/**
 * Fill the catalog tile: the headline and scatter (or, N−1, the finding and
 * stats it renders later and the standfirst), the stale-catalog note, and
 * the source fold.
 *
 * @param host - The tile's body.
 * @param snapshot - The dashboard snapshot.
 */
export function renderCatalogTile(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const section = snapshot.rain_events;
  const count = section.events.length;
  const events = `${count} event${count === 1 ? "" : "s"}`;
  host.classList.add("ev-tile");
  const scatter = scatterInput(section);
  const head = section.counts
    ? html`${headline(section.counts)}${scatter ? scatterMarkup(scatter) : ""}
<p class="ev-note ev-stale" role="status" hidden></p>`
    : html`<h2 class="tile-finding ev-title" data-part="finding"></h2>
<p class="standfirst" data-part="standfirst">Each storm the station recorded, from the Event Catalog. Open one to see how its rain built up and how hard it fell.</p>
<p class="ev-note ev-stale" role="status" hidden></p>
<dl class="ev-stats"></dl>`;
  host.insertAdjacentHTML("beforeend", html`${head}`);
  const chart = host.querySelector<HTMLElement>(".ev-sc-chart");
  if (chart && scatter) drawScatter(chart, scatter);
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
 * The catalog tile's finding and stat strip for the water year shown, on a
 * snapshot written before the counts (N−1); a tile with the headline has
 * neither, and this does nothing.
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
  const title = panel.querySelector(".ev-title");
  if (!title) return;
  const all = section.events;
  const stats = statsFor(section, season);
  const oldest = all.length ? all[all.length - 1]!.start_ms : null;
  title.textContent = all.length
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
}
