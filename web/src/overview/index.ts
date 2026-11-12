/**
 * The Overview tab (spec #460): the now tile, the feature tile, then the
 * mosaic's ten tiles (#472, #473).
 *
 * Layout, the same order on every width (DESIGN.md "Layout", README
 * decision 10): on a phone the now tile and the feature take the full
 * width and the mosaic sits two to a row; on the four-column grid the now
 * tile takes the first row, the feature two columns and two rows at the
 * left, the first four mosaic tiles the 2×2 beside it, and the rest one
 * column each, row first (Daylight and Inside alone on the last row). The Overview has no
 * finding rows: the accordion is gone (the Rain, Solar and Rain events tabs
 * keep theirs).
 *
 * A section the snapshot does not draw (`TileSection.shown`) is skipped and
 * the later ones move up: the feature without a `feature` (a snapshot
 * published before Release A, ADR-0008), a mosaic tile the record does not
 * cover yet (`record_depth.tiles`, #470), Inside on a station with no
 * indoor sensor, Daylight on one with no location.
 *
 * No Overview tile folds a source (prototype decision 22): under the grid,
 * one sources footer (`sources.ts`) lists each drawn section's `source`.
 *
 * Each tile carries the id `overview-<section>`, which the feature tile's
 * link reaches. No tile carries an accessible name of its own (#472): each
 * tile's content names it, so a screen reader hears its finding once.
 *
 * A section is an `OverviewSection` (a `TileSection`, `web/src/tiles.ts`).
 * A mosaic tile is a `MosaicTile` definition in `mosaic-tiles.ts`, drawn by
 * `mosaicSection` (`mosaic.ts`): to add one, add its definition to
 * `MOSAIC_TILES` at its place.
 *
 * The sections that left the Overview with the accordion (temperature across
 * periods, now against this date, the ten-day list, the almanac and the
 * rolling-window rain) stay in their modules as the zooms' content.
 *
 * Each tile's link opens its zoom at `#overview/<slug>` (#474): after every
 * render the zoom shell (`zoom.ts`) draws the zoom the route names, from
 * the registry in `zooms/index.ts`.
 */
import type { DashboardSnapshot } from "../snapshot.js";
import type { TileSection } from "../tiles.js";
import { renderTiles } from "../tiles.js";
import { featureTile } from "./feature.js";
import { firstScreen } from "./first-screen.js";
import { mosaicSection } from "./mosaic.js";
import { MOSAIC_TILES } from "./mosaic-tiles.js";
import { sourcesFooter } from "./sources.js";
import { drawZoom } from "./zoom.js";

/** One tile of the Overview. */
export type OverviewSection = TileSection;

/** The Overview's sections, in the mosaic's fixed order. */
export const SECTIONS: readonly OverviewSection[] = [
  firstScreen, // #468 the now tile
  featureTile, // #469 the feature tile
  ...MOSAIC_TILES.map(mosaicSection), // #472, #473 the ten tiles
];

/**
 * Render the Overview into its tab panel, replacing what was there.
 *
 * @param panel - The Overview tab's panel.
 * @param snapshot - The dashboard snapshot.
 * @param sections - The sections to render, in order.
 */
export function renderOverview(
  panel: HTMLElement,
  snapshot: DashboardSnapshot,
  sections: readonly OverviewSection[] = SECTIONS,
): void {
  const grid = renderTiles(panel, snapshot, sections, { unnamed: true });
  grid.classList.add("mosaic-grid");
  for (const tile of grid.querySelectorAll<HTMLElement>("[data-section]")) {
    tile.id = `overview-${tile.dataset.section}`;
  }
  const footer = sourcesFooter(snapshot, sections);
  if (footer) panel.append(footer);
  drawZoom(panel, snapshot);
}
