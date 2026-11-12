/**
 * The Overview tab: a grid of tiles (spec #332, DESIGN.md "Layout"). The now
 * tile and the feature tile first, the phone's fixed top; then temperature
 * across periods, now against this date, the last ten days, the rain almanac
 * and the rolling-window rain table, then the quick facts (wind, pressure,
 * air). In the four-column grid the now tile takes the first row, the
 * feature half the second beside temperature in context, now against this
 * date and the last ten days half the third each, the almanac three columns
 * beside the rolling windows, and the quick facts a row of one-column tiles
 * whose fourth cell is sun and moon's. On a phone the quick facts sit after
 * the finding rows, two to a row. A snapshot without a feature (published
 * before Release A, ADR-0008) draws no feature tile.
 *
 * History grows (#470): rows the record does not cover are not drawn, one
 * line under the window chart names the next unlock, and the rolling-window
 * rain tile is not drawn while the record covers none of its windows.
 *
 * Each tile or finding row carries the id `overview-<section>`, which the
 * feature tile's link reaches while the Overview has no zooms (Release B1).
 *
 * A section is a module under `web/src/overview/` exporting an
 * `OverviewSection` (a `TileSection`, `web/src/tiles.ts`): its id, label,
 * kicker, span, whether it is in the phone's fixed top, and its render. To
 * add one, import it here and add it to `SECTIONS` at its place: one import
 * line and one list line. Each section renders into its own tile, already
 * attached to the page and holding its kicker, so a chart can measure its
 * width when it draws.
 *
 * Release 1 of the Mosaic Overview (#458): no Overview tile carries a
 * standfirst or an archive lag chip (the header states freshness once), and
 * each tile's accessible name is its finding.
 */
import type { DashboardSnapshot } from "../snapshot.js";
import type { TileSection } from "../tiles.js";
import { renderTiles } from "../tiles.js";
import { almanacGrid, rollingHasRows, rollingRain } from "./almanac.js";
import { firstScreen } from "./first-screen.js";
import { featureTile } from "./feature.js";
import { airFact, pressureFact, sunMoonFact, windFact } from "./quick-facts.js";
import { againstNormal, temperaturePeriods, tenDays } from "./temperature.js";

/** One tile of the Overview. */
export type OverviewSection = TileSection;

/** The Overview's sections, in the board's order. */
export const SECTIONS: readonly OverviewSection[] = [
  firstScreen, // #78 the now tile
  featureTile, // #469 the feature tile, in place of What leads
  temperaturePeriods, // #81 temperature across periods
  againstNormal, // #81 "now against normal" strips
  tenDays, // #81 the 10-day high-low list
  almanacGrid, // #82 the almanac grid
  rollingRain, // #82 the rolling-window rain table
  windFact, // #340 the quick facts: one-column tiles, the phone's 2×2
  pressureFact, // #340
  airFact, // #340
  sunMoonFact, // #341
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
  // On a phone the finding rows sit under "In context" (DESIGN.md). Each
  // tile is named by its finding, for screen readers (#458).
  const hiddenSections = new Set<OverviewSection>();
  if (!snapshot.feature) hiddenSections.add(featureTile);
  if (!rollingHasRows(snapshot)) hiddenSections.add(rollingRain);
  // The almanac takes the rolling tile's column when that tile is not
  // drawn, so its four-column row stays whole.
  const shown = sections
    .filter((s) => !hiddenSections.has(s))
    .map((s) =>
      s === almanacGrid && hiddenSections.has(rollingRain) ? { ...s, span: 4 as const } : s,
    );
  const grid = renderTiles(panel, snapshot, shown, {
    rowsHeading: "In context",
    nameByFinding: true,
  });
  for (const tile of grid.querySelectorAll<HTMLElement>("[data-section]")) {
    tile.id = `overview-${tile.dataset.section}`;
  }
}
