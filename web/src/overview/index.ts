/**
 * The Overview tab: a grid of tiles (spec #332, DESIGN.md "Layout"). The now
 * tile and the leading tile first, the phone's fixed top; then temperature
 * across periods, now against this date, the last ten days, the rain almanac
 * and the rolling-window rain table, then the quick facts (wind, pressure,
 * air). The order and spans follow the desktop board: in the four-column
 * grid the now and leading tiles share the first row, temperature in context
 * takes half the second beside the two single tiles, the almanac takes three
 * columns beside the rolling windows, and the quick facts take a row of
 * one-column tiles whose fourth cell is sun and moon's. On a phone the quick
 * facts sit after the finding rows, two to a row.
 *
 * A section is a module under `web/src/overview/` exporting an
 * `OverviewSection` (a `TileSection`, `web/src/tiles.ts`): its id, label,
 * kicker, span, whether it is in the phone's fixed top, and its render. To
 * add one, import it here and add it to `SECTIONS` at its place: one import
 * line and one list line. Each section renders into its own tile, already
 * attached to the page and holding its kicker, so a chart can measure its
 * width when it draws.
 */
import type { DashboardSnapshot } from "../snapshot.js";
import type { TileSection } from "../tiles.js";
import { renderTiles } from "../tiles.js";
import { almanacGrid, rollingRain } from "./almanac.js";
import { firstScreen } from "./first-screen.js";
import { heroLead } from "./hero.js";
import { airFact, pressureFact, sunMoonFact, windFact } from "./quick-facts.js";
import { againstNormal, temperaturePeriods, tenDays } from "./temperature.js";

/** One tile of the Overview. */
export type OverviewSection = TileSection;

/** The Overview's sections, in the board's order. */
export const SECTIONS: readonly OverviewSection[] = [
  firstScreen, // #78 the now tile
  heroLead, // #79 the leading tile and "Why this leads"
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
  // On a phone the finding rows sit under "In context" (DESIGN.md).
  renderTiles(panel, snapshot, sections, { rowsHeading: "In context" });
}
