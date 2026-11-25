/**
 * The rain zoom (#475, spec #460 story 36; board `RainZoom`), titled "Now
 * and this season" (#647 call 5): under the Rain tile's number and delta,
 * the Rain tab's top section (now, the season and the water year, without
 * the tab's window strip) under its first part's label ("Now"), as the tab
 * titles it; the rain gauge in the strip's place (rain over each window,
 * `rain/gauge.ts`, #619), then the rain almanac and rolling-window rain,
 * and "More in Rain". No source fold (prototype
 * decision 22): the Overview's sources footer carries the tile's line.
 *
 * One home per chart: the water year is `rain/index.ts`'s `renderYears`,
 * the almanac and rolling table `almanac.ts`'s; this file only names them.
 *
 * Offers `rainZoom`, and `RAIN_TAB`, the tab both rain zooms end in a link to.
 */
import { gaugeShown, renderGauge } from "../../rain/gauge.js";
import { headKicker, renderYears } from "../../rain/index.js";
import { almanacHasMonths, renderGrid, renderRolling, rollingHasRows } from "../almanac.js";
import { rainTile } from "../mosaic-tiles.js";
import type { ZoomDefinition } from "../zoom.js";
import type { MoreTab } from "./more.js";
import { morePanel } from "./more.js";

/** The Rain tab, which the rain and dry spell zooms end in a link to. */
export const RAIN_TAB: MoreTab = { hash: "rain", name: "Rain" };

/** The rain zoom. */
export const rainZoom: ZoomDefinition = {
  slug: rainTile.slug,
  // A crop of the Rain tab head, which leads with now (#647 call 5).
  title: "Now and this season",
  lead: (snapshot) => rainTile.read(snapshot),
  panels: [
    {
      id: "water-year",
      // The Rain tab head's kicker, its first part's label: "Now" (#703).
      title: headKicker,
      wide: true,
      render: (host, snapshot) => renderYears(host, snapshot, { zoom: true }),
    },
    {
      id: "rain-windows",
      title: "Rain over each window",
      wide: true,
      shown: gaugeShown,
      render: renderGauge,
    },
    {
      id: "almanac",
      title: "Rain almanac",
      shown: almanacHasMonths,
      render: (host, snapshot) => renderGrid(host, snapshot, { zoom: true }),
    },
    {
      id: "rolling-rain",
      title: "Rain, rolling windows",
      shown: rollingHasRows,
      render: (host, snapshot) => renderRolling(host, snapshot, { zoom: true }),
    },
    morePanel(RAIN_TAB, "Each day of the last 90, dry spells, rain by the hour"),
  ],
};
