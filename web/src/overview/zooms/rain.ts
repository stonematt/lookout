/**
 * The rain zoom (#475, spec #460 story 36; board `RainZoom`): under the
 * Rain tile's number and delta, the Rain tab's top section (the water year
 * with the window strip under its chart), then the rain almanac and
 * rolling-window rain, and "More in Rain". No source fold (prototype
 * decision 22): the Overview's sources footer carries the tile's line.
 *
 * One home per chart: the water year is `rain/index.ts`'s `renderYears`,
 * the almanac and rolling table `almanac.ts`'s; this file only names them.
 *
 * Offers `rainZoom`, and `RAIN_TAB`, the tab both rain zooms end in a link to.
 */
import { renderYears } from "../../rain/index.js";
import { renderGrid, renderRolling, rollingHasRows } from "../almanac.js";
import { rainTile } from "../mosaic-tiles.js";
import type { ZoomDefinition } from "../zoom.js";
import type { MoreTab } from "./more.js";
import { morePanel } from "./more.js";

/** The Rain tab, which the rain and dry spell zooms end in a link to. */
export const RAIN_TAB: MoreTab = { hash: "rain", name: "Rain" };

/** The rain zoom. */
export const rainZoom: ZoomDefinition = {
  slug: rainTile.slug,
  title: rainTile.title,
  lead: (snapshot) => rainTile.read(snapshot),
  panels: [
    {
      id: "water-year",
      title: "The water year",
      wide: true,
      render: (host, snapshot) => renderYears(host, snapshot, { zoom: true }),
    },
    {
      id: "almanac",
      title: "Rain almanac",
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
