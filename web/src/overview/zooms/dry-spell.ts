/**
 * The dry spell zoom (#475, spec #460 story 37): under the Dry spell tile's
 * number and delta, the Rain tab's dry spell section, then "More in Rain".
 * No source fold (prototype decision 22): the Overview's sources footer
 * carries the tile's line.
 *
 * One home per chart: the section is `rain/index.ts`'s `renderDry`; this
 * file only names it.
 *
 * Offers `drySpellZoom`.
 */
import { renderDry } from "../../rain/index.js";
import { drySpellTile } from "../mosaic-tiles.js";
import type { ZoomDefinition } from "../zoom.js";
import { morePanel } from "./more.js";
import { RAIN_TAB } from "./rain.js";

/** The dry spell zoom. */
export const drySpellZoom: ZoomDefinition = {
  slug: drySpellTile.slug,
  title: drySpellTile.title,
  lead: (snapshot) => drySpellTile.read(snapshot),
  panels: [
    {
      id: "dry-spells",
      title: "Dry spells",
      wide: true,
      render: (host, snapshot) => renderDry(host, snapshot, { zoom: true }),
    },
    morePanel(RAIN_TAB, "The water year, each day of the last 90, rain by the hour"),
  ],
};
