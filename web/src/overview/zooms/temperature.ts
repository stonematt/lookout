/**
 * The temperature zoom (#474, spec #460 story 35; boards `Zoom`,
 * `ZoomDark`): under the Temperature tile's number and delta, the window
 * range chart with the Usual row on top and Full history, then the
 * now-against-this-date strips. No source fold (prototype decision 22): the
 * Overview's sources footer carries the tile's line.
 *
 * The content is `temperature.ts`'s; this file only names it.
 *
 * Offers `temperatureZoom`.
 */
import { temperatureTile } from "../mosaic-tiles.js";
import { renderPeriodsZoom, renderStripsZoom } from "../temperature.js";
import type { ZoomDefinition } from "../zoom.js";

/** The temperature zoom. */
export const temperatureZoom: ZoomDefinition = {
  slug: temperatureTile.slug,
  title: temperatureTile.title,
  lead: (snapshot) => temperatureTile.read(snapshot),
  panels: [
    { id: "periods", title: "Temperature, in context", render: renderPeriodsZoom },
    { id: "strips", title: "Now against this date", render: renderStripsZoom },
  ],
};
