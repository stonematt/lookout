/**
 * The wet hours zoom (#658): under the Wet hours tile's number and delta,
 * the wet-hour meter (the last 7 and 30 days, `wet-meter.ts`), then "More
 * in Rain". Its heading is the tile's title, read from the snapshot. No
 * source fold (prototype decision 22): the Overview's sources footer
 * carries the tile's line.
 *
 * One home per chart: the meter is `wet-meter.ts`'s; this file only names it.
 *
 * Offers `wetHoursZoom`.
 */
import { tileTitle } from "../mosaic.js";
import { wetHoursTile } from "../mosaic-wet.js";
import { wetMeter } from "../wet-meter.js";
import type { ZoomDefinition } from "../zoom.js";
import { morePanel } from "./more.js";
import { RAIN_TAB } from "./rain.js";

/** The wet hours zoom. */
export const wetHoursZoom: ZoomDefinition = {
  slug: wetHoursTile.slug,
  title: (snapshot) => tileTitle(wetHoursTile, snapshot),
  lead: (snapshot) => wetHoursTile.read(snapshot),
  panels: [
    {
      id: "wet-hours",
      title: "The last 7 and 30 days",
      wide: true,
      render: (host, snapshot) => {
        const meter = wetMeter(snapshot);
        if (meter) host.append(meter);
      },
    },
    morePanel(RAIN_TAB, "The water year, each day of the last 90, rain by the hour"),
  ],
};
