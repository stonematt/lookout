/**
 * The wet hours zoom (#658): under the Wet hours tile's number and delta,
 * the wet-hour bars (the last 7 and 30 days, `wet-meter.ts`), then each of
 * the last 30 days, a column a day (`stripChart`, moved here from the tile
 * by Call 4 on #647, #661), then "More in Rain". Its heading is the tile's
 * title, read from the snapshot. No source fold (prototype decision 22):
 * the Overview's sources footer carries the tile's line.
 *
 * One home per chart: the bars are `wet-meter.ts`'s, the strip
 * `mosaic-wet.ts`'s; this file only names them.
 *
 * Offers `wetHoursZoom`.
 */
import { drawSized } from "../../charts/svg.js";
import { make } from "../../dom.js";
import { tileTitle } from "../mosaic.js";
import { stripChart, wetHoursTile } from "../mosaic-wet.js";
import { wetMeter } from "../wet-meter.js";
import type { ZoomDefinition } from "../zoom.js";
import { morePanel } from "./more.js";
import { RAIN_TAB } from "./rain.js";

/** The strip's height in the zoom: taller than the tile's was. */
const STRIP_HEIGHT = 96;

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
    {
      id: "wet-days",
      title: "Each of the last 30 days, its wet hours of 24",
      wide: true,
      shown: (snapshot) => Boolean(snapshot.wet_hours?.days.length),
      render: (host, snapshot) => {
        const plot = make("div", "mosaic-chart");
        plot.dataset.part = "strip";
        plot.setAttribute("aria-hidden", "true");
        host.append(plot);
        drawSized(plot, (width) => {
          const svg = stripChart(snapshot, width, STRIP_HEIGHT);
          plot.replaceChildren(...(svg ? [svg] : []));
        });
      },
    },
    morePanel(RAIN_TAB, "The water year, each day of the last 90, rain by the hour"),
  ],
};
