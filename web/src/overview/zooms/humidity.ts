/**
 * The Humidity zoom (#477, spec #460; boards `Zoom`, `ZoomDark`): under the
 * tile's number and dew point, the Air card's sentence from before the
 * mosaic, humidity and dew point together. No source note (prototype
 * decision 22).
 *
 * Offers `humidityZoom` and `airSentence`.
 */
import { degrees, percent } from "../../format.js";
import type { DashboardSnapshot } from "../../snapshot.js";
import { humidityTile } from "../mosaic-conditions.js";
import type { ZoomDefinition } from "../zoom.js";
import { sentence } from "./parts.js";

/**
 * The Air sentence: the humidity, then the dew point.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Humidity 66%; dew point 39.1°."
 */
export function airSentence(snapshot: DashboardSnapshot): string {
  const c = snapshot.current;
  const dew = c.dew_point_f === null ? "not known" : degrees(c.dew_point_f);
  if (c.humidity_pct === null) {
    return c.dew_point_f === null
      ? "Humidity and dew point not known."
      : `Humidity not known; dew point ${dew}.`;
  }
  return `Humidity ${percent(c.humidity_pct)}; dew point ${dew}.`;
}

/** The Humidity zoom. */
export const humidityZoom: ZoomDefinition = {
  slug: humidityTile.slug,
  title: humidityTile.title,
  lead: (snapshot) => humidityTile.read(snapshot),
  panels: [
    {
      id: "air",
      title: "Air",
      wide: true,
      render: (host, snapshot) => {
        sentence(host, airSentence(snapshot), "sentence");
      },
    },
  ],
};
