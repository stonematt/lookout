/**
 * The Humidity zoom (#477, spec #460; boards `Zoom`, `ZoomDark`): under the
 * tile's number and dew point, the Air card's sentence from before the
 * mosaic, humidity and dew point together; then the tile's 24-hour line at
 * full size (`outdoor_trace`, #554), not drawn without one. No source note
 * (prototype decision 22).
 *
 * Offers `humidityZoom` and `airSentence`.
 */
import { degrees, percent } from "../../format.js";
import type { DashboardSnapshot } from "../../snapshot.js";
import { HUMIDITY_PAD, humidityTile, lastIndex } from "../mosaic-conditions.js";
import type { ZoomDefinition } from "../zoom.js";
import { dayTrace, sentence, sizedChart } from "./parts.js";

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
    {
      id: "day",
      title: "The last 24 hours",
      wide: true,
      shown: (snapshot) => lastIndex(humidityTrace(snapshot)) >= 0,
      render: (host, snapshot) => {
        sizedChart(host, "day-line", (width) =>
          dayTrace(humidityTrace(snapshot), width, (v) => percent(v), HUMIDITY_PAD),
        );
      },
    },
  ],
};

/** The humidity series of `outdoor_trace` (#554); empty without one. */
function humidityTrace(snapshot: DashboardSnapshot): (number | null)[] {
  return snapshot.outdoor_trace?.humidity_pct ?? [];
}
