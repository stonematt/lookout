/**
 * The Pressure zoom (#477, spec #460; boards `Zoom`, `ZoomDark`): under the
 * tile's number and trend, the Pressure card's full sentence from before
 * the mosaic: the reading and its 3-hour trend word (`barometer_trend`,
 * decided in Python); then the tile's 24-hour line at full size
 * (`outdoor_trace`, #554), not drawn without one. No source note
 * (prototype decision 22).
 *
 * Offers `pressureZoom` and `pressureSentence`.
 */
import { inHg } from "../../format.js";
import type { DashboardSnapshot } from "../../snapshot.js";
import { PRESSURE_PAD, pressureTile } from "../mosaic-conditions.js";
import type { ZoomDefinition } from "../zoom.js";
import { dayPanel, sentence } from "./parts.js";

/**
 * The pressure sentence: the reading, then the 3-hour trend word.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "29.92 inHg, rising over the last 3 hours."
 */
export function pressureSentence(snapshot: DashboardSnapshot): string {
  const c = snapshot.current;
  const trend = c.barometer_trend;
  if (c.barometer_inhg === null) {
    return trend === null
      ? "Reading and 3-hour trend not known."
      : `Reading not known; ${trend} over the last 3 hours.`;
  }
  const reading = inHg(c.barometer_inhg);
  return trend === null
    ? `${reading} inHg; the 3-hour trend is not known.`
    : `${reading} inHg, ${trend} over the last 3 hours.`;
}

/** The Pressure zoom. */
export const pressureZoom: ZoomDefinition = {
  slug: pressureTile.slug,
  title: pressureTile.title,
  lead: (snapshot) => pressureTile.read(snapshot),
  panels: [
    {
      id: "trend",
      title: "The last 3 hours",
      wide: true,
      render: (host, snapshot) => {
        sentence(host, pressureSentence(snapshot), "sentence");
      },
    },
    dayPanel((snapshot) => snapshot.outdoor_trace?.pressure_inhg ?? [], inHg, PRESSURE_PAD),
  ],
};
