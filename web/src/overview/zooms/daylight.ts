/**
 * The Daylight zoom (#477, spec #460; boards `Zoom`, `ZoomDark`): under the
 * tile's day length and its change from yesterday, the Sun and moon card's
 * content from before the mosaic, at full size:
 *
 * - Sunrise to sunset: the two times, the tile's arc at the panel's width,
 *   and the daylight sentence ("9 h 23 m of daylight, 2 m less than
 *   yesterday.").
 * - Moon: its phase, its lit share and the days to full.
 *
 * Every value is the snapshot's `sun_moon`, decided in Python, its times in
 * the station's zone. The zoom opens only from a drawn tile, and the tile
 * is not drawn for a station with no location (`sun_moon` null); that note
 * is the sources footer's (`mosaic-conditions.ts`, `daylightTile.absent`).
 * No source note here (prototype decision 22).
 *
 * Offers `daylightZoom` and `moonSentence`.
 */
import { formatHours, formatTime, percent } from "../../format.js";
import type { DashboardSnapshot, SunMoon } from "../../snapshot.js";
import { daylightTile } from "../mosaic-conditions.js";
import type { ZoomDefinition } from "../zoom.js";
import { factValue, sentence, sizedChart } from "./parts.js";

/**
 * The daylight sentence: the day's length, then the tile's own change.
 *
 * @param snapshot - The dashboard snapshot, with its sun and moon.
 * @param block - The snapshot's sun and moon.
 * @returns E.g. "9 h 23 m of daylight, 2 m less than yesterday."
 */
function daylightSentence(snapshot: DashboardSnapshot, block: SunMoon): string {
  const length = formatHours(block.daylight_minutes * 60_000);
  const change = daylightTile.read(snapshot).delta;
  return change ? `${length} of daylight, ${change}.` : `${length} of daylight.`;
}

/**
 * The moon sentence: its phase, its lit share and the days to full.
 *
 * @param block - The snapshot's sun and moon.
 * @returns E.g. "Waning gibbous moon, 68% lit, full in 24 days."
 */
export function moonSentence(block: SunMoon): string {
  const phase = block.moon_phase;
  const lit = `${phase.charAt(0).toUpperCase()}${phase.slice(1)} moon, ${percent(block.moon_illumination_pct)} lit`;
  const days = block.days_to_full_moon;
  if (phase === "full") return `${lit}.`;
  if (days === 0) return `${lit}, full today.`;
  return `${lit}, full in ${days} ${days === 1 ? "day" : "days"}.`;
}

/** Whether the snapshot holds a sun and moon block. */
const located = (snapshot: DashboardSnapshot): boolean => snapshot.sun_moon !== null;

/** The Daylight zoom. */
export const daylightZoom: ZoomDefinition = {
  slug: daylightTile.slug,
  title: daylightTile.title,
  lead: (snapshot) => daylightTile.read(snapshot),
  panels: [
    {
      id: "sun",
      title: "Sunrise to sunset",
      shown: located,
      render(host, snapshot) {
        const block = snapshot.sun_moon!;
        factValue(
          host,
          `${formatTime(block.sunrise_ms)} to ${formatTime(block.sunset_ms)}`,
          "sun-times",
        );
        const chart = daylightTile.chart;
        if (chart) sizedChart(host, "arc", (width) => chart(snapshot, width));
        sentence(host, daylightSentence(snapshot, block), "daylight");
      },
    },
    {
      id: "moon",
      title: "Moon",
      shown: located,
      render(host, snapshot) {
        sentence(host, moonSentence(snapshot.sun_moon!), "moon");
      },
    },
  ],
};
