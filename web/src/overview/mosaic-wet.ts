/**
 * The wet-or-dry tile in wet mode (#658, spec #647; Call 4 on #647, #661):
 * titled from the snapshot ("Wet hours"), with the bars' how-to-read words
 * after it (" · each bar is every hour; dark is wet", #703), its number the last 7 days' wet
 * hours of their countable hours, its mini chart the two wet-hour bars, the
 * last 7 days and the last 30, each a tally of every hour in its window
 * ("7 days", "19 h of 168"). The 30-day strip, a column a day, moved to the
 * tile's zoom (`stripChart`, `zooms/wet-hours.ts`). In dry mode the slot
 * holds the Dry spell tile (`mosaic-tiles.ts`) instead; while rain leads the
 * tile leaves the mosaic and its bars sit in the rain tile's foot (#661).
 *
 * Every number is the snapshot's `wet_hours` (GLOSSARY wet hour, countable
 * hour): an hour or a day the record cannot count is hatched, never drawn as
 * dry, and the tile then carries the hatch key.
 *
 * Offers `wetHoursTile` and `stripChart`.
 */
import { LABEL_CHAR_PX } from "../charts/svg.js";
import { svgEl } from "../dom.js";
import { svgHatch } from "../gaps.js";
import type { DashboardSnapshot, WetHoursWindow } from "../snapshot.js";
import { howToRead } from "../wet-bars.js";
import type { MosaicTile } from "./mosaic.js";
import { MINI_HEIGHT, miniChart } from "./mosaic.js";

/** A day's hours: a column full height holds a whole day of wet hours. */
const DAY_HOURS = 24;

/** The height a counted day with no wet hour draws, so dry reads as dry. */
const DRY_TICK = 2;

/** A bar's height in the tile. */
const BAR = 10;

/**
 * The 30-day strip as columns: each counted day's wet hours of 24, a dry
 * day a tick on the baseline, a day with no countable hour hatched.
 *
 * @param snapshot - The dashboard snapshot.
 * @param width - The width to draw at.
 * @param height - The height to draw at.
 * @returns The chart, or null without wet hours.
 */
export function stripChart(
  snapshot: DashboardSnapshot,
  width: number,
  height = MINI_HEIGHT,
): SVGSVGElement | null {
  const days = snapshot.wet_hours?.days ?? [];
  if (!days.length) return null;
  const gap = 2;
  const colW = (width - gap * (days.length - 1)) / days.length;
  const svg = miniChart(width, height);
  const hatch = svgHatch(svg);
  days.forEach((day, i) => {
    const x = i * (colW + gap);
    if (day.wet_hours === null) {
      svg.append(svgEl("rect", { x, y: 0, width: colW, height, class: "mosaic-gap", fill: hatch }));
      return;
    }
    const h = Math.max(DRY_TICK, (Math.min(day.wet_hours, DAY_HOURS) / DAY_HOURS) * height);
    svg.append(
      svgEl("rect", {
        x,
        y: height - h,
        width: colW,
        height: h,
        class: day.wet_hours > 0 ? "mosaic-wet" : "mosaic-dry",
      }),
    );
  });
  return svg;
}

/**
 * One window's bar in the tile: its name and count over it, then the bar.
 *
 * @param svg - The chart.
 * @param hatch - The hatch's fill.
 * @param name - "7 days".
 * @param window - The window.
 * @param top - The row's top.
 * @param width - The chart's width.
 */
function bar(
  svg: SVGSVGElement,
  hatch: string,
  name: string,
  window: WetHoursWindow,
  top: number,
  width: number,
): void {
  // The count in full when it fits beside the name, else the wet hours.
  const full = `${window.wet_hours} h of ${window.countable_hours}`;
  const fits = (name.length + full.length) * LABEL_CHAR_PX + 8 <= width;
  svg.append(
    svgEl("text", { class: "mosaic-label", x: 0, y: top + 11 }, name),
    svgEl(
      "text",
      { class: "mosaic-count", x: width, y: top + 11, "text-anchor": "end" },
      fits ? full : `${window.wet_hours} h`,
    ),
  );
  const y = top + 16;
  const inner = width - 1;
  const share = (hours: number) => (window.hours > 0 ? (inner * hours) / window.hours : 0);
  svg.append(
    svgEl("rect", {
      class: "mosaic-track",
      x: 0.5,
      y: y + 0.5,
      width: inner,
      height: BAR - 1,
      rx: (BAR - 1) / 2,
    }),
  );
  if (window.wet_hours > 0) {
    svg.append(
      svgEl("rect", {
        class: "mosaic-wet-fill",
        x: 0.5,
        y: y + 0.5,
        width: Math.max(3, share(window.wet_hours)),
        height: BAR - 1,
        rx: (BAR - 1) / 2,
      }),
    );
  }
  if (window.missing_hours > 0) {
    const w = Math.max(3, share(window.missing_hours));
    svg.append(
      svgEl("rect", {
        class: "mosaic-gap mosaic-hatch-end",
        x: 0.5 + inner - w,
        y: y + 0.5,
        width: w,
        height: BAR - 1,
        fill: hatch,
      }),
    );
  }
}

/**
 * The two wet-hour bars, the last 7 days over the last 30.
 *
 * @param snapshot - The dashboard snapshot.
 * @param width - The width to draw at.
 * @returns The chart, or null without wet hours.
 */
function barsChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const wet = snapshot.wet_hours ?? null;
  if (!wet) return null;
  const svg = miniChart(width);
  const hatch = svgHatch(svg);
  bar(svg, hatch, "7 days", wet.last_7_days, 2, width);
  bar(svg, hatch, "30 days", wet.last_30_days, 34, width);
  return svg;
}

/** Wet hours: the last 7 days' wet hours of their countable hours. */
export const wetHoursTile: MosaicTile = {
  slug: "wet-hours",
  depth: "wet_hours",
  title: "Wet hours",
  titleFrom: (snapshot) => snapshot.tile_titles?.wet_or_dry,
  // The bars' header word for word, as everywhere they appear (#647, #703).
  how: howToRead,
  zoomName: "wet hours",
  shown: (snapshot) => Boolean(snapshot.wet_hours),
  read(snapshot) {
    const week = snapshot.wet_hours?.last_7_days;
    if (!week) return { value: null, delta: null };
    return {
      value: String(week.wet_hours),
      unit: "h",
      delta: `of ${week.countable_hours} in the last 7 days`,
    };
  },
  chart: barsChart,
  hatched: (snapshot) =>
    (snapshot.wet_hours?.last_7_days.missing_hours ?? 0) > 0 ||
    (snapshot.wet_hours?.last_30_days.missing_hours ?? 0) > 0,
  source: (snapshot) => ({
    line: "clock hours with a tip, of the hours the record counts.",
    method:
      "A wet hour is a station-local clock hour with at least one tip (0.01 in); the number " +
      "is the last 7 days' wet hours of the hours the record can count. Each bar is every " +
      "hour of its window, the last 7 days and the last 30; the dark fill is its wet hours, " +
      "and the hours the record cannot count are hatched at its end, never read as dry. " +
      modeWords(snapshot),
  }),
};

/**
 * What decided the mode, in a sentence: words for the snapshot's
 * `decided_by`, never the rule's thresholds, which are Python's.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The sentence; empty without a mode.
 */
function modeWords(snapshot: DashboardSnapshot): string {
  const mode = snapshot.wet_mode;
  if (!mode) return "";
  const name = mode.mode === "wet" ? "Wet mode" : "Dry mode";
  return mode.decided_by === "calendar"
    ? `${name}, decided by the calendar: too few recent hours could be counted.`
    : `${name}, decided by the weather: the share of recent hours that were wet.`;
}
