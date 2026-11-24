/**
 * The wet-or-dry tile in wet mode (#658, spec #647; board Later): titled
 * from the snapshot ("Wet hours"), its number the last 7 days' wet hours of
 * their countable hours, its mini chart the 30-day strip, a column a day.
 * In dry mode the slot holds the Dry spell tile (`mosaic-tiles.ts`) instead.
 *
 * Every number is the snapshot's `wet_hours` (GLOSSARY wet hour, countable
 * hour): a day the record cannot count is hatched, never drawn as dry.
 *
 * Offers `wetHoursTile`.
 */
import { svgEl } from "../dom.js";
import { svgHatch } from "../gaps.js";
import type { DashboardSnapshot } from "../snapshot.js";
import type { MosaicTile } from "./mosaic.js";
import { MINI_HEIGHT, miniChart } from "./mosaic.js";

/** A day's hours: a column full height holds a whole day of wet hours. */
const DAY_HOURS = 24;

/** The height a counted day with no wet hour draws, so dry reads as dry. */
const DRY_TICK = 2;

/**
 * The 30-day strip as columns: each counted day's wet hours of 24, a dry
 * day a tick on the baseline, a day with no countable hour hatched.
 *
 * @param snapshot - The dashboard snapshot.
 * @param width - The width to draw at.
 * @returns The chart, or null without wet hours.
 */
function stripChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const days = snapshot.wet_hours?.days ?? [];
  if (!days.length) return null;
  const gap = 2;
  const colW = (width - gap * (days.length - 1)) / days.length;
  const svg = miniChart(width);
  const hatch = svgHatch(svg);
  days.forEach((day, i) => {
    const x = i * (colW + gap);
    if (day.wet_hours === null) {
      svg.append(
        svgEl("rect", {
          x,
          y: 0,
          width: colW,
          height: MINI_HEIGHT,
          class: "mosaic-gap",
          fill: hatch,
        }),
      );
      return;
    }
    const h = Math.max(DRY_TICK, (Math.min(day.wet_hours, DAY_HOURS) / DAY_HOURS) * MINI_HEIGHT);
    svg.append(
      svgEl("rect", {
        x,
        y: MINI_HEIGHT - h,
        width: colW,
        height: h,
        class: day.wet_hours > 0 ? "mosaic-wet" : "mosaic-dry",
      }),
    );
  });
  return svg;
}

/** Wet hours: the last 7 days' wet hours of their countable hours. */
export const wetHoursTile: MosaicTile = {
  slug: "wet-hours",
  depth: "wet_hours",
  title: "Wet hours",
  titleFrom: (snapshot) => snapshot.tile_titles?.wet_or_dry,
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
  chart: stripChart,
  source: (snapshot) => ({
    line: "clock hours with a tip, of the hours the record counts.",
    method:
      "A wet hour is a station-local clock hour with at least one tip (0.01 in); the number " +
      "is the last 7 days' wet hours of the hours the record can count. The columns are the " +
      "last 30 days, each day's wet hours of 24; a day the record cannot count is hatched, " +
      `never read as dry. ${modeWords(snapshot)}`,
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
