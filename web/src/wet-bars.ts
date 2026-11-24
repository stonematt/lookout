/**
 * The wet-hour bars' shared parts (#694): the Overview's bars
 * (`overview/wet-meter.ts`, in the rain tile's foot and the wet hours zoom)
 * and the Rain tab head's (`rain/head.ts`) paint one bar and say it in the
 * C boards' words, approved by Matt with Overview C (#647):
 *
 * - the header, "Wet hours · each bar is every hour; dark is wet", its
 *   name the snapshot's tile title (`tile_titles.wet_or_dry`, GLOSSARY tile
 *   title) where the bars stand for the Wet hours tile (the Overview); on
 *   the Rain tab head the season's label is always "Wet hours", as the
 *   since-Oct-1 bar is drawn in dry mode too, when the tile title is "Dry
 *   spell". The half-width Wet hours tile keeps its title on one line and
 *   sets the same how-to-read words under its bars (#706);
 * - each bar a tally of every hour in its span: the dark fill its wet
 *   hours, the hours the record cannot count hatched at its end, never
 *   drawn as dry, each part its share of the span's hours;
 * - the count, wet hours of countable hours ("19 h of 168"), never a
 *   percent;
 * - under the bars when any hour is hatched, the key: the swatch and
 *   "hatched: missing, not dry".
 *
 * The 7-day and 30-day bars are drawn only outside dry mode (#658, #647
 * call 3), where they would read 0 h for months (`meterShown`).
 *
 * Offers `METER_TITLE`, `HOW_TO_READ`, `HATCH_KEY`, `meterTitle`, `meterShown`,
 * `howToRead`, `barShare`, `barCount` and `hatchKey`.
 */
import "./wet-bars.css";

import { make } from "./dom.js";
import type { DashboardSnapshot } from "./snapshot.js";

/** The bars' name where no tile title names them (N−1, the Rain tab's season). */
export const METER_TITLE = "Wet hours";
/**
 * How to read the bars (the C boards): after their name where they run full
 * width, the caption under them on the half-width Wet hours tile (#706).
 */
export const HOW_TO_READ = "each bar is every hour; dark is wet";
/** The key under the bars when any hour is hatched (the C boards). */
export const HATCH_KEY = "hatched: missing, not dry";

/**
 * The bars' name where they stand for the Wet hours tile: the snapshot's
 * tile title, the literal only on a document without one (ADR-0008).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Wet hours".
 */
export function meterTitle(snapshot: DashboardSnapshot): string {
  return snapshot.tile_titles?.wet_or_dry ?? METER_TITLE;
}

/**
 * Whether the 7-day and 30-day bars are drawn: the snapshot has wet hours
 * and is not in dry mode (#658), where they would read 0 h for months.
 * Without a mode (N−1) they are drawn as before.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns True to draw them.
 */
export function meterShown(snapshot: DashboardSnapshot): boolean {
  return Boolean(snapshot.wet_hours) && snapshot.wet_mode?.mode !== "dry";
}

/**
 * How to read the bars, set after their name in a lighter weight.
 *
 * @returns " · each bar is every hour; dark is wet", as a span.
 */
export function howToRead(): HTMLElement {
  return make("span", "wet-meter-how", ` · ${HOW_TO_READ}`);
}

/**
 * One part of a bar, sized as a share of its span's hours; the stylesheet
 * draws it at least 3px, so 1 h of 720 shows.
 *
 * @param className - The part's class.
 * @param hours - Its hours.
 * @param total - The span's hours.
 * @returns The part, or null when it holds no hour.
 */
export function barShare(className: string, hours: number, total: number): HTMLElement | null {
  if (hours <= 0 || total <= 0) return null;
  const node = make("span", className);
  node.style.setProperty("--share", `${(100 * hours) / total}%`);
  return node;
}

/**
 * A bar's count: wet hours in bold, of countable hours.
 *
 * @param className - The count's class.
 * @param wet - Wet hours.
 * @param countable - Countable hours.
 * @returns E.g. "**19 h** of 168".
 */
export function barCount(className: string, wet: number, countable: number): HTMLElement {
  const count = make("span", className);
  count.dataset.part = "count";
  count.append(make("b", "", `${wet.toLocaleString("en-US")} h`));
  count.append(` of ${countable.toLocaleString("en-US")}`);
  return count;
}

/**
 * The key under the bars.
 *
 * @returns The hatch swatch, then "hatched: missing, not dry".
 */
export function hatchKey(): HTMLElement {
  const key = make("span", "wet-meter-key");
  key.dataset.part = "hatch-key";
  key.append(make("span", "hatch-swatch"), make("span", "", HATCH_KEY));
  return key;
}
