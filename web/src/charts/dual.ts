/**
 * The dual rain chart's scale (spec #647): rain as columns on one axis and
 * the running total as a line on the other, the two axes at a fixed 10:1
 * ratio that rescale together, so a column and the line always read on the
 * same grid. The Rain Events tab's open row draws it (#419); the Overview's
 * rain feature chart draws the same form (#661).
 *
 * Offers `DUAL_RATIO`, `dualScale` and `dualTickText`.
 */
import { zeroTicks } from "./svg.js";

/** The total axis reads this many times the column axis, always. */
export const DUAL_RATIO = 10;

/** Both axes of a dual chart, tick for tick on one grid. */
export interface DualScale {
  /** The total (line) axis's top and ticks, from 0. */
  totalTop: number;
  totalTicks: number[];
  /** The column axis: the total axis over `DUAL_RATIO`, tick for tick. */
  columnTop: number;
  columnTicks: number[];
}

/**
 * The two axes of a dual chart: round ticks from 0 past whichever is larger,
 * the total or ten times the tallest column, the column axis a tenth of the
 * total axis at every tick.
 *
 * @param totalMax - The largest value the line reaches (the median level too).
 * @param columnMax - The tallest column.
 * @param steps - About how many steps up the axis: 2 on a phone, 4 wider.
 * @returns Both axes' tops and ticks.
 */
export function dualScale(totalMax: number, columnMax: number, steps = 4): DualScale {
  const most = Math.max(totalMax, columnMax * DUAL_RATIO, 0);
  // A little headroom over the largest mark, for the label above it.
  const totalTicks = zeroTicks(most * 1.08, steps);
  const totalTop = totalTicks[totalTicks.length - 1]!;
  const columnTicks = totalTicks.map((v) => Number((v / DUAL_RATIO).toPrecision(6)));
  return { totalTop, totalTicks, columnTop: totalTop / DUAL_RATIO, columnTicks };
}

/**
 * An axis's tick labels: plain numbers, the unit said on the top one alone
 * ("0.3 in").
 *
 * @param ticks - The ticks, ascending.
 * @param unit - The unit, e.g. "in".
 * @returns One label per tick.
 */
export function dualTickText(ticks: number[], unit: string): string[] {
  return ticks.map((v, i) => (i === ticks.length - 1 ? `${v} ${unit}` : String(v)));
}
