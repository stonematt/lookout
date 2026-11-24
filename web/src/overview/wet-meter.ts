/**
 * The wet-hour bars (#657, spec #647 "Meter bars"; the C boards' words,
 * approved with Overview C, #661): two bars, the last 7 days and the last
 * 30, each a tally of every hour in its window, not a timeline. The dark
 * fill is the wet hours; the hours the record cannot count are hatched at
 * the bar's end, never drawn as dry. Each bar reads wet hours of countable
 * hours ("19 h of 168"), never a percent. The header says how to read them,
 * "Wet hours · each bar is every hour; dark is wet", and a bar with hatched
 * hours brings the key under the bars: the swatch and "hatched: missing,
 * not dry".
 *
 * The bars draw in the rain tile's season foot while rain leads (#661) and
 * in the wet hours zoom (`zooms/wet-hours.ts`); as one link to the Rain tab
 * (`wetMeter`) they stay at the now tile's foot only for a snapshot without
 * a wet mode, where no Wet hours tile carries them (Call 4 on #647: no count
 * shows twice). Every number is the snapshot's `wet_hours` (GLOSSARY wet
 * hour, countable hour); a snapshot without it (N−1, ADR-0008) draws none,
 * and nor does dry mode (#658).
 *
 * Offers `meterParts`, `wetMeter`, `meterShown` and `missingWords` (the Rain
 * tab head's note words).
 */
import "./wet-meter.css";

import { make, spoken } from "../dom.js";
import type { DashboardSnapshot, WetHours, WetHoursWindow } from "../snapshot.js";

/** The bars' header: their name, then how to read them (the C boards). */
export const METER_TITLE = "Wet hours";
export const METER_HOW = " · each bar is every hour; dark is wet";
/** The key under the bars when any hour is hatched (the C boards). */
export const HATCH_KEY = "hatched: missing, not dry";

/**
 * What a window is missing, phrase by phrase; none when it misses nothing.
 *
 * @param window - The window.
 * @returns ["24 h missing, not dry", "1 d missing"], without the days when
 *   no whole day is missing; read together with " · ".
 */
export function missingWords(window: WetHoursWindow): string[] {
  if (window.missing_hours === 0) return [];
  const hours = `${window.missing_hours} h missing, not dry`;
  return window.missing_days > 0 ? [hours, `${window.missing_days} d missing`] : [hours];
}

/**
 * One part of a bar, sized as a share of the window's hours.
 *
 * @param className - The part's class.
 * @param hours - Its hours.
 * @param total - The window's hours.
 * @returns The part, or null when it holds no hour.
 */
function part(className: string, hours: number, total: number): HTMLElement | null {
  if (hours <= 0 || total <= 0) return null;
  const node = make("span", className);
  // Its share; the stylesheet draws it at least 3px, so 1 h of 720 shows.
  node.style.setProperty("--share", `${(100 * hours) / total}%`);
  return node;
}

/**
 * One window's row: its name, its bar and its count.
 *
 * @param name - The window's name: "7 days".
 * @param window - The window.
 * @returns The row's cells, in grid order.
 */
function row(name: string, window: WetHoursWindow): HTMLElement[] {
  const label = make("span", "wet-meter-name", name);
  const bar = make("span", "wet-meter-bar");
  bar.dataset.part = "bar";
  const wet = part("wet-meter-wet", window.wet_hours, window.hours);
  const missing = part("wet-meter-missing gap-hatch", window.missing_hours, window.hours);
  if (wet) bar.append(wet);
  if (missing) bar.append(missing);
  const count = make("span", "wet-meter-count");
  count.dataset.part = "count";
  count.append(make("b", "", `${window.wet_hours} h`), ` of ${window.countable_hours}`);
  return [label, bar, count];
}

/**
 * The bars' parts: the header, the two rows, and the hatch key when any
 * hour is hatched.
 *
 * @param wet - The snapshot's wet hours.
 * @returns The parts, in order.
 */
export function meterParts(wet: WetHours): HTMLElement[] {
  const title = make("span", "wet-meter-title", METER_TITLE);
  title.append(make("span", "wet-meter-how", METER_HOW));
  const rows = make("span", "wet-meter-rows");
  rows.append(...row("7 days", wet.last_7_days), ...row("30 days", wet.last_30_days));
  const parts = [title, rows];
  if (wet.last_7_days.missing_hours > 0 || wet.last_30_days.missing_hours > 0) {
    const key = make("span", "wet-meter-key");
    key.dataset.part = "hatch-key";
    key.append(make("span", "hatch-swatch"), make("span", "", HATCH_KEY));
    parts.push(key);
  }
  return parts;
}

/**
 * Whether the bars are drawn: the snapshot has wet hours and is not in dry
 * mode (#658), where they would read 0 h for months. Without a mode (N−1)
 * they are drawn as before.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns True to draw them.
 */
export function meterShown(snapshot: DashboardSnapshot): boolean {
  return Boolean(snapshot.wet_hours) && snapshot.wet_mode?.mode !== "dry";
}

/**
 * The bars as one link to the Rain tab, or null when they are not drawn
 * (`meterShown`).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The link.
 */
export function wetMeter(snapshot: DashboardSnapshot): HTMLElement | null {
  const wet = snapshot.wet_hours ?? null;
  if (!wet || !meterShown(snapshot)) return null;
  const link = make("a", "wet-meter");
  link.href = "#rain";
  link.dataset.part = "wet-meter";
  link.append(...meterParts(wet), spoken("Open the rain view"));
  return link;
}
