/**
 * The wet-hour bars on the Overview (#657, spec #647 "Meter bars"; the C
 * boards' words, approved with Overview C, #661): two bars, the last 7 days
 * and the last 30, each a tally of every hour in its window, not a
 * timeline. The header, the bar's parts, the count and the hatch key are
 * the shared wet-hour bars' (`wet-bars.ts`), which the Rain tab head draws
 * too; the header's name is the snapshot's tile title.
 *
 * The bars draw in the rain tile's season foot while rain leads (#661) and
 * in the wet hours zoom (`zooms/wet-hours.ts`); as one link to the Rain tab
 * (`wetMeter`) they stay at the now tile's foot only for a snapshot without
 * a wet mode, where no Wet hours tile carries them (Call 4 on #647: no count
 * shows twice). Every number is the snapshot's `wet_hours` (GLOSSARY wet
 * hour, countable hour); a snapshot without it (N−1, ADR-0008) draws none,
 * and nor does dry mode (#658, `meterShown`).
 *
 * Offers `meterParts` and `wetMeter`.
 */
import "./wet-meter.css";

import { make, spoken } from "../dom.js";
import type { DashboardSnapshot, WetHours, WetHoursWindow } from "../snapshot.js";
import { barCount, barShare, hatchKey, howToRead, meterShown, meterTitle } from "../wet-bars.js";

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
  const wet = barShare("wet-meter-wet", window.wet_hours, window.hours);
  const missing = barShare("wet-meter-missing gap-hatch", window.missing_hours, window.hours);
  if (wet) bar.append(wet);
  if (missing) bar.append(missing);
  return [label, bar, barCount("wet-meter-count", window.wet_hours, window.countable_hours)];
}

/**
 * The bars' parts: the header, the two rows, and the hatch key when any
 * hour is hatched.
 *
 * @param snapshot - The dashboard snapshot, for the tile title.
 * @param wet - The snapshot's wet hours.
 * @returns The parts, in order.
 */
export function meterParts(snapshot: DashboardSnapshot, wet: WetHours): HTMLElement[] {
  const title = make("span", "wet-meter-title", meterTitle(snapshot));
  title.append(howToRead());
  const rows = make("span", "wet-meter-rows");
  rows.append(...row("7 days", wet.last_7_days), ...row("30 days", wet.last_30_days));
  const parts = [title, rows];
  if (wet.last_7_days.missing_hours > 0 || wet.last_30_days.missing_hours > 0) {
    parts.push(hatchKey());
  }
  return parts;
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
  link.append(...meterParts(snapshot, wet), spoken("Open the rain view"));
  return link;
}
