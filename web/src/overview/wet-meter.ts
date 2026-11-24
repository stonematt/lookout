/**
 * The wet-hour meter (#657, spec #647 "Meter bars"): two bars at the foot
 * of the now tile, the last 7 days and the last 30, each a tally of every
 * hour in its window, not a timeline. The dark fill is the wet hours; the
 * hours the record cannot count are hatched at the bar's end, never drawn
 * as dry. Each bar reads wet hours of countable hours ("19 h of 168"), never
 * a percent, and a bar with missing hours says so under it: "24 h missing,
 * not dry · 1 d missing".
 *
 * The meter is one link to the Rain tab. Every number is the snapshot's
 * `wet_hours` (GLOSSARY wet hour, countable hour); a snapshot without it
 * (N−1, ADR-0008) draws no meter, and nor does dry mode (#658).
 *
 * Offers `wetMeter`, `meterShown` and `missingWords`.
 */
import "./wet-meter.css";

import { make, spoken } from "../dom.js";
import type { DashboardSnapshot, WetHoursWindow } from "../snapshot.js";

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
 * One window's row: its name, its bar, its count, and what it is missing.
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
  const cells = [label, bar, count];
  const words = missingWords(window);
  if (words.length) {
    const note = make("span", "wet-meter-note");
    note.dataset.part = "missing";
    // Each phrase stays whole when the note wraps on a narrow phone.
    const phrases = make("span", "wet-meter-words");
    words.forEach((phrase, i) => {
      if (i > 0) phrases.append(" · ");
      phrases.append(make("span", "wet-meter-phrase", phrase));
    });
    note.append(make("span", "hatch-swatch"), phrases);
    cells.push(note);
  }
  return cells;
}

/**
 * Whether the meter is drawn: the snapshot has wet hours and is not in dry
 * mode (#658), where the meter would read 0 h for months. Without a mode
 * (N−1) it is drawn as before.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns True to draw it.
 */
export function meterShown(snapshot: DashboardSnapshot): boolean {
  return Boolean(snapshot.wet_hours) && snapshot.wet_mode?.mode !== "dry";
}

/**
 * The meter, or null when it is not drawn (`meterShown`).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The meter's link, to the Rain tab.
 */
export function wetMeter(snapshot: DashboardSnapshot): HTMLElement | null {
  const wet = snapshot.wet_hours ?? null;
  if (!wet || !meterShown(snapshot)) return null;
  const link = make("a", "wet-meter");
  link.href = "#rain";
  link.dataset.part = "wet-meter";
  const rows = make("span", "wet-meter-rows");
  rows.append(...row("7 days", wet.last_7_days), ...row("30 days", wet.last_30_days));
  link.append(make("span", "wet-meter-title", "Wet hours"), rows, spoken("Open the rain view"));
  return link;
}
