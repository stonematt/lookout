/**
 * History grows on the Overview (#470, spec #460 stories 28–32, PRODUCT.md
 * Principle 4): a comparison appears only once the record supports it, and
 * one line names the next unlock.
 *
 * Python decides all of it (`lookout/snapshot/depth.py`, ADR-0003): each gap
 * status carries `shown`, and `record_depth.next_unlock` names the one
 * comparison the record unlocks next and the day it does. This module only
 * reads those and words the line:
 *
 * - `WINDOW_NAMES`: each comparison's name, shared with the window chart.
 * - `isShown`: whether a month's or window's status is shown. A status
 *   without `shown` (a snapshot written before Release A, ADR-0008) is
 *   shown, so such a snapshot draws today's view.
 * - `nextUnlockText`: the line's words, a fixed template over the key and
 *   the day; null when the snapshot names no unlock.
 * - `nextUnlockLine`: the line as an element, or null: no line, no empty
 *   element.
 */
import { dayText } from "../format.js";
import { make } from "../dom.js";
import type { DashboardSnapshot, GapStatus } from "../snapshot.js";

/**
 * Whether the record is deep enough to show a month or window.
 *
 * @param status - Its gap status.
 * @returns False only when the snapshot says it is not shown.
 */
export function isShown(status: GapStatus): boolean {
  return status.shown !== false;
}

/**
 * Each comparison's name, by its `record_depth` key: the window chart's row
 * labels and the next-unlock line's.
 */
export const WINDOW_NAMES: Readonly<Record<string, string>> = {
  today: "Today",
  yesterday: "Yesterday",
  "7d": "7 days",
  "30d": "30 days",
  "90d": "90 days",
  "365d": "365 days",
  water_year: "Since Oct 1",
  earlier_year: "This date in earlier years",
  full_history: "Full history",
};

/**
 * The next-unlock line's words.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Next to unlock: 7 days, on Nov 23, 2025."; null when the
 *   snapshot names no next unlock (or predates `record_depth`).
 */
export function nextUnlockText(snapshot: DashboardSnapshot): string | null {
  const next = snapshot.record_depth?.next_unlock;
  if (!next) return null;
  const on = dayText(next.date, true);
  const name = WINDOW_NAMES[next.key];
  return name ? `Next to unlock: ${name}, on ${on}.` : `More of the record unlocks on ${on}.`;
}

/**
 * The next-unlock line, for the tile whose rows it adds to.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns A `p.ov-unlock`, or null when there is nothing to say.
 */
export function nextUnlockLine(snapshot: DashboardSnapshot): HTMLElement | null {
  const text = nextUnlockText(snapshot);
  if (text === null) return null;
  const line = make("p", "ov-unlock", text);
  line.dataset.part = "next-unlock";
  return line;
}
