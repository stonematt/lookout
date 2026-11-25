/**
 * History grows on the Overview (#470, spec #460 stories 28–32, PRODUCT.md
 * Principle 4): a comparison appears only once the record supports it.
 * What the record does not cover yet is not drawn and not explained (#720).
 *
 * Python decides all of it (`lookout/snapshot/depth.py`, ADR-0003): each gap
 * status carries `shown`. This module only reads it (`record_depth.next_unlock`
 * stays in the contract but is not drawn, #720):
 *
 * - `WINDOW_NAMES`: each comparison's name, shared with the window chart.
 * - `isShown`: whether a month's or window's status is shown. A status
 *   without `shown` (a snapshot written before Release A, ADR-0008) is
 *   shown, so such a snapshot draws today's view.
 */
import type { GapStatus } from "../snapshot.js";

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
 * labels.
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
