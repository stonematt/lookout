/**
 * The shared gap component: hatches a month or window that cannot be
 * compared, with the reason in place.
 *
 * The snapshot gives every month and window one gap status (`GapStatus`,
 * from `lookout/snapshot/gaps.py`): valid, prorated or invalid. DESIGN.md's
 * One Gap Status Rule says how each one shows:
 *
 * - invalid: the mark is replaced by the 135° hatch, with no value under it
 *   and a short reason in place ("n/a: gap in record", "not archived yet"),
 *   the full sentence in a title and in screen-reader text. `hatchGap` does
 *   this to a cell or bar slot; `gapElement` makes a standalone one.
 * - prorated: the value is shown with ≈ before it (`gapValue`), never called
 *   "full", and the sentence goes in a title.
 * - valid: shown as it is.
 *
 * Every view that shows a month or window value goes through `gapValue`
 * (null means: draw nothing, hatch it) and `hatchGap`.
 */
import "./styles/gaps.css";

import type { GapReason, GapStatus } from "./snapshot.js";

/** The short reason shown in place on the hatch. */
const LABELS: Record<GapReason, string> = {
  gap: "n/a: gap in record",
  low_coverage: "n/a: low coverage",
  not_archived: "not archived yet",
  before_archive: "n/a: before the archive",
  in_progress: "n/a: still running",
};

/** What a compact cell (an almanac square, a narrow table cell) reads. */
const COMPACT_LABEL = "n/a";

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** Options for `hatchGap` and `gapElement`. */
export interface HatchOptions {
  /** Show just "n/a" in place, for a cell too small for the reason. */
  compact?: boolean;
}

/**
 * Whether a month's or window's value may be shown at all.
 *
 * @param status - Its gap status.
 * @returns True when valid or prorated.
 */
export function isComparable(status: GapStatus): boolean {
  return status.status !== "invalid";
}

/**
 * The short reason a hatched gap shows in place.
 *
 * @param status - A month's or window's gap status.
 * @returns "n/a: gap in record", "not archived yet" and so on; null when it
 *   can be compared.
 */
export function gapLabel(status: GapStatus): string | null {
  if (isComparable(status)) return null;
  return LABELS[status.reason ?? "gap"];
}

/**
 * The full sentence for a title and for screen readers: which days, and why.
 *
 * @param status - A month's or window's gap status.
 * @returns E.g. "Jan 8, 2026: the archive does not reach this far yet."
 */
export function gapSentence(status: GapStatus): string {
  const span = daySpan(status.first_day, status.last_day);
  if (status.status === "valid") return `${span}: every day is in the record.`;
  switch (status.reason) {
    case "not_archived":
      return `${span}: the archive does not reach this far yet.`;
    case "before_archive":
      return `${span}: it starts before the archive began, so it is not compared.`;
    case "in_progress":
      return `${span}: it is not over yet, so it is not compared.`;
  }
  const missing =
    status.days === 1
      ? status.reason === "low_coverage"
        ? "under 80% of the day is in the record"
        : "the day is missing from the record"
      : status.reason === "low_coverage"
        ? `${status.missing_days} of ${status.days} days have under 80% of their readings`
        : `${status.missing_days} of ${status.days} days are missing from the record`;
  if (status.status === "prorated") {
    const counted = status.days - status.missing_days;
    return `${span}: ${missing}, so its total is scaled up from the other ${counted} (≈).`;
  }
  return `${span}: ${missing}, so it is not compared.`;
}

/**
 * A formatted value as its gap status allows it to be shown.
 *
 * @param status - The month's or window's gap status.
 * @param text - Its formatted value, e.g. "3.2 in".
 * @returns The text; "≈ " before it when prorated; null when invalid.
 */
export function gapValue(status: GapStatus, text: string): string | null {
  if (!isComparable(status)) return null;
  return status.status === "prorated" ? `≈ ${text}` : text;
}

/**
 * Hatch an element in place when its month or window is invalid.
 *
 * An invalid one loses its content for the hatch, the short reason and the
 * spoken sentence. A comparable one keeps its content; a prorated one is
 * marked `data-gap="prorated"`, with the sentence in its title and as
 * screen-reader text.
 *
 * @param el - A table cell, almanac cell, bar slot or chart row.
 * @param status - Its gap status.
 * @param options - `compact` for "n/a" alone in place.
 * @returns True when it was hatched.
 */
export function hatchGap(
  el: HTMLElement,
  status: GapStatus,
  options: HatchOptions = {},
): boolean {
  el.querySelector(":scope > .gap-sr")?.remove();
  if (isComparable(status)) {
    el.dataset.gap = status.status;
    if (status.status === "prorated") {
      el.title = gapSentence(status);
      el.append(spokenSentence(status));
    }
    return false;
  }
  const sentence = gapSentence(status);
  const label = document.createElement("span");
  label.className = "gap-label";
  label.setAttribute("aria-hidden", "true");
  label.textContent = options.compact ? COMPACT_LABEL : gapLabel(status);
  el.replaceChildren(label, spokenSentence(status));
  el.classList.add("gap-hatch", "gap");
  el.dataset.gap = status.reason ?? "gap";
  el.title = sentence;
  return true;
}

/**
 * A standalone hatched block, for a chart row or bar with no element of its
 * own. For a comparable status the block is empty and unhatched.
 *
 * @param status - The month's or window's gap status.
 * @param options - `compact` for "n/a" alone in place.
 * @returns A `span.gap-hatch` holding the reason.
 */
export function gapElement(status: GapStatus, options: HatchOptions = {}): HTMLElement {
  const block = document.createElement("span");
  block.classList.add("gap-block");
  hatchGap(block, status, options);
  return block;
}

/** The sentence as screen-reader text. */
function spokenSentence(status: GapStatus): HTMLElement {
  const spoken = document.createElement("span");
  spoken.className = "gap-sr";
  spoken.textContent = ` ${gapSentence(status)}`;
  return spoken;
}

/** "Jan 8, 2026", "Jan 1 to Jan 31, 2026" or "Dec 13, 2025 to Jan 11, 2026". */
function daySpan(first: string, last: string): string {
  const end = `${monthDay(last)}, ${last.slice(0, 4)}`;
  if (first === last) return end;
  const sameYear = first.slice(0, 4) === last.slice(0, 4);
  const start = sameYear ? monthDay(first) : `${monthDay(first)}, ${first.slice(0, 4)}`;
  return `${start} to ${end}`;
}

/** "Jan 8" from "2026-01-08". */
function monthDay(day: string): string {
  return `${MONTHS[Number(day.slice(5, 7)) - 1]} ${Number(day.slice(8, 10))}`;
}
