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
 *
 * The same hatch marks a span inside a chart: `svgHatch` gives an SVG chart
 * its pattern, `hatchSpan` is a hatched span for an HTML band to place, and
 * `plotlyHatch` is the pattern for a Plotly chart.
 */
import "./styles/gaps.css";

import { svgEl } from "./dom.js";
import { dayText } from "./format.js";
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

/** The hatch's repeat and its stripe (DESIGN.md: 4px on an 8px repeat). */
const HATCH_REPEAT_PX = 8;
const HATCH_STRIPE_PX = 4;

/**
 * A one-day gap status, so a day's reason reads like any window's.
 *
 * @param day - "YYYY-MM-DD".
 * @param reason - Why the day is not in the record.
 * @returns An invalid status over that day.
 */
export function dayStatus(day: string, reason: GapReason): GapStatus {
  return {
    key: day,
    first_day: day,
    last_day: day,
    days: 1,
    missing_days: 1,
    status: "invalid",
    reason,
  };
}

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
  return reasonLabel(status.reason ?? "gap");
}

/**
 * The short reason in place for a reason alone, e.g. a day's in a chart.
 *
 * @param reason - Why it is not in the record.
 * @returns "n/a: gap in record", "not archived yet" and so on.
 */
export function reasonLabel(reason: GapReason): string {
  return LABELS[reason];
}

/** Why a window or a water year is not compared: [one, many] verb phrases. */
const NOT_COMPARED: Record<string, [string, string]> = {
  not_archived: ["is not archived yet", "are not archived yet"],
  before_archive: ["starts before the archive began", "start before the archive began"],
  gap: ["has days missing from the record", "have days missing from the record"],
  low_coverage: [
    "has days with under 80% of their readings",
    "have days with under 80% of their readings",
  ],
  in_progress: ["is not over yet", "are not over yet"],
};

/**
 * Why a window or a water year is not compared, as a verb phrase.
 *
 * @param reason - Its gap reason; null reads as a gap in the record.
 * @param many - The plural, for several named together.
 * @returns E.g. "starts before the archive began".
 */
export function notCompared(reason: string | null, many = false): string {
  const words = NOT_COMPARED[reason ?? "gap"] ?? NOT_COMPARED.gap!;
  return words[many ? 1 : 0];
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
    // Undo an earlier hatch, so an element can follow a status that changes.
    if (el.classList.contains("gap-hatch")) el.replaceChildren();
    el.classList.remove("gap-hatch", "gap");
    el.removeAttribute("title");
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
  const end = dayText(last, true);
  if (first === last) return end;
  const sameYear = first.slice(0, 4) === last.slice(0, 4);
  return `${dayText(first, !sameYear)} to ${end}`;
}

let patternCount = 0;

/**
 * Give an SVG chart the 135° hatch as a pattern of its own (`gap-hatch-a`
 * stripes over `gap-hatch-b`, coloured in gaps.css), for the marks that
 * stand for what is not in the record.
 *
 * @param root - The chart's SVG.
 * @returns The fill that paints with it, `url(#...)`.
 */
export function svgHatch(root: SVGSVGElement): string {
  const id = `gap-hatch-${++patternCount}`;
  const pattern = svgEl("pattern", {
    id,
    width: HATCH_REPEAT_PX,
    height: HATCH_REPEAT_PX,
    patternUnits: "userSpaceOnUse",
    patternTransform: "rotate(45)",
  });
  pattern.append(
    svgEl("rect", { width: HATCH_REPEAT_PX, height: HATCH_REPEAT_PX, class: "gap-hatch-b" }),
    svgEl("rect", { width: HATCH_STRIPE_PX, height: HATCH_REPEAT_PX, class: "gap-hatch-a" }),
  );
  const defs = svgEl("defs");
  defs.append(pattern);
  root.append(defs);
  return `url(#${id})`;
}

/**
 * A hatched span for a band to place over the stretch it cannot speak for.
 *
 * @param reason - Why, e.g. "not archived yet"; kept as `data-reason`.
 * @returns A `div.gap-hatch`, unpositioned.
 */
export function hatchSpan(reason: string): HTMLElement {
  const span = document.createElement("div");
  span.className = "gap-hatch";
  span.dataset.reason = reason;
  return span;
}

/** A Plotly bar's pattern. */
export interface PlotlyHatch {
  /** Plotly's backslash: stripes falling left to right, the 135° hatch. */
  shape: "\\";
  fgcolor: string;
  bgcolor: string;
  size: number;
  solidity: number;
}

/**
 * The 135° hatch as a Plotly bar pattern.
 *
 * @param gapA - The stripe colour (the `--gap-a` token's value).
 * @param gapB - The ground colour (the `--gap-b` token's value).
 * @returns The `marker.pattern`.
 */
export function plotlyHatch(gapA: string, gapB: string): PlotlyHatch {
  return { shape: "\\", fgcolor: gapA, bgcolor: gapB, size: HATCH_REPEAT_PX, solidity: 0.45 };
}
