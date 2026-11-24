/**
 * The open event row's body: its finding, its stats, the chart's host, its
 * quality and its source fold. The tab
 * (`tab.ts`) decides which curve to draw and when; this module says what the
 * body reads. Markup goes through `html` (`web/src/dom.ts`).
 *
 * - `eventBody`: the body's markup around the empty chart host, with the
 *   note for a missing event (`MissingEvent`).
 * - `eventSource`: the source fold's what, method and gaps for the event
 *   (`EventSourceParts`).
 * - `chartLabel`: the chart's accessible name.
 */
import { html } from "../dom.js";
import type { Html } from "../dom.js";
import { inches, inchesPerHour } from "../format.js";
import type { RainEventRow } from "../snapshot.js";
import { statCell as stat } from "../tiles.js";
import { eventFinding, minutes, peakText } from "./model.js";

/** The open event's quality line. */
function qualityLine(row: RainEventRow): string {
  if (row.quality === "invalid") return "No archive readings.";
  const share =
    row.completeness_pct === null ? "" : `, ${Math.round(row.completeness_pct)}% of readings`;
  const gap = row.max_gap_minutes ? `, longest gap ${minutes(row.max_gap_minutes)}` : "";
  return `Quality ${row.quality}${share}${gap}.`;
}

/**
 * Why the open event is not the one the URL named: the link names an event
 * the catalog never held for this visitor ("linked"), or the event the
 * visitor had open here has left the catalog since ("vanished").
 */
export type MissingEvent = "linked" | "vanished" | null;

/** The note over the open row's finding, one per kind of missing event. */
const MISSING_NOTES: Record<"linked" | "vanished", string> = {
  linked: "The linked event is not in the catalog; showing the newest one instead.",
  vanished: "The event you had open is no longer in the catalog; showing the newest one instead.",
};

/**
 * The open row's body, around an empty `.ev-charts` the tab draws into.
 *
 * @param row - The open event.
 * @param count - Events in the catalog, for its finding.
 * @param missing - The event the URL named is not in the catalog: say why.
 * @param announce - The row is newly shown, so its finding is a live region;
 *   a newer snapshot redrawing the same row does not announce it again.
 * @returns The body's markup.
 */
export function eventBody(
  row: RainEventRow,
  count: number,
  missing: MissingEvent,
  announce: boolean,
): Html {
  const note = missing ? html`<p class="ev-note">${MISSING_NOTES[missing]}</p>` : "";
  const peak = peakText(row);
  const finding = eventFinding(row, count);
  return html`
${note}
${announce ? html`<p class="ev-finding" aria-live="polite">${finding}</p>` : html`<p class="ev-finding">${finding}</p>`}
<dl class="ev-facts">
  ${stat(inches(row.total_in, { digits: 2 }), row.live ? "so far (live)" : "total")}
  ${stat(minutes(row.duration_minutes), "length")}
  ${stat(peak, "peak rate")}
</dl>
<div class="ev-charts"></div>
<p class="ev-quality">${qualityLine(row)}</p>`;
}

/** The source fold's parts for an open event. */
export interface EventSourceParts {
  what: string;
  method: string;
  gaps: string | null;
}

/**
 * Where the open event's chart comes from: the catalog alone for an event
 * with no archive data, the live station and the archive for the live one,
 * the archive for any other.
 *
 * @param row - The open event.
 * @returns The fold's what, method and gaps.
 */
export function eventSource(row: RainEventRow): EventSourceParts {
  if (!row.has_curve) {
    return {
      what: "Event Catalog",
      method: "The Event Catalog. The archive holds no readings in this event's time range.",
      gaps: "no archive readings",
    };
  }
  if (row.live) {
    return {
      what: "live station and archive, 5-minute readings",
      method:
        "The live station for the newest reading and the total; the station archive for the rest. 5-minute readings, station time. One chart: the columns are the rain per 5 minutes (the rise in daily rain between readings), averaged over wider spans where 5-minute columns would run together, on the left axis; the line is the event's rain as it built up, on the right axis at ten times the left, against the median event's total, dashed.",
      gaps: null,
    };
  }
  return {
    what: "station archive, 5-minute readings",
    method:
      "The station archive, 5-minute readings, station time. One chart: the columns are the rain per 5 minutes (the rise in daily rain between readings), averaged over wider spans where 5-minute columns would run together, on the left axis; the line is the event's rain as it built up, on the right axis at ten times the left, against the median event's total, dashed.",
    gaps: null,
  };
}

/**
 * The open row's chart's accessible name: the running total against the
 * median event, then the columns and their peak rate.
 *
 * @param row - The open event.
 * @param medianIn - The median event's total in the water year shown.
 * @returns The label.
 */
export function chartLabel(row: RainEventRow, medianIn: number | null): string {
  const against =
    medianIn === null ? "" : `, against ${inches(medianIn, { digits: 2 })} for the median event`;
  const peak =
    row.peak_rate_in_hr === null ? "" : `, peaking at ${inchesPerHour(row.peak_rate_in_hr)}`;
  return `Rain so far in this event, rising to ${inches(row.total_in, { digits: 2 })}${against}; columns of the rain between readings under it${peak}`;
}
