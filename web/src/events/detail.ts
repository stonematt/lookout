/**
 * The open event row's body: its finding, its stats, the Total / Rate
 * switch, the chart's host, its quality and its source fold. The tab
 * (`tab.ts`) decides which curve to draw and when; this module says what the
 * body reads. Markup goes through `html` (`web/src/dom.ts`).
 *
 * - `stat`: one stat-strip cell, shared with the catalog tile.
 * - `eventBody`: the body's markup around the empty chart host.
 * - `eventSource`: the source fold's what, method and gaps for the event.
 * - `chartLabel`: the chart's accessible name, Total or Rate.
 */
import { html } from "../dom.js";
import type { Html } from "../dom.js";
import { inches, inchesPerHour } from "../format.js";
import type { RainEventRow } from "../snapshot.js";
import type { EventChart } from "./charts.js";
import { eventFinding, minutes } from "./model.js";

/**
 * A one-row stat strip cell.
 *
 * @param value - The figure.
 * @param label - What it counts.
 * @returns The cell's markup.
 */
export function stat(value: string, label: string): Html {
  return html`<div>
    <dt>${label}</dt>
    <dd>${value}</dd>
  </div>`;
}

/** The open event's quality line. */
function qualityLine(row: RainEventRow): string {
  if (row.quality === "invalid") return "No archive readings.";
  const share =
    row.completeness_pct === null ? "" : `, ${Math.round(row.completeness_pct)}% of readings`;
  const gap = row.max_gap_minutes ? `, longest gap ${minutes(row.max_gap_minutes)}` : "";
  return `Quality ${row.quality}${share}${gap}.`;
}

/**
 * The open row's body, around an empty `.ev-charts` the tab draws into.
 *
 * @param row - The open event.
 * @param count - Events in the catalog, for its finding.
 * @param chart - The chart picked, Total or Rate.
 * @param missing - The linked event is not in the catalog: say so.
 * @returns The body's markup.
 */
export function eventBody(
  row: RainEventRow,
  count: number,
  chart: EventChart,
  missing: boolean,
): Html {
  const note = missing
    ? html`<p class="ev-note">The linked event is not in the catalog; showing the newest one instead.</p>`
    : "";
  const switcher = row.has_curve
    ? html`<div class="ev-seg ev-mode" role="group" aria-label="Chart">
  <button type="button" data-chart="total" aria-pressed="${chart === "total"}">Total</button>
  <button type="button" data-chart="rate" aria-pressed="${chart === "rate"}">Rate</button>
</div>`
    : "";
  const peak =
    row.peak_rate_in_hr === null || !row.has_curve ? "n/a" : inchesPerHour(row.peak_rate_in_hr);
  return html`
${note}
<p class="ev-finding" aria-live="polite">${eventFinding(row, count)}</p>
<dl class="ev-facts">
  ${stat(inches(row.total_in, { digits: 2 }), row.live ? "so far (live)" : "total")}
  ${stat(minutes(row.duration_minutes), "length")}
  ${stat(peak, "peak rate")}
</dl>
${switcher}
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
        "The live station for the newest reading and the total; the station archive for the rest. 5-minute readings, station time. Total is the event's rain as it built up, against the median event's total, dashed. Rate is the rise in daily rain between readings.",
      gaps: null,
    };
  }
  return {
    what: "station archive, 5-minute readings",
    method:
      "The station archive, 5-minute readings, station time. Total is the event's rain as it built up, against the median event's total, dashed. Rate is the rise in daily rain between readings.",
    gaps: null,
  };
}

/**
 * The open row's chart's accessible name.
 *
 * @param row - The open event.
 * @param chart - Total or Rate.
 * @param medianIn - The median event's total in the water year shown.
 * @returns The label.
 */
export function chartLabel(row: RainEventRow, chart: EventChart, medianIn: number | null): string {
  if (chart === "total") {
    const against =
      medianIn === null ? "" : `, against ${inches(medianIn, { digits: 2 })} for the median event`;
    return `Rain so far in this event, rising to ${inches(row.total_in, { digits: 2 })}${against}`;
  }
  return row.peak_rate_in_hr === null
    ? "Rain rate between readings"
    : `Rain rate between readings, peaking at ${inchesPerHour(row.peak_rate_in_hr)}`;
}
