/**
 * The Rain Events tab's view logic, free of the DOM: which events show for
 * a water year and sort, the selected event in the URL, and the words.
 *
 * The numbers come from the snapshot (`rain_events`): the list, the ranks
 * and the per-water-year summaries are computed once in Python. The browser
 * only picks, sorts and words them.
 */
import { formatDuration, formatStamp, inches, stationFormat } from "../format.js";
import { isComparable, notCompared } from "../gaps.js";
import { listText } from "../rain/text.js";
import type { EventSeason, EventStats, RainEventRow, RainEvents } from "../snapshot.js";

/** The URL query parameter holding the selected event's start. */
export const EVENT_PARAM = "event";

/** Which events the list shows: one water year, or all of them. */
export type Season = number | "all";

/** How the list is ordered. */
export type EventSort = "newest" | "wettest";

/**
 * The selected event's start from a page query, or null.
 *
 * @param search - `location.search`.
 * @returns The start in ms, or null when absent or malformed.
 */
export function selectedStart(search: string): number | null {
  const value = new URLSearchParams(search).get(EVENT_PARAM);
  return value !== null && /^\d+$/.test(value) ? Number(value) : null;
}

/**
 * A page query with the selected event set (or cleared), every other
 * parameter kept.
 *
 * @param search - `location.search`.
 * @param startMs - The event's start, or null to clear it.
 * @returns The new query, with its `?`, or "" when empty.
 */
export function withEvent(search: string, startMs: number | null): string {
  const query = new URLSearchParams(search);
  if (startMs === null) query.delete(EVENT_PARAM);
  else query.set(EVENT_PARAM, String(startMs));
  const text = query.toString();
  return text ? `?${text}` : "";
}

/**
 * The events the list shows: one water year's or all, sorted, with the
 * live event pinned on top whatever the sort.
 *
 * @param section - The snapshot's `rain_events`.
 * @param season - A water year, or "all".
 * @param sort - "newest" or "wettest".
 * @returns The rows to list.
 */
export function visibleEvents(
  section: RainEvents,
  season: Season,
  sort: EventSort,
): RainEventRow[] {
  const rows = section.events.filter((e) => season === "all" || e.water_year === season);
  const order =
    sort === "newest"
      ? (a: RainEventRow, b: RainEventRow) => b.start_ms - a.start_ms
      : (a: RainEventRow, b: RainEventRow) => a.rank - b.rank || b.start_ms - a.start_ms;
  return rows.sort((a, b) => Number(b.live) - Number(a.live) || order(a, b));
}

/**
 * The summary for one water year or for all events, as Python computed it.
 *
 * @param section - The snapshot's `rain_events`.
 * @param season - A water year, or "all".
 * @returns Its stats; a zero count for a water year with no events.
 */
export function statsFor(section: RainEvents, season: Season): EventStats {
  if (season === "all") return section.all_time;
  return (
    section.seasons.find((s) => s.water_year === season)?.stats ?? {
      count: 0,
      total_in: 0,
      median_in: null,
      mean_duration_minutes: null,
    }
  );
}

/**
 * The words for a water year with a few days missing, the same in the one
 * year's note and under All.
 *
 * @param season - A ≈ water year's summary.
 * @returns E.g. "3 of 365 days of water year 2025".
 */
function missingDays(season: EventSeason): string {
  return `${season.status.missing_days} of ${season.status.days} days of water year ${season.water_year}`;
}

/**
 * What the list says beside the events-only sum when the record does not
 * cover the water year shown: the list can miss the events in its gaps, so
 * its total is never read as the year's rain. A ≈ year's sum is not scaled
 * up, so it gets the note but no ≈. Each year's status is the snapshot's
 * (`EventSeason.status`); the browser never makes one up.
 *
 * @param section - The snapshot's `rain_events`.
 * @param season - A water year, or "all".
 * @returns The note, or null when every year shown is whole.
 */
export function seasonNote(section: RainEvents, season: Season): string | null {
  if (season === "all") {
    const shown = section.seasons.filter((s) => s.status.status !== "valid");
    const lost = shown.filter((s) => !isComparable(s.status)).map((s) => String(s.water_year));
    const prorated = shown.filter((s) => isComparable(s.status)).map(missingDays);
    if (!lost.length && !prorated.length) return null;
    const many = lost.length > 1;
    const notCovered = `The record does not cover all of water year${many ? "s" : ""} ${listText(lost)}`;
    if (!prorated.length) return `${notCovered}, so the list can miss ${many ? "their" : "its"} events.`;
    const few = listText(prorated);
    if (!lost.length) return `${few} are missing from the record, so the list can miss events in them.`;
    return `${notCovered}, and ${few} are missing from it, so the list can miss events in them.`;
  }
  const found = section.seasons.find((s) => s.water_year === season);
  if (!found) return null;
  const { status } = found;
  if (status.status === "valid") return null;
  if (isComparable(status)) {
    return `${missingDays(found)} are missing from the record, so the list can miss events in them.`;
  }
  return `Water year ${season} ${notCompared(status.reason)}, so the list can miss its events and their total is not the year's rain.`;
}

/**
 * Which edges of a scrolling control have content hidden past them.
 *
 * @param scrollLeft - Its scroll position.
 * @param scrollWidth - Its content's width.
 * @param clientWidth - Its visible width.
 * @returns "none", "start", "end" or "both".
 */
export function scrollEdges(
  scrollLeft: number,
  scrollWidth: number,
  clientWidth: number,
): "none" | "start" | "end" | "both" {
  // A pixel of slack: a fractional scroll position never quite reaches the end.
  const start = scrollLeft > 1;
  const end = scrollLeft + clientWidth < scrollWidth - 1;
  return start && end ? "both" : start ? "start" : end ? "end" : "none";
}

/** A duration given in minutes, in the One Format Rule's words. */
export function minutes(value: number): string {
  return formatDuration(value * 60_000);
}

/** `1st`, `2nd`, `3rd`, `4th`, `11th`, `22nd`. */
export function ordinal(n: number): string {
  const teen = n % 100 >= 11 && n % 100 <= 13;
  const suffix = teen ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] ?? "th");
  return `${n}${suffix}`;
}

/** The year formatter in the station's zone. */
const yearOf = () => stationFormat("en-US", { year: "numeric" });

/**
 * An event's start for the list: `Jan 8, 2:00 pm`, with the year when it
 * is not this year's (`Jan 8, 2025, 2:00 pm`).
 *
 * @param ms - The start.
 * @param nowMs - The browser's clock.
 * @returns The label.
 */
export function startLabel(ms: number, nowMs: number): string {
  const stamp = formatStamp(ms);
  const year = yearOf().format(ms);
  if (year === yearOf().format(nowMs)) return stamp;
  const [day, time] = stamp.split(", ");
  return `${day}, ${year}, ${time}`;
}

/**
 * The quality chip's words: the rating, or "no data" for an event the
 * archive holds no readings for.
 *
 * @param row - The event.
 * @returns The words.
 */
export function qualityLabel(row: RainEventRow): string {
  return row.quality === "invalid" ? "no data" : row.quality;
}

/** Whether the quality chip is drawn as low (dashed). */
export function lowQuality(row: RainEventRow): boolean {
  return row.quality === "poor" || row.quality === "invalid";
}

/**
 * The selected event's finding headline, with its number in it.
 *
 * @param row - The event.
 * @param count - How many events the catalog lists.
 * @returns The headline.
 */
export function eventFinding(row: RainEventRow, count: number): string {
  if (row.live) {
    return `${inches(row.total_in, { digits: 2 })} so far over ${minutes(row.duration_minutes)}, and still raining`;
  }
  const place = row.rank === 1 ? "the wettest" : `the ${ordinal(row.rank)} wettest`;
  return `${inches(row.total_in, { digits: 2 })} over ${minutes(row.duration_minutes)}, ${place} of ${count} event${count === 1 ? "" : "s"}`;
}

/**
 * The list's finding headline for a water year or for all events: the
 * wettest event against the median one, the count as their context.
 *
 * @param stats - The summary.
 * @param season - A water year, or "all".
 * @param oldestMs - The oldest listed event's start, for "since".
 * @param rows - The events listed for that water year or for all.
 * @returns The headline.
 */
export function listFinding(
  stats: EventStats,
  season: Season,
  oldestMs: number | null,
  rows: RainEventRow[],
): string {
  const n = stats.count;
  const scope =
    season !== "all"
      ? ` in water year ${season}`
      : oldestMs === null
        ? ""
        : ` since ${formatStamp(oldestMs).split(",")[0]}, ${yearOf().format(oldestMs)},`;
  const wettest = rows.reduce<RainEventRow | null>(
    (best, e) => (best === null || e.total_in > best.total_in ? e : best),
    null,
  );
  if (n === 0 || wettest === null) return `No rain events${scope}`;
  const total = `${inches(wettest.total_in, { digits: 2 })}${wettest.live ? " so far" : ""}`;
  if (n === 1) return `The one rain event${scope} brought ${total}`;
  const median =
    stats.median_in === null
      ? ""
      : `, against ${inches(stats.median_in, { digits: 2 })} for the median event`;
  return `The wettest of ${n} events${scope} brought ${total}${median}`;
}
