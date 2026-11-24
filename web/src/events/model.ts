/**
 * The Rain Events tab's view logic, free of the DOM: which events show for
 * a water year and sort, the selected event in the URL, and the words.
 *
 * The numbers come from the snapshot (`rain_events`): the list, the ranks
 * and the per-water-year summaries are computed once in Python. The browser
 * only picks, sorts and words them.
 */
import {
  formatDuration,
  formatStamp,
  inches,
  inchesPerHour,
  ordinal,
  stationFormat,
} from "../format.js";
import { beyondRecord, isComparable, notCompared } from "../gaps.js";
import { listText } from "../rain/text.js";
import type {
  EventCategory,
  EventSeason,
  EventStats,
  RainEventRow,
  RainEvents,
} from "../snapshot.js";

/** The URL query parameter holding the selected event's start. */
export const EVENT_PARAM = "event";

/** Which events the list shows: one water year, or all of them. */
export type Season = number | "all";

/** How the list is ordered. */
export type EventSort = "newest" | "wettest" | "longest";

/** The sorts, in the control's order, with their words. */
export const SORTS: readonly { sort: EventSort; label: string }[] = [
  { sort: "newest", label: "Newest" },
  { sort: "wettest", label: "Wettest" },
  { sort: "longest", label: "Longest" },
];

/** The categories, smallest first: a category's step on the glyph is its place. */
export const CATEGORIES: readonly { category: EventCategory; label: string }[] = [
  { category: "sprinkle", label: "sprinkle" },
  { category: "shower", label: "shower" },
  { category: "rain", label: "rain" },
  { category: "storm", label: "storm" },
];

/**
 * The size filter: the smallest category the list shows, or every event.
 * Python decides each event's category; the browser only compares it.
 */
export type EventSize = "all" | "shower" | "rain" | "storm";

/** The sizes, in the filter's order, with their words and what they hide. */
export const SIZES: readonly { size: EventSize; label: string; hides: string | null }[] = [
  { size: "all", label: "All sizes", hides: null },
  { size: "shower", label: "Showers and up", hides: "sprinkles" },
  { size: "rain", label: "Rain and up", hides: "showers and sprinkles" },
  { size: "storm", label: "Storms only", hides: "smaller events" },
];

/** The size a first visit shows: showers and up, sprinkles hidden (#644). */
export const DEFAULT_SIZE: EventSize = "shower";

/**
 * A category's step on the four-step glyph: 1 for a sprinkle to 4 for a
 * storm.
 *
 * @param category - The event's category.
 * @returns The step.
 */
export function categoryStep(category: EventCategory): number {
  return CATEGORIES.findIndex((c) => c.category === category) + 1;
}

/**
 * Whether the size filter lets a row through. The live event always shows,
 * and so does a row with no category (a snapshot written before it).
 *
 * @param row - The event.
 * @param size - The size picked.
 * @returns True to list it.
 */
export function inSize(row: RainEventRow, size: EventSize): boolean {
  if (size === "all" || row.live || row.category == null) return true;
  return categoryStep(row.category) >= categoryStep(size);
}

/**
 * The size that lists an event of this category and as few smaller ones
 * as it can.
 *
 * @param category - The event's category.
 * @returns "all" for a sprinkle, the category itself otherwise.
 */
export function sizeFor(category: EventCategory): EventSize {
  return category === "sprinkle" ? "all" : category;
}

/**
 * A row's category word, e.g. "shower", or null when the snapshot has none.
 *
 * @param row - The event.
 * @returns The word.
 */
export function categoryLabel(row: RainEventRow): string | null {
  return CATEGORIES.find((c) => c.category === row.category)?.label ?? null;
}

/**
 * A water year as the filter names it: water year 2027 is "2026–27".
 *
 * @param waterYear - The water year, named by the year it ends in.
 * @returns The span.
 */
export function seasonLabel(waterYear: number): string {
  // Anything but a year (a snapshot this version cannot read) as it is.
  if (!Number.isInteger(waterYear)) return String(waterYear);
  return `${waterYear - 1}–${String(waterYear % 100).padStart(2, "0")}`;
}

/**
 * The water year a list opens on: the newest the snapshot holds, or all
 * when it holds none.
 *
 * @param section - The snapshot's `rain_events`.
 * @returns The water year, or "all".
 */
export function currentSeason(section: RainEvents): Season {
  const [first, ...rest] = section.seasons;
  if (!first) return "all";
  return rest.reduce((a, s) => (s.water_year > a ? s.water_year : a), first.water_year);
}

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
 * The events the list shows: one water year's or all, of the size
 * picked, sorted, with the live event pinned on top whatever the sort.
 *
 * @param section - The snapshot's `rain_events`.
 * @param season - A water year, or "all".
 * @param sort - "newest" by start, "wettest" by total (Python's rank) or
 *   "longest" by the duration each row carries; ties go newest first.
 * @param size - The size picked; every event when omitted.
 * @returns The rows to list.
 */
export function visibleEvents(
  section: RainEvents,
  season: Season,
  sort: EventSort,
  size: EventSize = "all",
): RainEventRow[] {
  const rows = inSeason(section, season, size);
  const key: Record<EventSort, (a: RainEventRow, b: RainEventRow) => number> = {
    newest: () => 0,
    wettest: (a, b) => a.rank - b.rank,
    longest: (a, b) => b.duration_minutes - a.duration_minutes,
  };
  const order = key[sort];
  return rows.sort(
    (a, b) => Number(b.live) - Number(a.live) || order(a, b) || b.start_ms - a.start_ms,
  );
}

/** One water year's events, or all of them, of the size picked, in the snapshot's order. */
function inSeason(section: RainEvents, season: Season, size: EventSize): RainEventRow[] {
  return section.events.filter(
    (e) => (season === "all" || e.water_year === season) && inSize(e, size),
  );
}

/**
 * The event open by default in a water year or in all: the live one, or
 * with none live, the newest by start, of the size picked.
 *
 * @param section - The snapshot's `rain_events`.
 * @param season - A water year, or "all".
 * @param size - The size picked; every event when omitted.
 * @returns The event, or null when the filter leaves none.
 */
export function defaultOpen(
  section: RainEvents,
  season: Season,
  size: EventSize = "all",
): RainEventRow | null {
  const rows = inSeason(section, season, size);
  return (
    rows.find((e) => e.live) ??
    rows.reduce<RainEventRow | null>(
      (a, e) => (a === null || e.start_ms > a.start_ms ? e : a),
      null,
    )
  );
}

/**
 * The catalog event that carries the "latest" chip: the newest by start,
 * and only when no event is live (the live mark says it then).
 *
 * @param section - The snapshot's `rain_events`.
 * @returns Its start, or null.
 */
export function latestStart(section: RainEvents): number | null {
  if (section.events.some((e) => e.live)) return null;
  return defaultOpen(section, "all")?.start_ms ?? null;
}

/**
 * A row's size bar: its total over the catalog's largest total, 0 to 1.
 *
 * @param row - The event.
 * @param most - The catalog's largest event total.
 * @returns The share of the track it fills.
 */
export function sizeShare(row: RainEventRow, most: number): number {
  return most > 0 ? Math.min(1, Math.max(0, row.total_in / most)) : 0;
}

/**
 * What the list's empty state says when a filter leaves no events.
 *
 * @param season - A water year, or "all".
 * @param hidden - Events the size filter hides there.
 * @returns The sentence (DESIGN.md "Event list").
 */
export function emptyList(season: Season, hidden = 0): string {
  const scope = season === "all" ? "" : ` in ${seasonLabel(season)}`;
  if (hidden > 0) {
    return `No events this size${scope}; ${hidden} smaller ${hidden === 1 ? "one is" : "ones are"} hidden.`;
  }
  return season === "all" ? "No rain events on record yet." : `No rain events${scope} yet.`;
}

/**
 * The list tile's heading: every event of the pick, in its order.
 *
 * @param sort - The order.
 * @returns E.g. "Every event, newest first".
 */
export function listHeading(sort: EventSort): string {
  return `Every event, ${sort} first`;
}

/**
 * The line under the filter: how many events the pick lists, and which
 * sizes it hides when it hides any.
 *
 * @param shown - Events listed.
 * @param hidden - Events of the water year the size filter hides.
 * @param size - The size picked.
 * @returns E.g. "8 events, sprinkles hidden".
 */
export function listCount(shown: number, hidden: number, size: EventSize): string {
  const count = `${shown} event${shown === 1 ? "" : "s"}`;
  const hides = SIZES.find((s) => s.size === size)?.hides ?? null;
  return hidden > 0 && hides ? `${count}, ${hides} hidden` : count;
}

/** The missing-days note: one line, hatched when the year is not compared. */
export interface MissingNote {
  /** E.g. "4 days missing", or "not archived yet". */
  text: string;
  /** The record does not cover the water year (or one of those shown). */
  hatched: boolean;
}

/**
 * The one-line note for the days the record misses in the water year
 * shown, or in all of them (#647: "4 days missing"). The sentence saying
 * what that does to the list is `seasonNote`, in the fold.
 *
 * @param section - The snapshot's `rain_events`.
 * @param season - A water year, or "all".
 * @returns The note, or null when every year shown is whole.
 */
export function missingNote(section: RainEvents, season: Season): MissingNote | null {
  const shown = section.seasons.filter(
    (s) => (season === "all" || s.water_year === season) && s.status.status !== "valid",
  );
  // Days inside the record that it misses: counted, and hatched when they
  // stop a year being compared.
  const inRecord = shown.filter((s) => !beyondRecord(s.status) && s.status.missing_days > 0);
  const days = inRecord.reduce((n, s) => n + s.status.missing_days, 0);
  if (days > 0) {
    const hatched = inRecord.some((s) => !isComparable(s.status));
    return { text: `${days} day${days === 1 ? "" : "s"} missing`, hatched };
  }
  // A year the record does not reach is named, as its option is, not
  // hatched (#470); the fold says why.
  const beyond = shown.filter((s) => beyondRecord(s.status));
  if (!beyond.length) return null;
  const archived = beyond.every((s) => s.status.reason === "not_archived");
  return { text: archived ? "not archived yet" : "partial year", hatched: false };
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
 * @returns E.g. "3 of 365 days of water year 2024–25".
 */
function missingDays(season: EventSeason): string {
  return `${season.status.missing_days} of ${season.status.days} days of water year ${seasonLabel(season.water_year)}`;
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
    const lost = shown.filter((s) => !isComparable(s.status)).map((s) => seasonLabel(s.water_year));
    const prorated = shown.filter((s) => isComparable(s.status)).map(missingDays);
    if (!lost.length && !prorated.length) return null;
    const many = lost.length > 1;
    const notCovered = `The record does not cover all of water year${many ? "s" : ""} ${listText(lost)}`;
    if (!prorated.length)
      return `${notCovered}, so the list can miss ${many ? "their" : "its"} events.`;
    const few = listText(prorated);
    if (!lost.length)
      return `${few} are missing from the record, so the list can miss events in them.`;
    return `${notCovered}, and ${few} are missing from it, so the list can miss events in them.`;
  }
  const found = section.seasons.find((s) => s.water_year === season);
  if (!found) return null;
  const { status } = found;
  if (status.status === "valid") return null;
  if (isComparable(status)) {
    return `${missingDays(found)} are missing from the record, so the list can miss events in them.`;
  }
  return `Water year ${seasonLabel(season)} ${notCompared(status.reason)}, so the list can miss its events and their total is not the year's rain.`;
}

/** A duration given in minutes, in the One Format Rule's words. */
export function minutes(value: number): string {
  return formatDuration(value * 60_000);
}

/** An event's peak rate, or "n/a" without one or without archive readings. */
export function peakText(e: RainEventRow): string {
  return e.peak_rate_in_hr === null || !e.has_curve ? "n/a" : inchesPerHour(e.peak_rate_in_hr);
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
      ? ` in ${seasonLabel(season)}`
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
