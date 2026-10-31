/**
 * The Rain tab's words: number and date formats, and each section's finding
 * headline. Pure functions of the snapshot, no DOM, so each headline is
 * tested against the golden snapshots and against worked examples.
 *
 * DESIGN.md rules held here: the Finding First Rule (a headline states what
 * the data shows, with the number in it), the Name the Years Rule (years
 * compared are named, newest first), the One Gap Status Rule (an invalid
 * window's value is never shown, a prorated one carries ≈), and the One
 * Format Rule (Pacific time, imperial units).
 */
import { gapValue } from "../gaps.js";
import type {
  DashboardSnapshot,
  GapStatus,
  RainTab,
  RainWindow,
  WaterYearCurve,
} from "../snapshot.js";

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** At most this many earlier years are named in the year-over-year headline. */
export const NAMED_YEARS = 3;

/** The windows' names as the strip and the sentences use them. */
export const WINDOW_NAMES: Record<string, string> = {
  today: "Today",
  yesterday: "Yesterday",
  "7d": "7 days",
  "30d": "30 days",
  "90d": "90 days",
  "365d": "365 days",
  water_year: "Water year",
};

/**
 * Inches as the tab writes them: two decimals, one from 10 in up.
 *
 * @param value - Inches.
 * @returns E.g. "0.30 in", "12.4 in".
 */
export function inches(value: number): string {
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} in`;
}

/**
 * A station-local day, e.g. "Jan 8", or "Jan 8, 2026" with the year.
 *
 * @param iso - "YYYY-MM-DD".
 * @param withYear - Add the year.
 * @returns The day in words.
 */
export function dayText(iso: string, withYear = false): string {
  const text = `${MONTHS[Number(iso.slice(5, 7)) - 1]} ${Number(iso.slice(8, 10))}`;
  return withYear ? `${text}, ${iso.slice(0, 4)}` : text;
}

/**
 * The day `offset` days after an ISO day.
 *
 * @param iso - "YYYY-MM-DD".
 * @param offset - Days to add.
 * @returns "YYYY-MM-DD".
 */
export function addDays(iso: string, offset: number): string {
  const ms = Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10));
  return new Date(ms + offset * 86_400_000).toISOString().slice(0, 10);
}

/**
 * An hour of the day, e.g. "midnight", "3 am", "noon", "9 pm".
 *
 * @param hour - 0 to 23.
 * @returns The hour in words.
 */
export function hourText(hour: number): string {
  if (hour === 0) return "midnight";
  if (hour === 12) return "noon";
  return hour < 12 ? `${hour} am` : `${hour - 12} pm`;
}

/**
 * A spell's length in the One Format Rule's duration style, e.g. "4 d",
 * "2.5 d" (a median can fall between two whole days).
 *
 * @param days - A count of days.
 * @returns The duration.
 */
export function daysText(days: number): string {
  return `${Math.round(days * 10) / 10} d`;
}

/**
 * "a", "a and b", "a, b and c".
 *
 * @param items - Words.
 * @returns The list in words.
 */
export function listText(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/**
 * A window's gap status by key.
 *
 * @param snapshot - The dashboard snapshot.
 * @param key - "today", "7d", "water_year", ...
 * @returns Its status.
 */
export function windowStatus(snapshot: DashboardSnapshot, key: string): GapStatus {
  const status = snapshot.gaps.windows.find((w) => w.key === key);
  if (!status) throw new Error(`no gap status for window ${key}`);
  return status;
}

/**
 * A window's rain as its gap status allows it to be shown.
 *
 * @param snapshot - The dashboard snapshot.
 * @param window - The Rain tab's window total.
 * @returns "0.30 in", "≈ 3.10 in", or null when it cannot be compared.
 */
export function windowValue(snapshot: DashboardSnapshot, window: RainWindow): string | null {
  if (window.total_in === null) return null;
  return gapValue(windowStatus(snapshot, window.key), inches(window.total_in));
}

/** A window's total by key, as shown, or null. */
function shown(snapshot: DashboardSnapshot, key: string): string | null {
  const window = snapshot.rain_tab.windows.find((w) => w.key === key);
  return window ? windowValue(snapshot, window) : null;
}

/**
 * The daily section's finding: today's rain, then the last 7 days'.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "0.30 in of rain so far today, and 0.85 in over the last 7
 *   days".
 */
export function dailyHeadline(snapshot: DashboardSnapshot): string {
  const today = snapshot.rain_tab.windows.find((w) => w.key === "today");
  const week = shown(snapshot, "7d");
  const weekWords = week === null ? null : `${week} over the last 7 days`;
  const todayText = today ? shown(snapshot, "today") : null;
  if (todayText !== null && today && today.total_in !== null) {
    const lead = today.total_in > 0 ? `${todayText} of rain so far today` : "No rain so far today";
    return weekWords ? `${lead}, and ${weekWords}` : lead;
  }
  if (weekWords) return `${week} of rain over the last 7 days`;
  return "Rain each day over the last 90 days";
}

/**
 * The dry-spell section's finding: days since the last rain against the
 * usual dry spell for this date.
 *
 * @param tab - The Rain tab's data.
 * @returns E.g. "4 days since the last rain, against a usual dry spell of
 *   2 days at this time of year".
 */
export function dryHeadline(tab: RainTab): string {
  const { current_days: days, usual_days: usual } = tab.dry_spells;
  if (days === null) return "Dry spells in the record";
  const usualWords = usual === null ? null : `a usual dry spell of ${daysText(usual)} at this time of year`;
  if (days === 0) return "It has rained today, so no dry spell is running";
  const since = `${daysText(days)} since the last rain`;
  return usualWords ? `${since}, against ${usualWords}` : since;
}

/** The current water year's curve: the newest one. */
export function currentYear(tab: RainTab): WaterYearCurve | null {
  return tab.water_years[tab.water_years.length - 1] ?? null;
}

/**
 * A water-year curve's value at one place, as shown. Each place is a window
 * from Oct 1 with its own state: ≈ when prorated, null when invalid.
 *
 * @param curve - The water year.
 * @param place - Days after Oct 1.
 * @returns E.g. "≈ 3.10 in", or null.
 */
export function curveValue(curve: WaterYearCurve, place: number): string | null {
  const value = curve.cumulative_in[place];
  if (value === null || value === undefined) return null;
  return `${curve.prorated[place] ? "≈ " : ""}${inches(value)}`;
}

/** A curve's value at today's place, as shown, or null. */
function toDate(tab: RainTab, curve: WaterYearCurve): string | null {
  return curve.to_date_in === null ? null : curveValue(curve, tab.day_index);
}

/**
 * Earlier water years that have a curve but cannot be compared at today's
 * place, so the source line can say the headline leaves them out.
 *
 * @param tab - The Rain tab's data.
 * @returns E.g. [2025], newest first.
 */
export function yearsSkipped(tab: RainTab): number[] {
  return tab.water_years
    .slice(0, -1)
    .filter((c) => c.to_date_in === null && c.cumulative_in.some((v) => v !== null))
    .map((c) => c.water_year)
    .reverse();
}

/**
 * The year-over-year finding, naming the years compared, newest first.
 *
 * @param tab - The Rain tab's data.
 * @returns E.g. "Water year 2026 has 4.20 in of rain since Oct 1, against
 *   3.10 in (2025) and 5.02 in (2024) by this date".
 */
export function yearsHeadline(tab: RainTab): string {
  const current = currentYear(tab);
  if (!current) return "No water year in the record yet";
  const name = `Water year ${current.water_year}`;
  const value = toDate(tab, current);
  if (value === null) return `${name} cannot be compared yet`;
  const earlier = tab.water_years
    .slice(0, -1)
    .reverse()
    .map((curve) => ({ year: curve.water_year, text: toDate(tab, curve) }))
    .filter((item): item is { year: number; text: string } => item.text !== null);
  const lead = `${name} has ${value} of rain since Oct 1`;
  if (!earlier.length) return lead;
  const named = earlier.slice(0, NAMED_YEARS).map((item) => `${item.text} (${item.year})`);
  return `${lead}, against ${listText(named)} by this date`;
}

/**
 * The heatmap's finding: how many hours it rained, and the wettest hour.
 * The hour count is a window value: ≈ when the window is prorated.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Rain fell in 18 hours of the last 90 days, the heaviest
 *   0.25 in at 10 am on Jan 9".
 */
export function hoursHeadline(snapshot: DashboardSnapshot): string {
  const { recent } = snapshot.rain_tab;
  const { wet_hours: wet, wettest_hour: wettest } = recent;
  const span = `the last ${recent.rain_in.length} days`;
  if (wet === null) return `Rain by the hour over ${span}`;
  if (wet === 0 || !wettest) return `No rain fell in ${span}`;
  const count = Math.round(wet);
  const hours = gapValue(windowStatus(snapshot, recent.window), `${count} ${count === 1 ? "hour" : "hours"}`);
  return (
    `Rain fell in ${hours} of ${span}, the heaviest ` +
    `${inches(wettest.rain_in)} at ${hourText(wettest.hour)} on ${dayText(wettest.day)}`
  );
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

/** The verb phrase for a reason, singular or plural. */
function notCompared(reason: string | null, many = false): string {
  const words = NOT_COMPARED[reason ?? "gap"] ?? NOT_COMPARED.gap!;
  return words[many ? 1 : 0];
}

/**
 * Why some windows are not shown, in one sentence for the source line.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "7 days to the water year start before the archive began."
 *   or null when every window can be compared.
 */
export function windowsNote(snapshot: DashboardSnapshot): string | null {
  const groups = new Map<string, string[]>();
  for (const window of snapshot.rain_tab.windows) {
    const status = windowStatus(snapshot, window.key);
    if (status.status !== "invalid") continue;
    const reason = status.reason ?? "gap";
    groups.set(reason, [...(groups.get(reason) ?? []), WINDOW_NAMES[window.key] ?? window.key]);
  }
  if (!groups.size) return null;
  const parts = [...groups].map(([reason, names]) => {
    const lowered = names.map((n, i) => (i === 0 ? n : n.toLowerCase()));
    return `${listText(lowered)} ${notCompared(reason, names.length > 1)}`;
  });
  return `Not compared: ${parts.join("; ")}.`;
}

/**
 * Why a water year has no curve at all, for the source line.
 *
 * @param curve - A water year whose every value is invalid.
 * @returns E.g. "Water year 2024 is left out: it starts before the archive
 *   began."
 */
export function yearLeftOut(curve: WaterYearCurve): string {
  return `Water year ${curve.water_year} is left out: it ${notCompared(curve.status.reason)}.`;
}
