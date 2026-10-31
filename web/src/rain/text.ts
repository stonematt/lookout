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
import { dayText, formatDayCount, formatDays, hourText, inches } from "../format.js";
import { gapValue } from "../gaps.js";
import type {
  DashboardSnapshot,
  GapStatus,
  LeadRule,
  RainTab,
  RainWindow,
  WaterYearCurve,
} from "../snapshot.js";

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
 * A window's rain gap status by key: the archive's, with any day whose rain
 * counter has no reading missing. Rain values use this, not `windowStatus`.
 *
 * @param snapshot - The dashboard snapshot.
 * @param key - "today", "7d", "water_year", ...
 * @returns Its rain status.
 */
export function rainStatus(snapshot: DashboardSnapshot, key: string): GapStatus {
  const window = snapshot.rain_tab.windows.find((w) => w.key === key);
  if (!window) throw new Error(`no rain window ${key}`);
  return window.status;
}

/**
 * A window's rain as its rain gap status allows it to be shown.
 *
 * @param window - The Rain tab's window total.
 * @returns "0.30 in", "≈ 3.10 in", or null when it cannot be compared.
 */
export function windowValue(window: RainWindow): string | null {
  if (window.total_in === null) return null;
  return gapValue(window.status, inches(window.total_in));
}

/** A window's total by key, as shown, or null. */
function shown(snapshot: DashboardSnapshot, key: string): string | null {
  const window = snapshot.rain_tab.windows.find((w) => w.key === key);
  return window ? windowValue(window) : null;
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
 * The calm lead rule: the days since it last rained and the usual dry
 * spell. The Rain tab reads them from the hero lead so the Overview and the
 * Rain tab show one number for each.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The rule.
 */
export function calmRule(snapshot: DashboardSnapshot): LeadRule | null {
  return snapshot.hero_lead.rules.find((r) => r.kind === "calm") ?? null;
}

/**
 * The dry-spell section's finding: the calm lead's days since it last
 * rained, against its usual dry spell.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "1 d 22 h since it last rained, against a usual dry spell
 *   of 2 days".
 */
export function dryHeadline(snapshot: DashboardSnapshot): string {
  const calm = calmRule(snapshot);
  if (!calm || calm.value === null) return "Dry spells in the record";
  const usual = calm.reference === null ? "" : `, against a usual dry spell of ${formatDayCount(calm.reference)}`;
  return `${formatDays(calm.value)} since it last rained${usual}`;
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
  const hours = gapValue(rainStatus(snapshot, recent.window), `${count} ${count === 1 ? "hour" : "hours"}`);
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
    if (window.status.status !== "invalid") continue;
    const reason = window.status.reason ?? "gap";
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
