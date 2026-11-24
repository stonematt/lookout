/**
 * The Rain tab's words: number and date formats, and each section's finding
 * headline. Pure functions of the snapshot, no DOM, so each headline is
 * tested against the golden snapshots and against worked examples.
 *
 * DESIGN.md rules held here: the Finding First Rule (a headline states what
 * the data shows, with the number in it), the Name the Years Rule (years
 * compared are named, newest first), the One Gap Status Rule (an invalid
 * window's value is never shown, a prorated one carries ≈), and the One
 * Format Rule (station time, imperial units).
 */
import { dayText, formatDayCount, formatDaysSpoken, hourText, inches } from "../format.js";
import { gapValue, notCompared } from "../gaps.js";
import { isShown } from "../overview/history.js";
import type {
  DashboardSnapshot,
  DrySpellNow,
  EarlierWetYear,
  GapStatus,
  RainTab,
  RainWindow,
  Verdict,
  WaterYearCurve,
  WetHoursToDate,
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

/** A dry spell with neither number known, for a snapshot without `dry_spell`. */
const NO_DRY_SPELL: DrySpellNow = { days_since_rain: null, usual_days: null };

/**
 * The dry spell now: the days since it last rained and the usual dry spell,
 * from the dashboard's `dry_spell` (#491), so the Overview, the Rain tab and
 * the widget show one number for each. A snapshot without it reads as both
 * unknown.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The two numbers, each null when unknown.
 */
export function drySpellNow(snapshot: DashboardSnapshot): DrySpellNow {
  return snapshot.dry_spell ?? NO_DRY_SPELL;
}

/**
 * The dry-spell section's finding: the days since it last rained, against
 * the usual dry spell.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "1 d 22 h since it last rained, against a usual dry spell
 *   of 2 days".
 */
export function dryHeadline(snapshot: DashboardSnapshot): string {
  const spell = drySpellNow(snapshot);
  if (spell.days_since_rain === null) return "Dry spells in the record";
  const usual =
    spell.usual_days === null
      ? ""
      : `, against a usual dry spell of ${formatDayCount(spell.usual_days)}`;
  return `${formatDaysSpoken(spell.days_since_rain)} since it last rained${usual}`;
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
 * The year-over-year finding: the water year's verdict when the snapshot
 * has one (`verdictHeadline`, #640), else the years compared, newest first.
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
  const verdict = verdictHeadline(tab);
  if (verdict) return verdict;
  const earlier = earlierToDate(tab);
  const lead = `${name} has ${value} of rain since Oct 1`;
  if (!earlier.length) return lead;
  const named = earlier.slice(0, NAMED_YEARS).map((item) => `${item.text} (${item.year})`);
  return `${lead}, against ${listText(named)} by this date`;
}

/** Each verdict word as the headline's phrase against the usual. */
const VERDICT_WORDS: Record<Verdict, string> = {
  wetter: "wetter than usual",
  drier: "drier than usual",
  "about the same": "about the same as usual",
};

/**
 * The water year's verdict as the Rain tab's headline (#640): the
 * snapshot's word against the usual rain to date, with the numbers it rests
 * on. The site decides nothing: the word is `usual_to_date.verdict`.
 *
 * @param tab - The Rain tab's data.
 * @returns E.g. "Water year 2026 is wetter than usual: 0.44 in since Oct 1,
 *   against a usual 0.30 in by this date (2025)", the years named newest
 *   first up to `NAMED_YEARS`, else counted; null with no verdict (a young
 *   station, or a snapshot written before it), when the headline falls
 *   back to naming the years.
 */
export function verdictHeadline(tab: RainTab): string | null {
  const usual = tab.usual_to_date;
  const current = currentYear(tab);
  const value = current ? toDate(tab, current) : null;
  if (!usual?.verdict || !current || value === null) return null;
  const named = usual.water_years ?? [];
  const years =
    named.length === usual.years && named.length <= NAMED_YEARS
      ? listText(named.map(String).reverse())
      : `${usual.years} earlier year${usual.years === 1 ? "" : "s"}`;
  return (
    `Water year ${current.water_year} is ${VERDICT_WORDS[usual.verdict]}: ` +
    `${value} since Oct 1, against a usual ${inches(usual.usual_in)} by this date (${years})`
  );
}

/**
 * The earlier water years compared by this date, newest first, each with
 * its value to date as shown: the years the year-over-year finding names.
 *
 * @param tab - The Rain tab's data.
 * @returns E.g. [{ year: 2025, text: "3.10 in" }, { year: 2024, text: "≈ 5.02 in" }].
 */
export function earlierToDate(tab: RainTab): { year: number; text: string }[] {
  return tab.water_years
    .slice(0, -1)
    .reverse()
    .map((curve) => ({ year: curve.water_year, text: toDate(tab, curve) }))
    .filter((item): item is { year: number; text: string } => item.text !== null);
}

/** A whole hour count, grouped: "1,292". */
function hourCount(hours: number): string {
  return hours.toLocaleString("en-US");
}

/**
 * This water year's wet hours since Oct 1, as wet hours of countable hours,
 * as the Overview's meter reads its windows (#663).
 *
 * @param wet - The Rain tab's wet hours to date.
 * @returns E.g. "113 h of 1,292".
 */
export function wetToDateValue(wet: WetHoursToDate): string {
  return `${hourCount(wet.wet_hours)} h of ${hourCount(wet.countable_hours)}`;
}

/**
 * Usual wet hours to date, with the earlier years it rests on (#663): a
 * band from 3 years, called a range below 4; the plain values below 3;
 * none without an earlier year. Then the hours missing, never dry.
 *
 * @param wet - The Rain tab's wet hours to date.
 * @returns E.g. "usual to date 147–186 h, range of 3 earlier years",
 *   "usual to date 160 h, from 2025", "usual to date 150 h (2024), 170 h
 *   (2025)" or "no earlier year to compare yet", each with " · 27 h
 *   missing, not dry" when hours are missing.
 */
export function usualWetWords(wet: WetHoursToDate): string {
  const usual = wet.usual;
  let words = "no earlier year to compare yet";
  if (usual && usual.band && usual.low_hours !== null && usual.high_hours !== null) {
    const of =
      usual.band === "percentile"
        ? `25th–75th percentile of ${usual.years} earlier years`
        : `range of ${usual.years} earlier years`;
    words = `usual to date ${hourCount(usual.low_hours)}–${hourCount(usual.high_hours)} h, ${of}`;
  } else if (usual && usual.earlier.length === 1) {
    const [{ usual_hours, water_year }] = usual.earlier as [EarlierWetYear];
    words = `usual to date ${hourCount(usual_hours)} h, from ${water_year}`;
  } else if (usual) {
    const values = usual.earlier.map((y) => `${hourCount(y.usual_hours)} h (${y.water_year})`);
    words = `usual to date ${values.join(", ")}`;
  }
  return wet.missing_hours > 0
    ? `${words} · ${hourCount(wet.missing_hours)} h missing, not dry`
    : words;
}

/**
 * The heatmap's finding: how many hours it rained, and the wettest hour.
 * The hour count is a window value: ≈ when the window is prorated. Once
 * the snapshot carries wet hours to date (#663), the hour count is the
 * water year's, beside the water-year curves, and the heatmap names only
 * the wettest hour: the prorated count is not read.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Rain fell in 18 hours of the last 90 days, the heaviest
 *   0.25 in at 10 am on Jan 9", or with wet hours to date "Rain by the
 *   hour over the last 90 days, the heaviest 0.25 in at 10 am on Jan 9".
 */
export function hoursHeadline(snapshot: DashboardSnapshot): string {
  const { recent } = snapshot.rain_tab;
  const span = `the last ${recent.rain_in.length} days`;
  if (snapshot.rain_tab.wet_hours_to_date) {
    const heaviest = recent.wettest_hour;
    if (heaviest) {
      return (
        `Rain by the hour over ${span}, the heaviest ` +
        `${inches(heaviest.rain_in)} at ${hourText(heaviest.hour)} on ${dayText(heaviest.day)}`
      );
    }
    return rainStatus(snapshot, recent.window).status === "invalid"
      ? `Rain by the hour over ${span}`
      : `No rain fell in ${span}`;
  }
  const { wet_hours: wet, wettest_hour: wettest } = recent;
  if (wet === null) return `Rain by the hour over ${span}`;
  if (wet === 0 || !wettest) return `No rain fell in ${span}`;
  const count = Math.round(wet);
  const hours = gapValue(
    rainStatus(snapshot, recent.window),
    `${count} ${count === 1 ? "hour" : "hours"}`,
  );
  return (
    `Rain fell in ${hours} of ${span}, the heaviest ` +
    `${inches(wettest.rain_in)} at ${hourText(wettest.hour)} on ${dayText(wettest.day)}`
  );
}

/**
 * Why some windows in the strip are hatched, in one sentence for the source
 * line. A window the record does not cover yet is left out of the strip
 * (#470), so it is left out here too.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Not compared: Today and yesterday are not archived yet."
 *   or null when every window shown can be compared.
 */
export function windowsNote(snapshot: DashboardSnapshot): string | null {
  const groups = new Map<string, string[]>();
  for (const window of snapshot.rain_tab.windows) {
    if (window.status.status !== "invalid" || !isShown(window.status)) continue;
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
