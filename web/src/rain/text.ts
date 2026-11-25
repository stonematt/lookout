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
import {
  WATER_YEAR_START_MONTH,
  dayText,
  formatDayCount,
  formatDays,
  formatDaysSpoken,
  hourText,
  inches,
} from "../format.js";
import { gapValue, notCompared } from "../gaps.js";
import { isShown } from "../overview/history.js";
import { agoWords, lastRainMs, nowStory, storyLine } from "../now-story.js";
import { WHOLE_DRY_DAYS_VERSION } from "../snapshot.js";
import { meterShown } from "../wet-bars.js";
import type {
  DashboardSnapshot,
  DrySpellNow,
  EarlierWetYear,
  GapStatus,
  RainTab,
  RainWindow,
  StripWindow,
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
 *   of 2 days"; "No dry spell yet, last rain 9 h ago" before a whole dry
 *   day has passed.
 */
export function dryHeadline(snapshot: DashboardSnapshot): string {
  const spell = drySpellNow(snapshot);
  const shown = drySpellShown(snapshot);
  if (shown === null) return "Dry spells in the record";
  if (shown.under !== null) return `${shown.spoken}, ${shown.under}`;
  const usual =
    spell.usual_days === null
      ? ""
      : `, against a usual dry spell of ${formatDayCount(spell.usual_days)}`;
  return `${shown.spoken} since it last rained${usual}`;
}

/** The dry spell now as it is shown: the number, its unit and its words. */
export interface DrySpellShown {
  /** "9" in whole dry days; "1 d 14 h" before them, the unit in the words. */
  value: string;
  /** "days" or "day" in whole dry days; null when the number carries it. */
  unit: string | null;
  /** The number for a sentence: "9 dry days"; "about 39 hours" before. */
  spoken: string;
  /**
   * Words under the reading when there is no dry spell yet (version 5, 0
   * whole dry days): "last rain 9 h ago"; null otherwise, or with no last
   * rain to name.
   */
  under: string | null;
}

/** The reading at 0 whole dry days: no whole dry day has passed since the last rain. */
const NO_DRY_SPELL_YET = "No dry spell yet";

/**
 * How the days since it last rained read, by the snapshot's version: whole
 * dry days from `WHOLE_DRY_DAYS_VERSION` (GLOSSARY dry spell, #667), so a
 * pause between showers never reads as a dry spell; before it, the time
 * since the last tip as a duration. At 0 whole dry days there is no dry
 * spell yet: "No dry spell yet", with last rain under it (#694, product
 * lead on #647), never "0 dry days". The Dry spell tile, the feature, the
 * Rain tab and the dry spell zoom all show this one reading.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The reading; null when the days since it last rained are unknown.
 */
export function drySpellShown(snapshot: DashboardSnapshot): DrySpellShown | null {
  const days = drySpellNow(snapshot).days_since_rain;
  if (days === null) return null;
  if (snapshot.schema_version >= WHOLE_DRY_DAYS_VERSION) {
    const whole = Math.round(days);
    if (whole === 0) {
      const last = lastRainMs(snapshot);
      const under = last === null ? null : `last rain ${agoWords(snapshot.built_at_ms - last)} ago`;
      return { value: NO_DRY_SPELL_YET, unit: null, spoken: NO_DRY_SPELL_YET, under };
    }
    const unit = whole === 1 ? "day" : "days";
    return { value: String(whole), unit, spoken: `${whole} dry ${unit}`, under: null };
  }
  return { value: formatDays(days), unit: null, spoken: formatDaysSpoken(days), under: null };
}

/**
 * A dry spell reading as one string, for a stat: "9 days", "1 d 14 h".
 *
 * @param shown - The reading.
 * @returns The number and its unit.
 */
export function drySpellText(shown: DrySpellShown): string {
  return shown.unit === null ? shown.value : `${shown.value} ${shown.unit}`;
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
  // No total to date yet: the year's name alone, no words for it (#720).
  if (value === null) return name;
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
  // No earlier year yet: no usual words at all (#720).
  let words = "";
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
  const missing = wet.missing_hours > 0 ? `${hourCount(wet.missing_hours)} h missing, not dry` : "";
  return [words, missing].filter(Boolean).join(" · ");
}

/**
 * The heatmap's finding: the wettest hour. It counts no hours: wet hours
 * come from the one wet-hour helper (`wet_hours`, `wet_hours_to_date`) in
 * the Rain tab's head, and the Rain tab's prorated hour count is never read
 * (#667). A snapshot without wet hours to date gets the same headline, with
 * no hour count anywhere: the missing field reads as null (ADR-0008).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Rain by the hour over the last 90 days, the heaviest
 *   0.25 in at 10 am on Jan 9".
 */
export function hoursHeadline(snapshot: DashboardSnapshot): string {
  const { recent } = snapshot.rain_tab;
  const span = `the last ${recent.rain_in.length} days`;
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

/**
 * The Rain tab head's now part (ADR-0011, #664): the now story in one line,
 * worded by the one now-story helper (`now-story.ts`) the Overview's rain
 * tile uses too, so the two never disagree (#694).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Raining · 0.12 in/hr · 0.30 in since 2:40 pm", "Last rain
 *   9 h ago"; null for a snapshot without `rain_now` (before #659), which
 *   draws no now part.
 */
export function nowLine(snapshot: DashboardSnapshot): string | null {
  const story = nowStory(snapshot);
  return story ? storyLine(story) : null;
}

/** "1 wet hour", "1,292 wet hours". */
const wetHourCount = (n: number) => `${hourCount(n)} wet hour${n === 1 ? "" : "s"}`;

/**
 * The Rain tab head's season part in words (#664): the wet hours of the
 * last 7 and 30 days (the Overview's `wet_hours`) while their bars are
 * drawn (`meterShown`: not in dry mode, #647 call 3), else the water
 * year's since Oct 1.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "19 wet hours in the last 7 days, 112 in the last 30", "113
 *   wet hours since Oct 1"; null when the snapshot has neither (written
 *   before #657 and #663), which draws no season part.
 */
export function seasonLead(snapshot: DashboardSnapshot): string | null {
  const wet = snapshot.wet_hours;
  if (wet && meterShown(snapshot)) {
    return (
      `${wetHourCount(wet.last_7_days.wet_hours)} in the last 7 days, ` +
      `${hourCount(wet.last_30_days.wet_hours)} in the last 30`
    );
  }
  const toDate = snapshot.rain_tab.wet_hours_to_date;
  return toDate ? `${wetHourCount(toDate.wet_hours)} since Oct 1` : null;
}

/**
 * A window strip row's usual in words (#664), for its label under the track
 * and its accessible name: the span the snapshot sends, called what its
 * band says it is.
 *
 * @param window - The strip window.
 * @returns E.g. "usual 0.30–0.50 in, range of 3 earlier years", "usual 0.30
 *   in, from 2025", "usual 0.00–0.12 in, 2 earlier years", "no earlier
 *   year". One earlier year is named: the water year for the water-year
 *   window, else the year its window ended in.
 */
export function stripUsualWords(window: StripWindow): string {
  const { low_in: low, high_in: high, years } = window;
  if (years === 0 || low === null || high === null) return "no earlier year";
  const span = low === high ? inches(low) : `${inches(low, { unit: false })}–${inches(high)}`;
  const of =
    window.band === "percentile"
      ? `25th–75th percentile of ${years} earlier years`
      : window.band === "range"
        ? `range of ${years} earlier years`
        : years === 1 && window.earlier[0]
          ? `from ${yearOf(window.key, window.earlier[0].last_day)}`
          : `${years} earlier years`;
  return `usual ${span}, ${of}`;
}

/** The year an earlier window is named by: its water year, or the year it ended in. */
function yearOf(key: string, lastDay: string): number {
  const year = Number(lastDay.slice(0, 4));
  const month = Number(lastDay.slice(5, 7));
  return key === "water_year" && month >= WATER_YEAR_START_MONTH ? year + 1 : year;
}
