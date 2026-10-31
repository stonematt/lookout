/**
 * The Solar tab's view logic, free of the DOM: the selected day in the URL,
 * reading one day out of the snapshot, and the words.
 *
 * Every number comes from the snapshot's `solar_tab` (built by
 * `lookout/snapshot/solar.py`): the daily, monthly and window totals are
 * computed once in Python and gap-judged there. The browser only picks a
 * day, finds the brightest one, and words them.
 */
import { DAY_MS, MONTHS, addDays, dayText, kwhPerM2, wattsPerM2 } from "../format.js";
import { gapValue } from "../gaps.js";
import type { DashboardSnapshot, GapReason, SolarTab, SolarTrace, SolarWindow } from "../snapshot.js";

/** The URL query parameter holding the selected solar day. */
export const SOLAR_DAY_PARAM = "solar_day";

/** The slots a day's 15-minute chart and the day heatmap show: 4 am to 10 pm. */
export const FIRST_SLOT = 16;
export const LAST_SLOT = 87;

/**
 * The selected day from a page query, or null.
 *
 * @param search - `location.search`.
 * @returns "YYYY-MM-DD", or null when absent or malformed.
 */
export function selectedDay(search: string): string | null {
  const value = new URLSearchParams(search).get(SOLAR_DAY_PARAM);
  return value !== null && /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : null;
}

/**
 * A page query with the selected day set (or cleared), every other
 * parameter kept.
 *
 * @param search - `location.search`.
 * @param day - "YYYY-MM-DD", or null to clear it.
 * @returns The new query, with its `?`, or "" when empty.
 */
export function withSolarDay(search: string, day: string | null): string {
  const query = new URLSearchParams(search);
  if (day === null) query.delete(SOLAR_DAY_PARAM);
  else query.set(SOLAR_DAY_PARAM, day);
  const text = query.toString();
  return text ? `?${text}` : "";
}

/** Today, the last of the tab's days. */
export function lastDay(tab: SolarTab): string {
  return addDays(tab.days.first_day, tab.days.energy_kwh_m2.length - 1);
}

/**
 * The day the tab opens on: the URL's when the record holds it, else today.
 *
 * @param tab - The Solar tab's data.
 * @param search - `location.search`.
 * @returns "YYYY-MM-DD".
 */
export function openingDay(tab: SolarTab, search: string): string {
  const day = selectedDay(search);
  const today = lastDay(tab);
  return day !== null && day >= tab.days.first_day && day <= today ? day : today;
}

/** Whole days from `first` to `day`. */
export function daysBetween(first: string, day: string): number {
  const at = (iso: string) => Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10));
  return Math.round((at(day) - at(first)) / DAY_MS);
}

/** One day as the tab reads it. */
export interface SolarDay {
  day: string;
  /** kWh per m², null when it cannot be counted. */
  energy: number | null;
  reason: GapReason | null;
  /** Its 96 slots in Wh per m², or null when it is older than the 90 days. */
  slots: (number | null)[] | null;
  isToday: boolean;
}

/**
 * Read one day out of the snapshot.
 *
 * @param tab - The Solar tab's data.
 * @param day - "YYYY-MM-DD", within the record.
 * @returns The day.
 */
export function readDay(tab: SolarTab, day: string): SolarDay {
  const i = daysBetween(tab.days.first_day, day);
  const r = daysBetween(tab.recent.first_day, day);
  return {
    day,
    energy: tab.days.energy_kwh_m2[i] ?? null,
    reason: tab.days.reason[i] ?? null,
    slots: (r >= 0 ? tab.recent.energy_wh_m2[r] : undefined) ?? null,
    isToday: day === lastDay(tab),
  };
}

/** 15 minutes' energy in Wh per m², e.g. "37 Wh/m²". */
export function wh(value: number): string {
  return `${Math.round(value)} Wh/m²`;
}


/**
 * A slot's clock time in the One Format Rule, e.g. "10:15 am", "noon".
 *
 * @param slot - 0 to 95, from midnight.
 * @returns The time.
 */
export function slotText(slot: number): string {
  const hour = Math.floor(slot / 4);
  const minute = (slot % 4) * 15;
  if (minute === 0 && hour === 12) return "noon";
  if (minute === 0 && hour === 0) return "midnight";
  const h = hour % 12 === 0 ? 12 : hour % 12;
  return `${h}:${String(minute).padStart(2, "0")} ${hour < 12 ? "am" : "pm"}`;
}

/** A month key "YYYY-MM" in words, e.g. "Jun 2025". */
export function monthText(key: string): string {
  return `${MONTHS[Number(key.slice(5, 7)) - 1]} ${key.slice(0, 4)}`;
}

/**
 * A window's energy as its own gap status allows it to be shown: solar's
 * status, which also counts the days the energy catalog lacks.
 *
 * @param window - The Solar tab's window total.
 * @returns "2.41 kWh/m²", "≈ 61.2 kWh/m²", or null when not compared.
 */
export function windowValue(window: SolarWindow): string | null {
  if (window.energy_kwh_m2 === null) return null;
  return gapValue(window.status, kwhPerM2(window.energy_kwh_m2));
}

/** A window's energy per day, from its total, or null. */
function perDay(snapshot: DashboardSnapshot, key: string): number | null {
  const window = snapshot.solar_tab?.windows.find((w) => w.key === key);
  if (!window || window.energy_kwh_m2 === null) return null;
  return window.energy_kwh_m2 / window.status.days;
}

/** "up from", "down from" or "about the same as": a change in words. */
function change(now: number, before: number): string {
  if (Math.abs(now - before) <= Math.max(0.05, 0.05 * before)) return "about the same as";
  return now > before ? "up from" : "down from";
}

/**
 * The 24-hour view's finding.
 *
 * @param trace - The 24-hour view.
 * @param liveWm2 - The live solar reading.
 * @returns E.g. "2.41 kWh/m² of sun in the last 24 hours, up from 1.92 the 24 before".
 */
export function traceHeadline(trace: SolarTrace, liveWm2: number | null): string {
  const now = trace.energy_kwh_m2;
  if (now === null) {
    return liveWm2 === null ? "The last 24 hours cannot be totalled yet" : `Solar radiation is ${wattsPerM2(liveWm2)} now`;
  }
  const before = trace.previous_kwh_m2;
  const head = `${kwhPerM2(now)} of sun in the last 24 hours`;
  return before === null ? head : `${head}, ${change(now, before)} ${kwhPerM2(before, { unit: false })} the 24 before`;
}

/** How many earlier years the daily headline names, newest first. */
export const NAMED_YEARS = 3;

/**
 * The last 7 full days (to yesterday), each day's average, and the same
 * dates in each earlier year whose 7 days are all in the record.
 *
 * @param tab - The Solar tab's data.
 * @returns This year's average and the earlier years', newest first; null
 *   when the last 7 full days are not all counted.
 */
export function sameWeek(tab: SolarTab): { now: number; years: { year: number; perDay: number }[] } | null {
  const end = addDays(lastDay(tab), -1);
  const dates = Array.from({ length: 7 }, (_, i) => addDays(end, i - 6));
  const average = (days: string[]): number | null => {
    let total = 0;
    for (const day of days) {
      const i = daysBetween(tab.days.first_day, day);
      const v = i >= 0 ? tab.days.energy_kwh_m2[i] : null;
      if (v === null || v === undefined) return null;
      total += v;
    }
    return total / days.length;
  };
  const now = average(dates);
  if (now === null) return null;
  const years: { year: number; perDay: number }[] = [];
  const thisYear = Number(end.slice(0, 4));
  for (let year = thisYear - 1; year >= Number(tab.days.first_day.slice(0, 4)); year--) {
    const shifted = dates.map((d) => `${Number(d.slice(0, 4)) - (thisYear - year)}${d.slice(4)}`);
    if (shifted.some((d) => addDays(d, 0) !== d)) continue; // Feb 29 in a common year
    const perDay = average(shifted);
    if (perDay !== null) years.push({ year, perDay });
  }
  return { now, years };
}

/**
 * The daily section's finding: the last 7 full days against the same dates
 * in earlier years (the sun follows the season, so a week is compared with
 * its own dates); against the last 30 days only when no earlier year can be.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "Brighter than this week in earlier years: 3.10 kWh/m² a day,
 *   against 2.40 (2025) and 2.65 (2024)".
 */
export function dailyHeadline(snapshot: DashboardSnapshot): string {
  const tab = snapshot.solar_tab;
  const week = tab ? sameWeek(tab) : null;
  if (week && week.years.length) {
    const named = week.years.slice(0, NAMED_YEARS);
    const usual = named.reduce((sum, y) => sum + y.perDay, 0) / named.length;
    const word = change(week.now, usual);
    const lead =
      word === "about the same as"
        ? "A usual week for sun"
        : week.now > usual
          ? "Brighter than this week in earlier years"
          : "Duller than this week in earlier years";
    const against = named.map((y) => `${kwhPerM2(y.perDay, { unit: false })} (${y.year})`);
    const list = against.length > 1 ? `${against.slice(0, -1).join(", ")} and ${against[against.length - 1]}` : against[0];
    return `${lead}: ${kwhPerM2(week.now)} a day over the last 7 full days, against ${list}`;
  }
  const month = perDay(snapshot, "30d");
  if (week && month !== null) {
    return `${kwhPerM2(week.now)} a day over the last 7 full days, against ${kwhPerM2(month, { unit: false })} over 30 days`;
  }
  if (month !== null) return `${kwhPerM2(month)} of sun a day over the last 30 days`;
  return "Sun each day over the last 90 days";
}

/** The brightest 15 minutes of a day's slots: [slot, Wh], or null. */
export function peakSlot(slots: (number | null)[]): [number, number] | null {
  let best: [number, number] | null = null;
  slots.forEach((v, i) => {
    if (v !== null && v > 0 && (best === null || v > best[1])) best = [i, v];
  });
  return best;
}

/**
 * The selected day's finding.
 *
 * @param day - The day as read.
 * @returns E.g. "Jun 19, 2025: 4.12 kWh/m², brightest at 1:15 pm".
 */
export function dayHeadline(day: SolarDay): string {
  const when = `${dayText(day.day, true)}${day.isToday ? " so far" : ""}`;
  if (day.energy === null) return `${when}: not in the record`;
  const peak = day.slots ? peakSlot(day.slots) : null;
  const total = `${when}: ${kwhPerM2(day.energy)}`;
  return peak ? `${total}, brightest at ${slotText(peak[0])}` : total;
}

/**
 * The brightest valid month and day in the record, for the heatmaps'
 * findings; null when there is none.
 */
export function brightestMonth(tab: SolarTab): { key: string; energy: number } | null {
  let best: { key: string; energy: number } | null = null;
  for (const m of tab.months) {
    if (m.energy_kwh_m2 !== null && (best === null || m.energy_kwh_m2 > best.energy)) {
      best = { key: m.key, energy: m.energy_kwh_m2 };
    }
  }
  return best;
}

/** The brightest counted day in the record, or null. */
export function brightestDay(tab: SolarTab): { day: string; energy: number } | null {
  let best: { day: string; energy: number } | null = null;
  tab.days.energy_kwh_m2.forEach((v, i) => {
    if (v !== null && (best === null || v > best.energy)) best = { day: addDays(tab.days.first_day, i), energy: v };
  });
  return best;
}
