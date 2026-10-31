/**
 * The snapshot contract, TypeScript side.
 *
 * The Python refresh run publishes two documents per station on Storj
 * (`lookout/snapshot/model.py` holds the rules; `tests/golden/snapshots/`
 * holds the goldens both sides test against):
 *
 * - `snapshot/<MAC>.dashboard.json`, served by `/dashboard`.
 * - `snapshot/<MAC>.context.json`, served by `/current` next to the live
 *   Ambient `lastData`.
 *
 * Shape rules: every document starts with `schema_version` and `kind`; times
 * are integer epoch milliseconds UTC in fields ending `_ms`; units are named
 * in the field; an unknown value is `null`, never missing.
 *
 * This module is shared by the browser and the read functions, so it uses no
 * DOM and no Node APIs.
 */

/** The schema version this site reads. A breaking change bumps both sides. */
export const SCHEMA_VERSION = 2;

/** The station a snapshot describes. */
export interface Station {
  /** Its MAC address, the key of every file it has on Storj. */
  mac: string;
  /** Its human-readable name in Ambient, or null when unnamed. */
  name: string | null;
}

/** "rising", "falling" or "steady". */
export type Trend = "rising" | "falling" | "steady";

/** The newest reading, live where the live reading reports it. */
export interface SnapshotConditions {
  temperature_f: number | null;
  feels_like_f: number | null;
  dew_point_f: number | null;
  humidity_pct: number | null;
  barometer_inhg: number | null;
  wind_speed_mph: number | null;
  wind_gust_mph: number | null;
  wind_dir_deg: number | null;
  rain_rate_in_hr: number | null;
  rain_today_in: number | null;
  rain_event_in: number | null;
  solar_wm2: number | null;
  uv_index: number | null;
  last_rain_ms: number | null;
  temp_trend: Trend | null;
  barometer_trend: Trend | null;
}

/** The rain event still under way. */
export interface ActiveRainEvent {
  event_id: string;
  start_ms: number;
  duration_minutes: number;
  total_in: number;
  avg_rate_in_hr: number | null;
}

/** A month's or window's gap status: valid, prorated (≈) or invalid. */
export type GapState = "valid" | "prorated" | "invalid";

/** Why a month or window is not valid. */
export type GapReason =
  | "gap"
  | "low_coverage"
  | "not_archived"
  | "before_archive"
  | "in_progress";

/**
 * One month's or one window's gap status (`lookout/snapshot/gaps.py` holds
 * the rules). An invalid one has no value anywhere in the snapshot.
 */
export interface GapStatus {
  /** "YYYY-MM" for a month, else the window's name ("7d", "water_year"). */
  key: string;
  /** Its first station-local day, "YYYY-MM-DD". */
  first_day: string;
  /** Its last station-local day, inclusive. */
  last_day: string;
  days: number;
  /** Days that cannot be counted, whatever the reason. */
  missing_days: number;
  status: GapState;
  /** Null when valid; for a prorated one, why days are missing. */
  reason: GapReason | null;
}

/** The gap status of every archive month and every standard window. */
export interface Gaps {
  /** Oldest first, from the archive's first month through the current one. */
  months: GapStatus[];
  /** today, yesterday, 7d, 30d, 90d, 365d, water_year, in that order. */
  windows: GapStatus[];
}

/**
 * The last 48 hours in 15-minute slots, for the Overview's first screen.
 * Slot `i` starts at `start_ms + i * slot_minutes` minutes; the last slot
 * holds the newest observation. The per-slot arrays are parallel; null means
 * no reading. A slot with a null `rain_in` is not known from the archive (a
 * gap, or the span only the live reading covers) and is hatched.
 */
export interface ConditionRibbon {
  start_ms: number;
  slot_minutes: number;
  /** The newest outdoor temperature in each slot; the last is the live one. */
  temperature_f: (number | null)[];
  /** The rain that fell in each slot, from the daily counter. */
  rain_in: (number | null)[];
  /** Whether the sun was up (any solar radiation) in each slot. */
  daylight: (boolean | null)[];
  /** The lowest and highest reading over the 48 hours, live included. */
  temperature_low_f: number | null;
  temperature_high_f: number | null;
  /** Hours of slots with rain, and hours of slots with a rain value. */
  rain_hours: number;
  known_hours: number;
}

/** Event Catalog quality: "invalid" means no archive data in its range. */
export type EventQuality = "excellent" | "good" | "fair" | "poor" | "invalid";

/**
 * One Event Catalog event as the Rain Events tab lists it. Keyed by
 * `start_ms`, which is stable across refresh runs (the catalog's id is not).
 */
export interface RainEventRow {
  start_ms: number;
  end_ms: number;
  duration_minutes: number;
  /** The live event total while `live`. */
  total_in: number;
  peak_rate_in_hr: number | null;
  quality: EventQuality;
  completeness_pct: number | null;
  max_gap_minutes: number | null;
  /** The water year it started in, named by the year it ends in. */
  water_year: number;
  /** 1 for the wettest listed event. */
  rank: number;
  /**
   * It is the snapshot's `active_rain_event`: the one "raining now"
   * decision, which the hero reads too.
   */
  live: boolean;
  /** False: the archive has no readings for it; show totals only. */
  has_curve: boolean;
}

/** Summary metrics over a set of events. */
export interface EventStats {
  count: number;
  /**
   * The events' totals added up: rain in events only, not a window's rain.
   * Showers under the catalog's minimum are not in it, and an event counts
   * whole in the water year it started in; the water year's rain is
   * `RainTab.water_years`. Label it as rain in events.
   */
  total_in: number;
  median_in: number | null;
  mean_duration_minutes: number | null;
}

/** One water year's events, summarised. */
export interface EventSeason {
  water_year: number;
  stats: EventStats;
}

/**
 * One event's readings, oldest first, as parallel arrays. A reading after a
 * gap of more than 10 minutes carries `gap_minutes` and no rate.
 */
export interface EventCurve {
  start_ms: number;
  end_ms: number;
  t_ms: number[];
  event_in: (number | null)[];
  rate_in_hr: (number | null)[];
  gap_minutes: (number | null)[];
}

/** The Rain Events tab's section of the dashboard. */
export interface RainEvents {
  /** Newest first. */
  events: RainEventRow[];
  all_time: EventStats;
  /** Newest first. */
  seasons: EventSeason[];
  /**
   * The curve of the event the catalog still holds open, live stitched (its
   * curve file is written once the catalog closes it); null with none open.
   * Open is not live: the row's `live` says whether it is still raining.
   */
  open_curve: EventCurve | null;
}

/**
 * `snapshot/<MAC>.events/<start_ms>.json`: one completed event's curve,
 * written once by the refresh run and served by `/event`.
 */
export interface EventCurveDocument extends EventCurve {
  schema_version: number;
  kind: "event_curve";
  station: Station;
}

/** Which rule of the Adaptive Lead Rule leads. */
export type LeadKind = "rain" | "temperature" | "solar" | "calm";

/** Why a lead rule's value or reference is not known. */
export type LeadMissing = "no_event" | "no_reading" | "no_comparison";

/**
 * One lead rule with its live value, for "Why this leads". Built by
 * `lookout/snapshot/lead.py`, which documents each rule.
 */
export interface LeadRule {
  kind: LeadKind;
  /** Whether it holds; calm holds only when no rule before it does. */
  met: boolean;
  /**
   * In `unit`: rain, hours since the newest event ended (0 while one is
   * live); temperature, degrees from the month median (signed); solar,
   * W/m²; calm, days since it last rained.
   */
  value: number | null;
  unit: "hours" | "degrees_f" | "wm2" | "days";
  /** 12 h, 10°, 400 W/m²; null for calm. */
  threshold: number | null;
  /** The month median at this hour (temperature), the usual dry spell (calm). */
  reference: number | null;
  /** The `GapStatus` keys in `gaps` the reference was computed over, newest first. */
  compared: string[];
  missing: LeadMissing | null;
}

/**
 * The hero chart's bold line in slots from `start_ms`, with a usual band per
 * slot for temperature and solar (empty for rain). Null is no reading.
 */
export interface LeadTrace {
  start_ms: number;
  slot_minutes: number;
  values: (number | null)[];
  usual_low: (number | null)[];
  usual_median: (number | null)[];
  usual_high: (number | null)[];
}

/** The Overview's hero lead and every rule's value. */
export interface HeroLead {
  kind: LeadKind;
  /** Rain, temperature, solar, calm, in that order. */
  rules: LeadRule[];
  /** The chart's unit. */
  unit: "in" | "f" | "wm2" | "days";
  /** The bold "now" value, observed at `now_ms`. */
  now: number | null;
  now_ms: number | null;
  /** The usual range now: quartiles of the comparison set. */
  usual_low: number | null;
  usual_median: number | null;
  usual_high: number | null;
  /** How many days, events or dry spells the usual range is over. */
  sample: number;
  /** The bold line over time; null for calm. */
  trace: LeadTrace | null;
}

/** The rain over one standard window, with its own rain gap status. */
export interface RainWindow {
  key: string;
  /**
   * Its rain gap status: `gaps.windows`' with the same key, except that a
   * day whose rain counter has no reading is missing here. Show the total
   * by this status.
   */
  status: GapStatus;
  /** Prorated when its status is; null when invalid. */
  total_in: number | null;
}

/** One station-local hour's rain. */
export interface RainHour {
  day: string;
  /** The hour it starts, 0 to 23 local. */
  hour: number;
  rain_in: number;
}

/** One station-local day's rain. */
export interface RainDay {
  day: string;
  rain_in: number;
}

/**
 * Daily and hourly rain over the days of one standard window ("90d").
 * Per-day arrays are parallel, oldest first, from `first_day`; a day that
 * cannot be counted has a null total, its reason and 24 null hours. Today's
 * total is the live one.
 */
export interface RecentRain {
  window: string;
  first_day: string;
  rain_in: (number | null)[];
  reason: (GapReason | null)[];
  /** Each day's 24 local hours; null for an hour still to come today. */
  hourly_in: (number | null)[][];
  /** Prorated when the window is; null when invalid. */
  wet_hours: number | null;
  wettest_hour: RainHour | null;
}

/** A run of dry days with a rain day on each side. */
export interface DrySpell {
  first_day: string;
  last_day: string;
  days: number;
}

/**
 * Dry spells and rain days in the record. The days since it last rained and
 * the usual dry spell are the calm lead rule's (`HeroLead.rules`).
 */
export interface DrySpells {
  longest: DrySpell | null;
  /** Prorated when the water year is; null when invalid. */
  water_year_rain_days: number | null;
  water_year_wettest: RainDay | null;
}

/**
 * One water year's cumulative rain from Oct 1. Place `i` is `i` days after
 * Oct 1 with Feb 29 folded into Feb 28, so years line up; null where the
 * window from Oct 1 is invalid.
 */
export interface WaterYearCurve {
  /** Named by the year it ends in. */
  water_year: number;
  first_day: string;
  cumulative_in: (number | null)[];
  prorated: boolean[];
  /** The whole year so far. */
  status: GapStatus;
  total_in: number | null;
  /** The value at `RainTab.day_index`: the same date as today. */
  to_date_in: number | null;
}

/** Everything the Rain tab shows. */
export interface RainTab {
  /** In `Gaps.windows` order. */
  windows: RainWindow[];
  recent: RecentRain;
  dry_spells: DrySpells;
  /** Today's place in the water-year curves. */
  day_index: number;
  /** Oldest first. */
  water_years: WaterYearCurve[];
}

/** One window's range bar: low, middle half and high of its readings. */
export interface TemperaturePeriod {
  /** `status.key` names it: today, yesterday, 7d, 30d, 90d, 365d. */
  status: GapStatus;
  low: number | null;
  q1: number | null;
  q3: number | null;
  high: number | null;
  readings: number;
  /** Where `now` sits against the middle half; null when unknown. */
  position: "above" | "below" | "inside" | null;
}

/** One day of the ten-day list against its usual low and high. */
export interface TemperatureDay {
  /** The station-local day, "YYYY-MM-DD". */
  day: string;
  status: GapStatus;
  low: number | null;
  high: number | null;
  /** Median daily low and high on this date in earlier years. */
  usual_low: number | null;
  usual_high: number | null;
  /** The earlier years that usual is over, newest first. */
  usual_years: number[];
  usual_days: number;
}

/** One "now against normal" strip. */
export interface NormalStrip {
  kind: "temperature" | "rain";
  unit: "f" | "in";
  now: number | null;
  now_ms: number | null;
  /** One value per past day, ascending. */
  values: number[];
  below_pct: number | null;
  equal_pct: number | null;
  above_pct: number | null;
  /** The earlier years used, newest first. */
  years: number[];
  missing: "no_reading" | "no_comparison" | null;
}

/** The Overview's temperature context (#81). */
export interface TemperatureContext {
  now: number | null;
  now_ms: number | null;
  /** The window the headline places now in; null when none can be compared. */
  finding: string | null;
  /** Today, yesterday, 7, 30, 90 and 365 days, in that order. */
  periods: TemperaturePeriod[];
  /** The last ten local days, oldest first, ending today. */
  ten_days: TemperatureDay[];
  /** Every earlier year the list's usual is over, newest first. */
  ten_day_years: number[];
  /** The temperature strip, then the rain strip. */
  strips: NormalStrip[];
}

/**
 * One calendar month in the almanac grid: an archive sum with today's live
 * total in it (`lookout/snapshot/almanac.py`).
 */
export interface AlmanacMonth {
  /** Keyed "YYYY-MM"; the current month is invalid, "in_progress". */
  status: GapStatus;
  /** Prorated when its status is; null when invalid. */
  rain_in: number | null;
  /** The current month only: its first day to today. */
  to_date: GapStatus | null;
  /** The rain over `to_date`; null when invalid or there is none. */
  to_date_in: number | null;
}

/** One water year's row: Oct through Sep. */
export interface AlmanacYear {
  /** Named by the year it ends in. */
  water_year: number;
  /** Twelve months, Oct first; null for a month not begun. */
  months: (AlmanacMonth | null)[];
  /** Oct 1 to Sep 30; invalid, "in_progress", while it runs. */
  season: GapStatus;
  season_in: number | null;
}

/** One year's rain over a rolling window ending on today's date that year. */
export interface RollingYear {
  /** The calendar year the window ends in. */
  year: number;
  /** Its rain gap status; the current year's is the `RainTab.windows` status with the same key. */
  status: GapStatus;
  rain_in: number | null;
}

/** One row of the rolling-window rain table. */
export interface RollingWindow {
  /** "7d", "30d", "90d", "365d" or "water_year" (since Oct 1). */
  key: string;
  /** The window ending today, then the same days in earlier years. */
  years: RollingYear[];
}

/** The Overview's almanac grid and rolling-window rain table. */
export interface Almanac {
  /** Newest first. */
  years: AlmanacYear[];
  /** Shortest first, the season last. */
  rolling: RollingWindow[];
}

/** The solar energy over one standard window. */
export interface SolarWindow {
  /** Its key in `Gaps.windows`. */
  key: string;
  /** kWh per m²; prorated when the window is; null when it is invalid. */
  energy_kwh_m2: number | null;
  /** Its gap status for solar: the archive's, with days the energy catalog lacks missing too. */
  status: GapStatus;
}

/**
 * Every day's solar energy from the archive's first day through today,
 * parallel tuples, oldest first. Today's is the energy so far.
 */
export interface SolarDays {
  first_day: string;
  /** kWh per m²; null for a day that cannot be counted. */
  energy_kwh_m2: (number | null)[];
  /** Why a day cannot be counted, null when it can. */
  reason: (GapReason | null)[];
}

/** One month's solar energy; null when the month is invalid or under way. */
export interface SolarMonth {
  /** "YYYY-MM", its key in `Gaps.months`. */
  key: string;
  energy_kwh_m2: number | null;
  /** Its gap status for solar, as `SolarWindow.status`. */
  status: GapStatus;
}

/** The last 90 days, 15 minutes at a time, for picking a day. */
export interface SolarRecent {
  /** The standard window the days are: "90d". */
  window: string;
  first_day: string;
  slot_minutes: number;
  /**
   * Each day's slots by local clock from midnight (96), in Wh per m²; null
   * for a day that cannot be counted, a slot with no period, a slot to come.
   */
  energy_wh_m2: (number | null)[][];
}

/**
 * The 24-hour view: five-minute solar readings up to the newest
 * observation (the last slot, the live reading), over the usual range at
 * each time of day across the 30 days before today.
 */
export interface SolarTrace {
  start_ms: number;
  /** A duration, so not an `_ms` field (those are epoch times). */
  step_minutes: number;
  /** W/m²; null where nothing was read (a gap, or the span only live covers). */
  solar_wm2: (number | null)[];
  /** The comparison days' gap status. */
  usual: GapStatus;
  /** Parallel to `solar_wm2`; null when `usual` is invalid. */
  usual_low_wm2: (number | null)[] | null;
  usual_median_wm2: (number | null)[] | null;
  usual_high_wm2: (number | null)[] | null;
  /** Yesterday and today's gap status: the days the 24 hours reach. */
  energy: GapStatus;
  /** kWh per m² over the 24 hours; null when `energy` is invalid. */
  energy_kwh_m2: number | null;
  /** kWh per m² over the 24 hours before; null when not all counted. */
  previous_kwh_m2: number | null;
}

/** Everything the Solar tab shows. */
export interface SolarTab {
  /** Today to 365 days, in `Gaps.windows` order. */
  windows: SolarWindow[];
  days: SolarDays;
  /** In `Gaps.months` order. */
  months: SolarMonth[];
  recent: SolarRecent;
  /** Null without an observation. */
  last_24h: SolarTrace | null;
}

/** A catalog the refresh run could not update (#118). */
export type CatalogName = "events" | "energy";

/**
 * A catalog the run built the snapshot from as last stored, because its
 * update failed: the Event Catalog ("events", the Rain Events tab) or the
 * energy catalog ("energy", the Solar tab).
 */
export interface StaleCatalog {
  catalog: CatalogName;
  /** The newest time the stored catalog covers; null when none is stored. */
  as_of_ms: number | null;
}

/** The fields both documents share. */
interface SnapshotHead {
  schema_version: number;
  station: Station;
  /** When the refresh run built the snapshot. */
  built_at_ms: number;
  /** The newest observation, live or archived, whichever is newer. */
  observed_at_ms: number | null;
  current: SnapshotConditions;
  active_rain_event: ActiveRainEvent | null;
}

/** `snapshot/<MAC>.dashboard.json`: every field of the snapshot. */
export interface DashboardSnapshot extends SnapshotHead {
  kind: "dashboard";
  /** The archive's newest row, so the site can name the archive lag. */
  archive_newest_ms: number | null;
  /** The gap status of each month and standard window. */
  gaps: Gaps;
  /** The 48-hour condition ribbon, or null without an observation. */
  condition_ribbon: ConditionRibbon | null;

  rain_events: RainEvents;
  /** The hero lead and every lead rule's live value. */
  hero_lead: HeroLead;
  /** The Rain tab's data. */
  rain_tab: RainTab;
  /** Temperature across periods, the ten-day list and the strips. */
  temperature_context: TemperatureContext;
  /** The Overview's almanac grid and rolling-window rain table. */
  almanac: Almanac;
  /** The Solar tab's data; null without an energy catalog. */
  solar_tab: SolarTab | null;
  /**
   * The catalogs this run could not update; empty on a normal run. Absent
   * from documents published before #118, which the parse reads as empty.
   */
  stale_catalogs: StaleCatalog[];
}

/** `snapshot/<MAC>.context.json`: what `/current` serves with `lastData`. */
export interface ContextBlock extends SnapshotHead {
  kind: "context";
}

/**
 * Ambient's live `lastData` for one device, passed through as Ambient sends
 * it. Only `dateutc` (epoch ms UTC) is relied on here.
 */
export interface LastData {
  dateutc: number;
  [field: string]: unknown;
}

/** The body `/current` returns. */
export interface CurrentReading {
  schema_version: number;
  kind: "current";
  station: Station;
  /** The newest observation: the live reading, or the context's if newer. */
  observed_at_ms: number | null;
  /** When the snapshot behind `context` was built, or null without one. */
  built_at_ms: number | null;
  /** Ambient's live reading, or null when Ambient could not be read. */
  last_data: LastData | null;
  /** The context block, or null when Storj could not be read. */
  context: ContextBlock | null;
}

/** Thrown when a document does not match the contract this site reads. */
export class SnapshotContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SnapshotContractError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isTime(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

function checkHead(doc: unknown, kind: string): Record<string, unknown> {
  if (!isRecord(doc)) {
    throw new SnapshotContractError(`${kind}: not a JSON object`);
  }
  if (doc.schema_version !== SCHEMA_VERSION) {
    throw new SnapshotContractError(
      `${kind}: schema_version ${String(doc.schema_version)}, ` +
        `this site reads ${SCHEMA_VERSION}`,
    );
  }
  if (doc.kind !== kind) {
    throw new SnapshotContractError(
      `expected kind "${kind}", got ${JSON.stringify(doc.kind)}`,
    );
  }
  const station = doc.station;
  if (!isRecord(station) || typeof station.mac !== "string") {
    throw new SnapshotContractError(`${kind}: station.mac missing`);
  }
  if (station.name !== null && typeof station.name !== "string") {
    throw new SnapshotContractError(`${kind}: station.name not a string`);
  }
  if (!isTime(doc.built_at_ms)) {
    throw new SnapshotContractError(`${kind}: built_at_ms not an integer`);
  }
  if (doc.observed_at_ms !== null && !isTime(doc.observed_at_ms)) {
    throw new SnapshotContractError(`${kind}: observed_at_ms not an integer`);
  }
  if (!isRecord(doc.current)) {
    throw new SnapshotContractError(`${kind}: current missing`);
  }
  if (doc.active_rain_event !== null && !isRecord(doc.active_rain_event)) {
    throw new SnapshotContractError(`${kind}: active_rain_event malformed`);
  }
  return doc;
}

/**
 * Check a parsed `dashboard.json` against the contract and type it.
 *
 * Checks the head (version, kind, station, times) rather than every field,
 * so an added field never breaks the site.
 *
 * @param doc - The parsed JSON.
 * @returns The same object, typed.
 * @throws SnapshotContractError when the version, kind or head is wrong.
 */
export function parseDashboard(doc: unknown): DashboardSnapshot {
  const head = checkHead(doc, "dashboard");
  if (head.archive_newest_ms !== null && !isTime(head.archive_newest_ms)) {
    throw new SnapshotContractError(
      "dashboard: archive_newest_ms not an integer",
    );
  }
  const gaps = head.gaps;
  if (!isRecord(gaps) || !Array.isArray(gaps.months) || !Array.isArray(gaps.windows)) {
    throw new SnapshotContractError("dashboard: gaps missing");
  }
  checkRibbon(head.condition_ribbon);
  const events = head.rain_events;
  if (!isRecord(events) || !Array.isArray(events.events)) {
    throw new SnapshotContractError("dashboard: rain_events.events missing");
  }
  checkLead(head.hero_lead);
  checkRainTab(head.rain_tab);
  checkTemperature(head.temperature_context);
  checkAlmanac(head.almanac);
  checkSolarTab(head.solar_tab);
  const stale_catalogs = checkStaleCatalogs(head.stale_catalogs);
  return { ...head, stale_catalogs } as unknown as DashboardSnapshot;
}

function checkStaleCatalogs(list: unknown): StaleCatalog[] {
  if (list === undefined) return [];
  if (!Array.isArray(list)) {
    throw new SnapshotContractError("dashboard: stale_catalogs not a list");
  }
  for (const item of list) {
    if (
      !isRecord(item) ||
      (item.catalog !== "events" && item.catalog !== "energy") ||
      (item.as_of_ms !== null && !isTime(item.as_of_ms))
    ) {
      throw new SnapshotContractError("dashboard: stale_catalogs malformed");
    }
  }
  return list as StaleCatalog[];
}

/**
 * The snapshot's stale mark for one catalog.
 *
 * @param snapshot - The dashboard snapshot.
 * @param catalog - Which catalog.
 * @returns Its mark, or null when the run updated it.
 */
export function staleCatalog(
  snapshot: DashboardSnapshot,
  catalog: CatalogName,
): StaleCatalog | null {
  return snapshot.stale_catalogs.find((s) => s.catalog === catalog) ?? null;
}

function checkSolarTab(tab: unknown): void {
  if (tab === null || tab === undefined) return;
  const days = isRecord(tab) ? tab.days : null;
  const recent = isRecord(tab) ? tab.recent : null;
  if (
    !isRecord(tab) ||
    !Array.isArray(tab.windows) ||
    !Array.isArray(tab.months) ||
    !isRecord(days) ||
    !Array.isArray(days.energy_kwh_m2) ||
    !Array.isArray(days.reason) ||
    days.reason.length !== days.energy_kwh_m2.length ||
    !isRecord(recent) ||
    !Array.isArray(recent.energy_wh_m2)
  ) {
    throw new SnapshotContractError("dashboard: solar_tab malformed");
  }
  for (const item of [...tab.windows, ...tab.months]) {
    if (!isRecord(item) || !isRecord(item.status)) {
      throw new SnapshotContractError("dashboard: solar_tab window or month has no gap status");
    }
  }
}

function checkRainTab(tab: unknown): void {
  if (!isRecord(tab) || !Array.isArray(tab.windows) || !Array.isArray(tab.water_years)) {
    throw new SnapshotContractError("dashboard: rain_tab malformed");
  }
  const recent = tab.recent;
  if (
    !isRecord(recent) ||
    !Array.isArray(recent.rain_in) ||
    !Array.isArray(recent.hourly_in) ||
    recent.hourly_in.length !== recent.rain_in.length
  ) {
    throw new SnapshotContractError("dashboard: rain_tab.recent days not parallel");
  }
}

function checkAlmanac(almanac: unknown): void {
  if (!isRecord(almanac) || !Array.isArray(almanac.years) || !Array.isArray(almanac.rolling)) {
    throw new SnapshotContractError("dashboard: almanac malformed");
  }
  for (const year of almanac.years) {
    if (!isRecord(year) || !Array.isArray(year.months) || year.months.length !== 12) {
      throw new SnapshotContractError("dashboard: almanac year is not twelve months");
    }
  }
}

function checkRibbon(ribbon: unknown): void {
  if (ribbon === null || ribbon === undefined) return;
  if (!isRecord(ribbon) || !isTime(ribbon.start_ms)) {
    throw new SnapshotContractError("dashboard: condition_ribbon.start_ms missing");
  }
  const slots = ribbon.temperature_f;
  const parallel = [ribbon.rain_in, ribbon.daylight];
  if (!Array.isArray(slots) || !parallel.every((a) => Array.isArray(a) && a.length === slots.length)) {
    throw new SnapshotContractError("dashboard: condition_ribbon slots not parallel");
  }
}

function checkTemperature(context: unknown): void {
  if (
    !isRecord(context) ||
    !Array.isArray(context.periods) ||
    !Array.isArray(context.ten_days) ||
    !Array.isArray(context.strips)
  ) {
    throw new SnapshotContractError("dashboard: temperature_context missing");
  }
}

function checkLead(lead: unknown): void {
  if (!isRecord(lead) || typeof lead.kind !== "string" || !Array.isArray(lead.rules)) {
    throw new SnapshotContractError("dashboard: hero_lead missing");
  }
}

/**
 * Check a parsed curve file against the contract and type it.
 *
 * @param doc - The parsed JSON.
 * @returns The same object, typed.
 * @throws SnapshotContractError when the version, kind or arrays are wrong.
 */
export function parseEventCurve(doc: unknown): EventCurveDocument {
  if (!isRecord(doc) || doc.kind !== "event_curve") {
    throw new SnapshotContractError('expected kind "event_curve"');
  }
  if (doc.schema_version !== SCHEMA_VERSION) {
    throw new SnapshotContractError(
      `event_curve: schema_version ${String(doc.schema_version)}, ` +
        `this site reads ${SCHEMA_VERSION}`,
    );
  }
  if (!isTime(doc.start_ms) || !isTime(doc.end_ms)) {
    throw new SnapshotContractError("event_curve: start_ms or end_ms not an integer");
  }
  const t = doc.t_ms;
  const columns = [doc.event_in, doc.rate_in_hr, doc.gap_minutes];
  if (
    !Array.isArray(t) ||
    !t.every(isTime) ||
    !columns.every((c) => Array.isArray(c) && c.length === t.length)
  ) {
    throw new SnapshotContractError("event_curve: arrays missing or uneven");
  }
  return doc as unknown as EventCurveDocument;
}

/**
 * Check a parsed `context.json` against the contract and type it.
 *
 * @param doc - The parsed JSON.
 * @returns The same object, typed.
 * @throws SnapshotContractError when the version, kind or head is wrong.
 */
export function parseContext(doc: unknown): ContextBlock {
  return checkHead(doc, "context") as unknown as ContextBlock;
}

/**
 * Check a parsed `/current` body and type it.
 *
 * @param doc - The parsed JSON.
 * @returns The same object, typed.
 * @throws SnapshotContractError when the version, kind or station is wrong.
 */
export function parseCurrent(doc: unknown): CurrentReading {
  if (!isRecord(doc) || doc.kind !== "current") {
    throw new SnapshotContractError('expected kind "current"');
  }
  if (doc.schema_version !== SCHEMA_VERSION) {
    throw new SnapshotContractError(
      `current: schema_version ${String(doc.schema_version)}, ` +
        `this site reads ${SCHEMA_VERSION}`,
    );
  }
  if (!isRecord(doc.station) || typeof doc.station.mac !== "string") {
    throw new SnapshotContractError("current: station.mac missing");
  }
  if (doc.observed_at_ms !== null && !isTime(doc.observed_at_ms)) {
    throw new SnapshotContractError("current: observed_at_ms not an integer");
  }
  if (doc.context !== null) parseContext(doc.context);
  return doc as unknown as CurrentReading;
}

/**
 * The newer of two observation times; null only when both are.
 *
 * @param a - One observation time, or null.
 * @param b - Another, or null.
 * @returns The later one.
 */
export function newestObservation(
  a: number | null | undefined,
  b: number | null | undefined,
): number | null {
  const times = [a, b].filter((t): t is number => typeof t === "number");
  return times.length ? Math.max(...times) : null;
}
