/**
 * The feature tile (#469, spec #460 stories 10–17, 52–53): one tile under
 * the now tile showing the snapshot's `feature`, picked by the Feature Rule
 * in Python (ADR-0009): the rain event, the unusual reading with its
 * percentile, the long dry spell, or a usual day. It replaces "What leads
 * right now" and the browser rules that went with it.
 *
 * The tile is one link: kicker, value and finding all sit inside it, and its
 * accessible name comes from its content, as the mosaic tiles' does (#543):
 * the visible finding, read once, with hidden words round it that say the
 * kicker and value in full (both `aria-hidden`) and where the link goes
 * ("Open the dry spell view"). No `aria-label` replaces the content. A
 * feature opens the zoom of the tile that holds its evidence: a temperature
 * feature (unusual, or a usual day) the temperature zoom (#474), a long dry
 * spell the dry spell zoom (#475), an unusual gust the wind zoom (#476), and
 * unusual humidity, pressure or sun that reading's zoom (#477). A rain event
 * has no zoom: it opens the Rain Events tab.
 *
 * Every value is the snapshot's: this module words, formats and links; it
 * does not pick, rank or compare. A percentile at either end reads as past
 * every reading for this hour ("highest for this hour on record"), never
 * "0th" or "100th" (#563); an unusual reading's hour names its day when it
 * is from before midnight (`readingDayWords`, the hero's).
 *
 * Offers `featureView` (the tile's words and link, for a snapshot that
 * carries a feature) and `featureTile` (the Overview section).
 */
import "./feature.css";

import { spoken } from "../dom.js";
import {
  degrees,
  formatDays,
  formatDaysSpoken,
  formatDayCount,
  formatDuration,
  formatDurationSpoken,
  hourName,
  inches,
  inHg,
  mph,
  ordinal,
  percent,
} from "../format.js";
import type { DashboardSnapshot, Feature, FeatureReading } from "../snapshot.js";
import { tileFinding } from "../tiles.js";
import type { OverviewSection } from "./index.js";
import type { MosaicTile } from "./mosaic.js";
import { humidityTile, pressureTile, sunTile, windTile } from "./mosaic-conditions.js";
import { drySpellTile, temperatureTile } from "./mosaic-tiles.js";
import { readingDayWords } from "./today-hero.js";

/** Where the tile's link goes, and the words that say so. */
interface Target {
  href: string;
  /** E.g. "Open the Rain Events tab", the end of the link's name. */
  words: string;
}

/** What the feature tile shows. */
export interface FeatureView {
  kind: Feature["kind"];
  kicker: string;
  /** The value's number and its unit (null when the number carries it). */
  value: [string, string | null] | null;
  /** The finding beside the value. */
  finding: string;
  /**
   * The words a screen reader hears around the visible finding, which it
   * reads once. The kicker and the value are `aria-hidden`; their spoken
   * forms are here.
   */
  spoken: { before: string; after: string };
  /** Whether a rain event is live: the kicker carries the rain dot. */
  live: boolean;
  target: Target;
}

const RAIN_EVENTS: Target = { href: "#events", words: "Open the Rain Events tab" };
/** A mosaic tile's zoom as a target, in the tile's own words. */
const zoomTarget = (tile: MosaicTile): Target => ({
  href: `#overview/${tile.slug}`,
  words: `Open the ${tile.zoomName} view`,
});
const TEMPERATURE = zoomTarget(temperatureTile);
const DRY_SPELL = zoomTarget(drySpellTile);

/** Per reading: its name, the two comparatives, its Overview target. */
const READINGS: Record<
  FeatureReading,
  { name: string; high: string; low: string; target: Target }
> = {
  temperature: { name: "temperature", high: "warmer", low: "cooler", target: TEMPERATURE },
  humidity: {
    name: "humidity",
    high: "more humid",
    low: "drier",
    target: zoomTarget(humidityTile),
  },
  wind_gust: {
    name: "wind gust",
    high: "gustier",
    low: "calmer",
    target: zoomTarget(windTile),
  },
  pressure: {
    name: "pressure",
    high: "higher",
    low: "lower",
    target: zoomTarget(pressureTile),
  },
  solar: {
    name: "sun",
    high: "brighter",
    low: "duller",
    target: zoomTarget(sunTile),
  },
};

const upperFirst = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

/** A value as written in a sentence: "0.30 in", "72°". */
const valueText = (value: [string, string | null]) =>
  value[1] === null ? value[0] : `${value[0]} ${value[1]}`;

/** An unusual reading's value, number and unit apart. */
function readingValue(feature: Feature, value: number): [string, string | null] {
  switch (feature.unit) {
    case "pct":
      return [percent(value), null];
    case "mph":
      return [mph(value, { unit: false }), "mph"];
    case "inhg":
      return [inHg(value), "inHg"];
    case "wm2":
      return [String(Math.round(value)), "W/m²"];
    default:
      return [degrees(value), null];
  }
}

/** The usual event, as the end of a rain finding: ", against 0.42 in for a usual event". */
function againstUsualEvent(snapshot: DashboardSnapshot): string {
  const median = snapshot.rain_events.all_time.median_in;
  return median === null ? "" : `, against ${inches(median, { digits: 2 })} for a usual event`;
}

function rainView(snapshot: DashboardSnapshot): FeatureView {
  const usual = againstUsualEvent(snapshot);
  const active = snapshot.active_rain_event;
  if (active) {
    const value: [string, string] = [inches(active.total_in, { digits: 2, unit: false }), "in"];
    const finding = `so far${usual}`;
    return {
      kind: "rain",
      kicker: `Raining now · ${formatDuration(active.duration_minutes * 60_000)}`,
      value,
      finding,
      spoken: { before: `Raining now: ${valueText(value)} `, after: "" },
      live: true,
      target: RAIN_EVENTS,
    };
  }
  // The rule held on an event that ended: the one the feature names by its
  // start (`event_start_ms`, #523). A snapshot before A3 names none; then
  // the catalog's newest, the list's first row (newest first), as Python's
  // rain rule read it.
  const start = snapshot.feature?.event_start_ms;
  const event =
    typeof start === "number"
      ? snapshot.rain_events.events.find((row) => row.start_ms === start)
      : snapshot.rain_events.events.find((row) => !row.live);
  if (!event) {
    return {
      kind: "rain",
      kicker: "Rain",
      value: null,
      finding: "Rain lately",
      spoken: { before: "", after: "" },
      live: false,
      target: RAIN_EVENTS,
    };
  }
  const ago = Math.max(0, snapshot.built_at_ms - event.end_ms);
  const value: [string, string] = [inches(event.total_in, { digits: 2, unit: false }), "in"];
  const finding = `over ${formatDuration(event.duration_minutes * 60_000)}${usual}`;
  return {
    kind: "rain",
    kicker: `Rain ended ${formatDuration(ago)} ago`,
    value,
    finding,
    spoken: {
      before: `Rain ended ${formatDurationSpoken(ago)} ago, after ${valueText(value)} `,
      after: "",
    },
    live: false,
    target: RAIN_EVENTS,
  };
}

/**
 * Where an unusual reading sits among its usual readings at this hour, in
 * words. At either end it is past every one of them, which "0th" or "100th
 * percentile" would misstate (#563). The feature carries no count of the
 * years it compared with, so an end reads "on record".
 *
 * @param percentile - The feature's percentile, 0 to 100.
 * @returns E.g. "the 95th percentile around this date in earlier years",
 *   "highest for this hour on record", "lowest for this hour on record".
 */
function percentileWords(percentile: number): string {
  if (percentile >= 100) return "highest for this hour on record";
  if (percentile <= 0) return "lowest for this hour on record";
  return `the ${ordinal(percentile)} percentile around this date in earlier years`;
}

function unusualView(snapshot: DashboardSnapshot, feature: Feature): FeatureView {
  // The parse guarantees an unusual feature its reading, value and percentile.
  const reading = READINGS[feature.reading!];
  const percentile = feature.percentile!;
  const value = readingValue(feature, feature.value!);
  // A held reading back inside its usual band (Release A3's 3-hour hold,
  // #533; #578): say it was outside and when, from `since_ms`, with no
  // percentile as now. Absent `outside_usual` (a snapshot before Release A3)
  // counts as outside. The snapshot carries no side (#584 adds one), so the
  // wording names none.
  const backInsideBand = feature.outside_usual === false;
  if (backInsideBand) {
    const when = feature.since_ms === undefined ? "" : ` at ${hourName(feature.since_ms)}`;
    const finding = `${upperFirst(reading.name)} was outside its usual range${when}; back inside now`;
    return {
      kind: "unusual",
      kicker: `Earlier · ${reading.name}`,
      value,
      finding,
      spoken: { before: `${valueText(value)} now. `, after: "" },
      live: false,
      target: reading.target,
    };
  }
  // Python judges the newest observation's hour (`lookout.snapshot.feature`);
  // just after midnight that hour is yesterday's, and the words say so.
  const hour =
    hourName(snapshot.observed_at_ms ?? snapshot.built_at_ms) + readingDayWords(snapshot);
  // The percentile is the share of usual readings below the value and always
  // lies outside 10 to 90, so above the middle it reads high, below it low.
  const finding = `${percentile > 50 ? reading.high : reading.low} than usual for ${hour}: ${percentileWords(percentile)}`;
  return {
    kind: "unusual",
    kicker: `Unusual · ${reading.name}`,
    value,
    finding,
    spoken: { before: `${upperFirst(reading.name)} ${valueText(value)}, `, after: "" },
    live: false,
    target: reading.target,
  };
}

function drySpellView(snapshot: DashboardSnapshot): FeatureView {
  const days = snapshot.dry_spell?.days_since_rain ?? null;
  const usualDays = snapshot.dry_spell?.usual_days ?? null;
  const usual =
    usualDays === null ? "" : `, against a usual dry spell of ${formatDayCount(usualDays)}`;
  if (days === null) {
    const finding = "No last rain on record";
    return {
      kind: "dry_spell",
      kicker: "Dry spell",
      value: null,
      finding,
      spoken: { before: "", after: "" },
      live: false,
      target: DRY_SPELL,
    };
  }
  // The value and the name say it as the Rain tab does: its stat ("1 d 14 h")
  // and its finding ("about 39 hours since it last rained").
  const finding = `since it last rained${usual}`;
  return {
    kind: "dry_spell",
    kicker: "Dry spell",
    value: [formatDays(days), null],
    finding,
    spoken: { before: `${upperFirst(formatDaysSpoken(days))} `, after: "" },
    live: false,
    target: DRY_SPELL,
  };
}

function quietView(snapshot: DashboardSnapshot): FeatureView {
  const t = snapshot.current.temperature_f;
  const value: [string, null] | null = t === null ? null : [degrees(t), null];
  return {
    kind: "quiet",
    kicker: "Today",
    value,
    finding: "a usual day",
    spoken: { before: "", after: value === null ? "" : `, ${valueText(value)} now` },
    live: false,
    target: TEMPERATURE,
  };
}

/**
 * What the feature tile shows for a snapshot.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The tile's words and link; null when the snapshot carries no
 *   feature (one published before Release A).
 */
export function featureView(snapshot: DashboardSnapshot): FeatureView | null {
  const feature = snapshot.feature;
  if (!feature) return null;
  switch (feature.kind) {
    case "rain":
      return rainView(snapshot);
    case "unusual":
      return unusualView(snapshot, feature);
    case "dry_spell":
      return drySpellView(snapshot);
    case "quiet":
      return quietView(snapshot);
  }
}

function render(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const view = featureView(snapshot);
  if (!view) return;
  host.dataset.kind = view.kind;

  const link = document.createElement("a");
  link.className = "feature-link";
  link.href = view.target.href;

  // The kicker renderTiles put first moves into the link, with the chevron.
  const kicker = host.querySelector<HTMLElement>(":scope > [data-part=kicker]");
  const top = document.createElement("div");
  top.className = "feature-top";
  if (kicker) {
    if (view.live) {
      const dot = document.createElement("span");
      dot.className = "feature-live-dot";
      dot.dataset.part = "live";
      dot.setAttribute("aria-hidden", "true");
      kicker.prepend(dot);
    }
    // Its spoken form is the hidden words before the finding.
    kicker.setAttribute("aria-hidden", "true");
    top.append(kicker);
  }
  const chevron = document.createElement("span");
  chevron.className = "feature-chevron";
  chevron.setAttribute("aria-hidden", "true");
  top.append(chevron);

  const body = document.createElement("div");
  body.className = "feature-body";
  if (view.value) {
    const value = document.createElement("p");
    value.className = "feature-value";
    value.dataset.part = "value";
    value.setAttribute("aria-hidden", "true");
    const number = document.createElement("span");
    number.className = "feature-number";
    number.textContent = view.value[0];
    value.append(number);
    if (view.value[1] !== null) {
      const unit = document.createElement("span");
      unit.className = "feature-unit";
      // A no-break space, so the unit never wraps from its number.
      unit.textContent = `${String.fromCharCode(160)}${view.value[1]}`;
      value.append(unit);
    }
    body.append(value);
  }
  // The finding is read once, as written; the hidden words round it carry
  // what the aria-hidden kicker and value show, and where the link goes.
  if (view.spoken.before) body.append(spoken(view.spoken.before));
  body.append(tileFinding(view.finding, { lead: true }));
  body.append(spoken(`${view.spoken.after}. ${view.target.words}.`));

  link.append(top, body);
  host.replaceChildren(link);
}

/** The feature tile, as an Overview section. */
export const featureTile: OverviewSection = {
  id: "feature",
  label: "The feature",
  kicker: (snapshot) => featureView(snapshot)?.kicker ?? "The feature",
  // The whole row of the four-column grid, under the now tile and above the
  // mosaic (prototype decision 23, #545); the whole width of the two-column
  // grid and of a phone.
  span: 4,
  phoneTop: true,
  // A snapshot without a feature (published before Release A, ADR-0008)
  // draws no feature tile.
  shown: (snapshot) => Boolean(snapshot.feature),
  source: () => ({
    what: "the refresh run's choice, by the Feature Rule",
    method:
      "A rain event that is live or ended in the last 12 hours leads; then a reading below the 10th " +
      "or above the 90th percentile of its readings at this hour within a week of this date in " +
      "earlier years, the furthest out winning; " +
      "then a dry spell longer than the usual one; otherwise a usual day.",
  }),
  render,
};
