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
 * ("Open the Rain tab"). No `aria-label` replaces the content. A temperature
 * feature (unusual, or a usual day) opens the temperature zoom
 * (`#overview/temperature`, #474). Until the other zooms land, the link goes
 * where the evidence is today: the Rain Events tab for rain, the Rain tab
 * for a dry spell, the Solar tab for unusual sun, and otherwise the
 * Overview tile carrying the reading (`#overview-<section>`: the Wind,
 * Pressure or Humidity tile).
 *
 * Every value is the snapshot's: this module words, formats and links; it
 * does not pick, rank or compare.
 *
 * Offers `featureView` (the tile's words and link, for a snapshot that
 * carries a feature) and `featureTile` (the Overview section).
 */
import "./feature.css";

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
import { spoken } from "./mosaic.js";
import { temperatureTile } from "./mosaic-tiles.js";

/** Where the tile's link goes, and the words that say so. */
interface Target {
  href: string;
  /** E.g. "Open the Rain tab", the end of the link's name. */
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
const RAIN: Target = { href: "#rain", words: "Open the Rain tab" };
// The temperature zoom (#474), in the tile's own words.
const TEMPERATURE: Target = {
  href: `#overview/${temperatureTile.slug}`,
  words: `Open the ${temperatureTile.zoomName} view`,
};

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
    target: { href: "#overview-air", words: "Go to Air" },
  },
  wind_gust: {
    name: "wind gust",
    high: "gustier",
    low: "calmer",
    target: { href: "#overview-wind", words: "Go to Wind" },
  },
  pressure: {
    name: "pressure",
    high: "higher",
    low: "lower",
    target: { href: "#overview-pressure", words: "Go to Pressure" },
  },
  solar: {
    name: "sun",
    high: "brighter",
    low: "duller",
    target: { href: "#solar", words: "Open the Solar tab" },
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
  // The rule held on an event that ended: the catalog's newest, the list's
  // first row (newest first), as Python's rain rule reads it.
  const event = snapshot.rain_events.events.find((row) => !row.live);
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

function unusualView(snapshot: DashboardSnapshot, feature: Feature): FeatureView {
  // The parse guarantees an unusual feature its reading, value and percentile.
  const reading = READINGS[feature.reading!];
  const percentile = feature.percentile!;
  const value = readingValue(feature, feature.value!);
  // Python judges the newest observation's hour (`lookout.snapshot.feature`).
  const hour = hourName(snapshot.observed_at_ms ?? snapshot.built_at_ms);
  // The percentile is the share of usual readings below the value and always
  // lies outside 10 to 90, so above the middle it reads high, below it low.
  const finding =
    `${percentile > 50 ? reading.high : reading.low} than usual for ${hour}: ` +
    `the ${ordinal(percentile)} percentile around this date in earlier years`;
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
      target: RAIN,
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
    target: RAIN,
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
  // Two columns and two rows of the four-column grid, at the left, with the
  // first four mosaic tiles beside it (`mosaic.css`); the whole width of the
  // two-column grid and of a phone.
  span: 2,
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
