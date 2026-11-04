/**
 * The now tile (DESIGN.md "Now tile", "The Phone First-Screen Rule"): the
 * current temperature with its condition sentence beside it (together, the
 * tile's finding), the Right-now strip (feels like, wind, rain today,
 * humidity), then the 48-hour condition ribbon, and the folded source. It is
 * the one tile on `now-surface`, and in the phone's fixed top.
 *
 * Every value is the snapshot's live-stitched `current`, the one value every
 * view reads, and the source note dates it by the snapshot's own reading, so
 * it never borrows the freshness of a newer `/current`. The archive lag goes
 * in the source note under the ribbon, never in the sentence.
 */
import "./first-screen.css";

import { degrees, formatDuration, formatStamp, inches, mph, percent } from "../format.js";
import type { DashboardSnapshot } from "../snapshot.js";
import { sourceNote, tileFolds } from "../tiles.js";
import { conditionSentence } from "./condition.js";
import type { OverviewSection } from "./index.js";
import { renderRibbon } from "./ribbon.js";

/** A value in its unit, or "n/a" with none. */
function shown(value: number | null, format: (v: number) => string): string {
  return value === null ? "n/a" : format(value);
}

function part(tag: keyof HTMLElementTagNameMap, name: string, className: string): HTMLElement {
  const node = document.createElement(tag);
  node.className = className;
  node.dataset.part = name;
  return node;
}

function temperature(snapshot: DashboardSnapshot): HTMLElement {
  const node = part("p", "temperature", "now-temp");
  const t = snapshot.current.temperature_f;
  if (t === null) {
    node.textContent = "n/a";
    return node;
  }
  const unit = document.createElement("sup");
  unit.textContent = "°F";
  node.append(t.toFixed(1), unit);
  return node;
}

function rightNow(snapshot: DashboardSnapshot): HTMLElement {
  const c = snapshot.current;
  const list = part("dl", "right-now", "right-now");
  list.setAttribute("aria-label", "Right now");
  const items: [string, string][] = [
    ["feels like", shown(c.feels_like_f, degrees)],
    ["wind", shown(c.wind_speed_mph, mph)],
    ["rain today", shown(c.rain_today_in, (v) => inches(v, { digits: 2 }))],
    ["humidity", shown(c.humidity_pct, percent)],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    item.append(dt, dd);
    list.append(item);
  }
  return list;
}

/**
 * The source line: where the numbers came from and when they were read, and
 * the archive lag.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The sentence.
 */
export function firstScreenSource(snapshot: DashboardSnapshot): string {
  const { archive_newest_ms: archived, observed_at_ms: observed } = snapshot;
  // Dated by the snapshot's own reading: a fresher `/current` in the header
  // lends these figures nothing.
  const strip = "Temperature, feels like, wind and rain today";
  const reading =
    observed === null
      ? `${strip} have no reading yet.`
      : `${strip} are the station's reading at ${formatStamp(observed)}.`;
  const ribbon = snapshot.condition_ribbon;
  if (!ribbon) return reading;
  const slots = `The ribbon is the archive in ${ribbon.slot_minutes}-minute slots`;
  if (archived === null) return `${reading} ${slots}; nothing is archived yet.`;
  // Name the lag only when the ribbon hatches its end for it.
  const lag = observed === null ? 0 : observed - archived;
  if (lag <= 0 || ribbon.rain_in.at(-1) !== null) return `${reading} ${slots}.`;
  return (
    `${reading} ${slots}, to ${formatStamp(archived)}; the ${formatDuration(lag)} ` +
    "after it is hatched, as only the live total is known there."
  );
}

/**
 * How many of the ribbon's archived slots have no rain value: a gap in the
 * record hides that part of the rain track. The unarchived tail is the
 * archive lag's, which the chip names on its own.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The chip's short reason, or null.
 */
export function ribbonGaps(snapshot: DashboardSnapshot): string | null {
  const ribbon = snapshot.condition_ribbon;
  const archived = snapshot.archive_newest_ms;
  if (!ribbon || archived === null) return null;
  const slotMs = ribbon.slot_minutes * 60_000;
  const gaps = ribbon.rain_in.filter(
    (rain, i) => rain === null && ribbon.start_ms + i * slotMs <= archived,
  ).length;
  return gaps ? `${formatDuration(gaps * slotMs)} of the ribbon missing` : null;
}

function render(host: HTMLElement, snapshot: DashboardSnapshot): void {
  // The temperature beside its condition sentence is the tile's finding.
  const line = part("div", "finding", "now-line");
  const sentence = part("p", "condition", "now-condition");
  sentence.textContent = conditionSentence(snapshot.current);
  line.append(temperature(snapshot), sentence);

  host.append(line, rightNow(snapshot));
  const ribbon = renderRibbon(snapshot);
  if (ribbon) host.append(ribbon);

  host.append(
    tileFolds(
      sourceNote(snapshot, {
        what: "the station's reading",
        method: firstScreenSource(snapshot),
        gaps: ribbonGaps(snapshot),
      }),
    ),
  );
}

/** The now tile, as an Overview section. */
export const firstScreen: OverviewSection = {
  id: "first-screen",
  label: "Right now",
  kicker: "Right now",
  span: 2,
  phoneTop: true,
  now: true,
  render,
};
