/**
 * The now tile (DESIGN.md "Now tile", "The Phone First-Screen Rule"): the
 * current temperature with its condition sentence beside it (together, the
 * tile's finding), the Right-now strip (feels like, wind, rain today,
 * humidity), then the 48-hour condition ribbon, and the folded source. It is
 * the one tile on `now-surface`, and in the phone's fixed top.
 *
 * Every value is the snapshot's live-stitched `current`, the one value every
 * view reads, and the source note dates it by the snapshot's own reading, so
 * it never borrows the freshness of a newer `/current`. The source note
 * carries no archive lag chip (#458): the header states freshness once, and
 * the ribbon hatches its unarchived end in place.
 *
 * Offers `firstScreen`, the section, and `nowFinding`, its finding in words.
 */
import "./first-screen.css";

import { degrees, formatDuration, formatStamp, inches, known, mph, percent } from "../format.js";
import type { DashboardSnapshot } from "../snapshot.js";
import { sourceNote, tileFolds } from "../tiles.js";
import { conditionSentence } from "./condition.js";
import type { OverviewSection } from "./index.js";
import { renderRibbon } from "./ribbon.js";

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
    ["feels like", known(c.feels_like_f, degrees)],
    ["wind", known(c.wind_speed_mph, mph)],
    ["rain today", known(c.rain_today_in, (v) => inches(v, { digits: 2 }))],
    ["humidity", known(c.humidity_pct, percent)],
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
 * where the ribbon's archive ends.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The sentence.
 */
function firstScreenSource(snapshot: DashboardSnapshot): string {
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
  // Explain the hatch only when the ribbon's last slot is past the archive.
  const lastSlot = ribbon.start_ms + (ribbon.rain_in.length - 1) * ribbon.slot_minutes * 60_000;
  if (lastSlot <= archived || ribbon.rain_in.at(-1) !== null) return `${reading} ${slots}.`;
  return (
    `${reading} ${slots}, to ${formatStamp(archived)}; after it the ribbon is hatched, ` +
    "as only the live total is known there."
  );
}

/**
 * The now tile's finding in words, its accessible name: the temperature,
 * then the condition sentence beside it.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "59.3°F. Mild and lightly breezy; ..."
 */
export function nowFinding(snapshot: DashboardSnapshot): string {
  const t = snapshot.current.temperature_f;
  const temp = t === null ? "Temperature not known" : `${t.toFixed(1)}°F`;
  return `${temp}. ${conditionSentence(snapshot.current)}`;
}

/**
 * How many of the ribbon's archived slots have no rain value: a gap in the
 * record hides that part of the rain track. The unarchived tail is the
 * archive lag's, which the chip names on its own.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The chip's short reason, or null.
 */
function ribbonGaps(snapshot: DashboardSnapshot): string | null {
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
        lag: false,
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
  finding: nowFinding,
  render,
};
