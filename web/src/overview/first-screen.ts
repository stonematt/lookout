/**
 * The Overview's first screen (DESIGN.md, "The Phone First-Screen Rule"):
 * under the header's station name and live timestamp, the current
 * temperature with its condition sentence beside it, the Right-now strip
 * (feels like, wind, rain today), then the 48-hour condition ribbon.
 *
 * Every value is the snapshot's live-stitched `current`, the one value every
 * view reads. The archive lag goes in the source line under the ribbon,
 * never in the sentence.
 */
import "./first-screen.css";

import { degrees, formatDuration, formatStamp, inches, mph } from "../format.js";
import type { DashboardSnapshot } from "../snapshot.js";
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
 * The source line: where the numbers came from, and the archive lag.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The sentence.
 */
export function firstScreenSource(snapshot: DashboardSnapshot): string {
  const live = "Temperature, feels like, wind and rain today are the live reading.";
  const ribbon = snapshot.condition_ribbon;
  if (!ribbon) return live;
  const slots = `The ribbon is the archive in ${ribbon.slot_minutes}-minute slots`;
  const { archive_newest_ms: archived, observed_at_ms: observed } = snapshot;
  if (archived === null) return `${live} ${slots}; nothing is archived yet.`;
  // Name the lag only when the ribbon hatches its end for it.
  const lag = observed === null ? 0 : observed - archived;
  if (lag <= 0 || ribbon.rain_in.at(-1) !== null) return `${live} ${slots}.`;
  return (
    `${live} ${slots}, to ${formatStamp(archived)}; the ${formatDuration(lag)} ` +
    "after it is hatched, as only the live total is known there."
  );
}

function render(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const heading = document.createElement("h2");
  heading.className = "ov-sr";
  heading.textContent = "Right now";

  const line = document.createElement("div");
  line.className = "now-line";
  const sentence = part("p", "condition", "now-condition");
  sentence.textContent = conditionSentence(snapshot.current);
  line.append(temperature(snapshot), sentence);

  host.append(heading, line, rightNow(snapshot));
  const ribbon = renderRibbon(snapshot);
  if (ribbon) host.append(ribbon);

  const source = document.createElement("p");
  source.className = "ov-source";
  source.textContent = firstScreenSource(snapshot);
  host.append(source);
}

/** The first screen, as an Overview section. */
export const firstScreen: OverviewSection = {
  id: "first-screen",
  label: "Right now",
  render,
};
