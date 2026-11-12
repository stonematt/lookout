/**
 * The now tile, the Overview's fixed top (spec #460, #468): the current
 * temperature large, then today hour by hour against the usual band for this
 * date (`today-hero.ts`), and the folded source. It is the one tile on
 * `now-surface`, in the phone's fixed top, and the full width of the grid.
 * The 48-hour ribbon, the condition sentence and the Right-now strip it once
 * held are gone.
 *
 * The temperature is the snapshot's live-stitched `current`, the one value
 * every view reads, and the source note dates it by the snapshot's own
 * reading, so it never borrows the freshness of a newer `/current`. The
 * source note carries no archive lag chip (#458): the header states
 * freshness once, and the chart hatches its unarchived end in place.
 *
 * Offers `firstScreen`, the section, and `nowFinding`, its finding in words.
 */
import "./first-screen.css";

import { formatDuration, formatStamp } from "../format.js";
import type { DashboardSnapshot } from "../snapshot.js";
import { sourceNote, tileFolds } from "../tiles.js";
import type { OverviewSection } from "./index.js";
import { bandBasisWords, todayChart, todayMissingMs } from "./today-hero.js";

function temperature(snapshot: DashboardSnapshot): HTMLElement {
  const node = document.createElement("p");
  node.className = "now-temp";
  node.dataset.part = "temperature";
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

/**
 * The source line: where the numbers came from and when they were read, and
 * where today's line and the band come from.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The sentences.
 */
function firstScreenSource(snapshot: DashboardSnapshot): string {
  const { archive_newest_ms: archived, observed_at_ms: observed } = snapshot;
  // Dated by the snapshot's own reading: a fresher `/current` in the header
  // lends this figure nothing.
  const reading =
    observed === null
      ? "The temperature has no reading yet."
      : `The temperature is the station's reading at ${formatStamp(observed)}.`;
  const band = snapshot.temperature_context.usual_band ?? null;
  const usual = band
    ? ` The band is the usual for each hour of the day: ${bandBasisWords(band)}.`
    : "";
  const ribbon = snapshot.condition_ribbon;
  if (!ribbon) return `${reading}${usual}`;
  const slots = `Today's line is the archive in ${ribbon.slot_minutes}-minute slots`;
  if (archived === null) return `${reading} ${slots}; nothing is archived yet.${usual}`;
  // Explain the hatch only when the last slot is past the archive.
  const lastSlot = ribbon.start_ms + (ribbon.rain_in.length - 1) * ribbon.slot_minutes * 60_000;
  if (lastSlot <= archived || ribbon.rain_in.at(-1) !== null) return `${reading} ${slots}.${usual}`;
  return (
    `${reading} ${slots}, to ${formatStamp(archived)}; after it the live reading is joined by ` +
    `a dotted line.${usual}`
  );
}

/**
 * The now tile's finding in words, its accessible name.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "59.3°F now."
 */
export function nowFinding(snapshot: DashboardSnapshot): string {
  const t = snapshot.current.temperature_f;
  return t === null ? "Temperature not known." : `${t.toFixed(1)}°F now.`;
}

/** The chip's short reason when a gap in the record hides part of today. */
function todayGaps(snapshot: DashboardSnapshot): string | null {
  const missing = todayMissingMs(snapshot);
  return missing ? `${formatDuration(missing)} of today missing` : null;
}

function render(host: HTMLElement, snapshot: DashboardSnapshot): void {
  // The temperature is the tile's finding; the chart sits beside it on the
  // grids and under it on a phone.
  const line = document.createElement("div");
  line.className = "now-line";
  line.dataset.part = "finding";
  line.append(temperature(snapshot));
  const chart = todayChart(snapshot);
  host.append(line, chart.block);
  chart.start();
  host.append(
    tileFolds(
      sourceNote(snapshot, {
        what: "the station's reading",
        method: firstScreenSource(snapshot),
        gaps: todayGaps(snapshot),
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
  span: 4,
  phoneTop: true,
  now: true,
  finding: nowFinding,
  render,
};
