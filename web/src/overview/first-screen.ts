/**
 * The now tile, the Overview's fixed top (spec #460, #468): the current
 * temperature large, then today hour by hour against the usual band for this
 * date (`today-hero.ts`). It is the one tile on `now-surface`, in the
 * phone's fixed top, and the full width of the grid. The 48-hour ribbon,
 * the condition sentence and the Right-now strip it once held are gone.
 *
 * The temperature is the snapshot's live-stitched `current`, the one value
 * every view reads. The tile folds no source (prototype decision 22): its
 * line is in the Overview's sources footer, with no time, as the header
 * states freshness once and the chart hatches its unarchived end in place.
 *
 * Offers `firstScreen`, the section.
 */
import "./first-screen.css";

import { signed } from "../format.js";
import type { DashboardSnapshot } from "../snapshot.js";
import type { TileSource } from "../tiles.js";
import type { OverviewSection } from "./index.js";
import { bandBasisWords, nowDeltaText, todayChart } from "./today-hero.js";

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
  node.append(signed(t), unit);
  return node;
}

/**
 * The now tile's line in the Overview's sources footer: where today's line
 * and the band come from, with no time (the header dates the reading).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns What it reads and how it compares.
 */
function firstScreenSource(snapshot: DashboardSnapshot): TileSource {
  const band = snapshot.temperature_context.usual_band ?? null;
  const usual = band
    ? ` The band is the usual for each hour of the day: ${bandBasisWords(band)}, and the delta is the live reading against its own hour's median.`
    : "";
  const slots = snapshot.condition_ribbon?.slot_minutes ?? 15;
  return {
    what: "the station's live reading, and today's archive",
    method:
      `Today's line is the archive in ${slots}-minute slots; where the archive has not caught up, ` +
      `the live reading is joined by a dotted line, and a stretch missing from the record is hatched.${usual}`,
  };
}

/** The delta line under the temperature; none when the snapshot decided none. */
function deltaLine(snapshot: DashboardSnapshot): HTMLElement | null {
  const text = nowDeltaText(snapshot);
  if (!text) return null;
  const node = document.createElement("p");
  node.className = "now-delta";
  node.dataset.part = "delta";
  node.textContent = text;
  return node;
}

function render(host: HTMLElement, snapshot: DashboardSnapshot): void {
  // The temperature is the tile's finding; the chart sits beside it on the
  // grids and under it on a phone.
  const line = document.createElement("div");
  line.className = "now-line";
  line.dataset.part = "finding";
  line.append(temperature(snapshot));
  const delta = deltaLine(snapshot);
  if (delta) line.append(delta);
  const chart = todayChart(snapshot);
  host.append(line, chart.block);
  chart.start();
}

/** The now tile, as an Overview section. */
export const firstScreen: OverviewSection = {
  id: "first-screen",
  label: "Right now",
  kicker: "Right now",
  span: 4,
  phoneTop: true,
  now: true,
  source: firstScreenSource,
  render,
};
