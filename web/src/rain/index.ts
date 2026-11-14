/**
 * The Rain tab (#83): the water year with its window strip, daily rain, dry
 * spells and the accumulation heatmap, everything read from the snapshot's
 * `rain_tab` (built by `lookout/snapshot/rain_tab.py`).
 *
 * Each section is a tile (`web/src/tiles.ts`, spec #332): kicker, finding,
 * chart or stats, then the folded source note, whose chip names what the
 * chart hatches. No standfirst and no lag chip (spec #460): the header
 * states freshness once. `SECTIONS` is the registry, in the Overview's
 * shape, led by the water year (#475) so the tab's top section is the rain
 * zoom's. A window the record does not cover yet is left out of the strip
 * (#470); one it covers is hatched with the reason when invalid, ≈ when
 * prorated. Today's rain is the live total, the same number the Overview
 * shows.
 *
 * Offers `renderRain`, `SECTIONS`, and the two sections the Overview's
 * zooms draw too, one home per chart: `renderYears` (the rain zoom) and
 * `renderDry` (the dry spell zoom), each taking `SectionOptions` to draw
 * in a zoom: no source note (prototype decision 22) and its finding under
 * the panel's title (`SectionOptions` in `tiles.ts`).
 */
import "./rain.css";

import { dayText, formatDayCount, formatDays, formatStamp, inches, WET_DAY_IN } from "../format.js";
import { beyondRecord, gapElement, gapValue, hatchGap } from "../gaps.js";
import type { DashboardSnapshot, RainTab } from "../snapshot.js";
import { drawDays } from "../charts/days-switch.js";
import { drawSized, missingDay } from "../charts/svg.js";
import { make } from "../dom.js";
import { isShown } from "../overview/history.js";
import type { SectionOptions, TileSection } from "../tiles.js";
import { gapReasons, hatchedCount, renderTiles, sectionSource, tileFinding } from "../tiles.js";
import { drawDailyBars, drawHourGrid, drawYearCurves, lastDays } from "./charts.js";
import {
  WINDOW_NAMES,
  dailyHeadline,
  daysText,
  dryHeadline,
  drySpellNow,
  hoursHeadline,
  rainStatus,
  windowStatus,
  windowsNote,
  windowValue,
  yearLeftOut,
  yearsHeadline,
  yearsSkipped,
} from "./text.js";

/** The rain steps the heatmap key decodes (`rainStep` at 60 minutes). */
const KEY_STEPS: [string, string][] = [
  ["light", "up to 0.05 in"],
  ["moderate", "up to 0.10 in"],
  ["steady", "up to 0.30 in"],
  ["heavy", "over 0.30 in"],
];

/** A chart figure with its live readout; returns both. */
function figure(host: HTMLElement, name: string): { chart: HTMLElement; readout: HTMLElement } {
  const chart = make("figure", "rain-chart");
  chart.setAttribute("aria-label", `${name}. Use the arrow keys to read each mark.`);
  const readout = make("p", "rain-readout");
  readout.setAttribute("aria-live", "polite");
  host.append(chart, readout);
  return { chart, readout };
}

/** Days the bars and the hour grid hatch: the ones not in the record. */
function hatchedDays(tab: RainTab): number {
  return tab.recent.reason.filter(missingDay).length;
}

/** A stat strip item: value above its label. */
function stat(list: HTMLElement, label: string, value: string | null, note?: string): HTMLElement {
  const item = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value ?? "";
  item.append(dt, dd);
  if (note) item.append(make("dd", "rain-stat-note", note));
  list.append(item);
  return dd;
}

/**
 * The window strip: each window's rain, ending today. A window the record
 * does not cover yet is left out, not hatched (#470); one it covers but
 * cannot compare is hatched, compact, and the source note says why.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The strip, or null when the record covers no window.
 */
function windowStrip(snapshot: DashboardSnapshot): HTMLElement | null {
  const windows = snapshot.rain_tab.windows.filter((w) => isShown(w.status));
  if (!windows.length) return null;
  const strip = make("dl", "rain-stats rain-windows");
  strip.dataset.part = "windows";
  for (const window of windows) {
    const dd = stat(strip, WINDOW_NAMES[window.key] ?? window.key, windowValue(window));
    hatchGap(dd, window.status, { compact: true });
  }
  return strip;
}

/** The daily tile: each day of the last 90 as a bar. */
function renderDaily(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const tab = snapshot.rain_tab;
  host.append(tileFinding(dailyHeadline(snapshot)));
  const { chart, readout } = figure(host, "Rain each day over the last 90 days");
  drawDays(chart, {
    all: tab.recent.rain_in.length,
    name: (days) => `Rain each day over the last ${days} days`,
    draw: (days, width) =>
      drawDailyBars(chart, lastDays(tab.recent, days), width, (t) => (readout.textContent = t)),
  });
  sectionSource(
    host,
    snapshot,
    {},
    "daily rain counter",
    [
      "One bar for each day; the day marked today is the live total.",
      "The station archive, each day's highest daily rain counter, with today's live total.",
    ],
    gapReasons([[hatchedDays(tab), "day", "days"]]),
  );
}

/**
 * The dry-spell section: the current spell against the record. The Rain
 * tab's tile and the dry spell zoom both draw it.
 *
 * @param host - The tile, or the zoom's panel.
 * @param snapshot - The dashboard snapshot.
 * @param options - Whether it draws in a zoom panel.
 */
export function renderDry(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  options: SectionOptions = {},
): void {
  const spells = snapshot.rain_tab.dry_spells;
  host.append(tileFinding(dryHeadline(snapshot), options));
  const strip = make("dl", "rain-stats rain-dry-stats");
  const lastRain = snapshot.current.last_rain_ms;
  stat(strip, "Last rain", lastRain === null ? null : formatStamp(lastRain));
  const spell = drySpellNow(snapshot);
  stat(
    strip,
    "Since it last rained",
    spell.days_since_rain === null ? null : formatDays(spell.days_since_rain),
  );
  const usualDd = stat(
    strip,
    "Usual dry spell",
    spell.usual_days === null ? null : formatDayCount(spell.usual_days),
  );
  if (spell.usual_days === null) hatchGap(usualDd, windowStatus(snapshot, "365d"));
  const longest = spells.longest;
  stat(
    strip,
    "Longest on record",
    longest ? daysText(longest.days) : null,
    longest ? `${dayText(longest.first_day)} to ${dayText(longest.last_day, true)}` : undefined,
  );
  const wyStatus = rainStatus(snapshot, "water_year");
  const rainDays = spells.water_year_rain_days;
  const daysDd = stat(
    strip,
    "Rain days this water year",
    rainDays === null ? null : gapValue(wyStatus, String(Math.round(rainDays))),
  );
  hatchGap(daysDd, wyStatus);
  const wettest = spells.water_year_wettest;
  const wetDd = stat(
    strip,
    "Wettest day this water year",
    wettest ? inches(wettest.rain_in) : wyStatus.status === "invalid" ? null : "none",
    wettest ? dayText(wettest.day, true) : undefined,
  );
  hatchGap(wetDd, wyStatus);
  for (const dd of strip.querySelectorAll("dd:not(.gap-hatch)")) {
    if (!dd.textContent) dd.textContent = "not known";
  }
  host.append(strip);
  sectionSource(
    host,
    snapshot,
    options,
    "daily totals",
    [
      "How long it has been since it last rained, against the dry spells of the last 365 days, and the longest in the record.",
      "The station archive's daily totals; the last rain is the station's own record.",
      `A rain day has at least ${WET_DAY_IN} in. Spells with a day missing from the record are left out, so a missing day is never read as dry.`,
      "The usual dry spell is the median completed spell in the last 365 days, the same comparison the Overview's Dry spell tile makes.",
    ],
    gapReasons([[strip.querySelectorAll("dd.gap-hatch").length, "figure", "figures"]]),
  );
}

/**
 * The water-year section: one curve per water year, then the window strip
 * under the chart. The Rain tab's top tile and the rain zoom both draw it;
 * the zoom leaves the strip to its rain gauge panel (#619).
 *
 * @param host - The tile, or the zoom's panel.
 * @param snapshot - The dashboard snapshot.
 * @param options - Whether it draws in a zoom panel.
 */
export function renderYears(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  options: SectionOptions = {},
): void {
  const tab: RainTab = snapshot.rain_tab;
  host.append(tileFinding(yearsHeadline(tab), options));
  const current = tab.water_years[tab.water_years.length - 1];
  // When the archive holds none of any water year there is no chart, not a
  // hatched block (#606).
  if (!tab.water_years.every((c) => beyondRecord(c.status))) {
    if (current && tab.water_years.every((c) => c.cumulative_in.every((v) => v === null))) {
      // No year can be compared: the hatch in place of an empty chart.
      const block = gapElement(current.status);
      block.classList.add("rain-years-gap");
      host.append(block);
    } else {
      const { chart, readout } = figure(host, "Rain since Oct 1, one line per water year");
      drawSized(chart, (width) =>
        drawYearCurves(chart, tab, width, (t) => (readout.textContent = t)),
      );
    }
  }
  // In the zoom the rain gauge, its own panel, draws the windows (#619).
  const strip = options.zoom ? null : windowStrip(snapshot);
  if (strip) host.append(strip);
  const left = tab.water_years.filter((c) => c.cumulative_in.every((v) => v === null));
  const live = snapshot.current.rain_today_in;
  const liveNote =
    rainStatus(snapshot, "today").status === "invalid" && live !== null
      ? `The live reading has ${inches(live)} today; only the live total is known until the archive catches up.`
      : null;
  const hatchedWindows = hatchedCount(tab.windows.map((w) => w.status).filter(isShown));
  sectionSource(
    host,
    snapshot,
    options,
    "daily totals with today's live total",
    [
      "Rain since Oct 1, one line per water year, named by the year it ends in; the bold line is this year. Under it, the rain over each window ending today.",
      "The station archive's daily totals, with today's live total.",
      liveNote,
      "Feb 29 is folded into Feb 28 so the years line up by date; ≈ marks a total scaled up over a few missing days.",
      ...left.map(yearLeftOut),
      yearsSkipped(tab).length
        ? `Not compared by this date, so left out of the headline: ${yearsSkipped(tab).join(", ")}.`
        : null,
      windowsNote(snapshot),
    ],
    gapReasons([
      [left.length, "water year", "water years"],
      [hatchedWindows, "window", "windows"],
    ]),
  );
}

/** The heatmap tile: rain by the hour over the last 90 days. */
function renderHours(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const tab = snapshot.rain_tab;
  host.append(tileFinding(hoursHeadline(snapshot)));
  const { chart, readout } = figure(host, "Rain by the hour over the last 90 days");
  drawDays(chart, {
    all: tab.recent.rain_in.length,
    name: (days) => `Rain by the hour over the last ${days} days`,
    draw: (days, width) =>
      drawHourGrid(chart, lastDays(tab.recent, days), width, (t) => (readout.textContent = t)),
  });
  const key = make("ul", "rain-key");
  key.setAttribute("aria-label", "Key");
  KEY_STEPS.forEach(([word, amount], i) => {
    const item = make("li", "");
    item.append(
      make("span", `rain-swatch rain-step-${i + 1}`),
      document.createTextNode(`${word}, ${amount}`),
    );
    key.append(item);
  });
  const gap = make("li", "");
  gap.append(make("span", "rain-swatch gap-hatch"), document.createTextNode("not in the record"));
  key.append(gap);
  readout.before(key);
  sectionSource(
    host,
    snapshot,
    {},
    "archive and live reading, hourly",
    [
      "Each column is a day and each row an hour, midnight at the top; the key shows the rain each shade stands for.",
      "The station archive and the live reading. Each hour's rain is the rise in the station's daily counter within it; rain counted during a gap lands in the hour after it.",
    ],
    gapReasons([[hatchedDays(tab), "day", "days"]]),
  );
}

/** One tile of the Rain tab. */
export type RainSection = TileSection;

/**
 * The Rain tab's sections (#475): the water year with its window strip
 * across the grid (the phone's fixed top, and the rain zoom's content),
 * then the daily bars beside the dry spells, then rain by the hour across
 * the grid.
 */
export const SECTIONS: readonly RainSection[] = [
  {
    id: "years",
    label: "Rain since Oct 1, each water year, and over each window",
    kicker: "The water year",
    span: 4,
    phoneTop: true,
    render: (host, snapshot) => renderYears(host, snapshot),
  },
  {
    id: "daily",
    label: "Rain each day of the last 90",
    kicker: "Rain, each day",
    span: 2,
    render: renderDaily,
  },
  {
    id: "dry",
    label: "Dry spells",
    kicker: "Dry spells",
    span: 2,
    render: (host, snapshot) => renderDry(host, snapshot),
  },
  {
    id: "hours",
    label: "Rain by the hour over the last 90 days",
    kicker: "Rain by the hour",
    span: 4,
    render: renderHours,
  },
];

/**
 * Render the Rain tab into its panel as a grid of tiles, replacing what was
 * there.
 *
 * @param panel - The Rain tab's panel.
 * @param snapshot - The dashboard snapshot.
 * @param sections - The sections to render, in order.
 */
export function renderRain(
  panel: HTMLElement,
  snapshot: DashboardSnapshot,
  sections: readonly RainSection[] = SECTIONS,
): void {
  if (!snapshot.rain_tab) {
    panel.replaceChildren(make("p", "standfirst", "The rain record is not in this snapshot yet."));
    return;
  }
  renderTiles(panel, snapshot, sections);
}
