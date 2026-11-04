/**
 * The Rain tab (#83): daily rain, dry spells, year-over-year accumulation
 * and the accumulation heatmap, everything read from the snapshot's
 * `rain_tab` (built by `lookout/snapshot/rain_tab.py`).
 *
 * Each section is a tile (`web/src/tiles.ts`, spec #332): kicker, finding,
 * standfirst, chart or stats, then the folded source note, whose chip names
 * the archive lag and what the chart hatches. `SECTIONS` is the registry, in
 * the Overview's shape. Every window value follows its gap status: hatched
 * with the reason when invalid, ≈ when prorated. Today's rain is the live
 * total, the same number the Overview shows.
 */
import "./rain.css";

import { dayText, formatDayCount, formatDays, formatStamp, inches, WET_DAY_IN } from "../format.js";
import { gapElement, gapValue, hatchGap } from "../gaps.js";
import type { DashboardSnapshot, RainTab } from "../snapshot.js";
import { drawDays } from "../charts/days-switch.js";
import { drawSized, missingDay } from "../charts/svg.js";
import { make } from "../dom.js";
import type { TileSection } from "../tiles.js";
import {
  gapReasons,
  hatchedCount,
  renderTiles,
  sourceNote,
  tileFinding,
  tileFolds,
} from "../tiles.js";
import { drawDailyBars, drawHourGrid, drawYearCurves, lastDays } from "./charts.js";
import {
  WINDOW_NAMES,
  calmRule,
  dailyHeadline,
  daysText,
  dryHeadline,
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

/**
 * The tile's finding and standfirst, its first parts after the kicker.
 *
 * @param host - The tile.
 * @param finding - The finding.
 * @param standfirst - How to read the chart.
 */
function heading(host: HTMLElement, finding: string, standfirst: string): void {
  const lede = make("p", "standfirst", standfirst);
  lede.dataset.part = "standfirst";
  host.append(tileFinding(finding), lede);
}

/** A chart figure with its live readout; returns both. */
function figure(host: HTMLElement, name: string): { chart: HTMLElement; readout: HTMLElement } {
  const chart = make("figure", "rain-chart");
  chart.setAttribute("aria-label", `${name}. Use the arrow keys to read each mark.`);
  const readout = make("p", "rain-readout");
  readout.setAttribute("aria-live", "polite");
  host.append(chart, readout);
  return { chart, readout };
}

/**
 * The folded source note at the tile's foot: the archive time on its
 * closed line, the method inside, and a chip when the archive lags or a gap
 * hides part of the chart.
 *
 * @param host - The tile.
 * @param snapshot - The dashboard snapshot.
 * @param what - What the numbers come from, short.
 * @param sentences - The method paragraph's sentences; nulls are dropped.
 * @param gaps - What the chart hatches, as the chip's reason; null for none.
 */
function source(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  what: string,
  sentences: (string | null)[],
  gaps: string | null,
): void {
  const method = sentences.filter(Boolean).join(" ");
  host.append(tileFolds(sourceNote(snapshot, { what, method, gaps })));
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

/** The daily tile: window totals, then 90 days of bars. */
function renderDaily(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const tab = snapshot.rain_tab;
  const lede = (days: number) =>
    `Rain over each window ending today, then each day of the last ${days}; the line marked today is the live total.`;
  heading(host, dailyHeadline(snapshot), lede(90));
  // On the four-column grid the strip stands beside the bars (tiles.css).
  const split = make("div", "tile-split");
  const strip = make("dl", "rain-stats rain-windows");
  for (const window of tab.windows) {
    const dd = stat(strip, WINDOW_NAMES[window.key] ?? window.key, windowValue(window));
    // Compact in the strip: the source note says why each one is not shown.
    hatchGap(dd, window.status, { compact: true });
  }
  split.append(strip);
  host.append(split);
  const bars = make("div", "tile-split-chart");
  split.append(bars);
  const { chart, readout } = figure(bars, "Rain each day over the last 90 days");
  drawDays(chart, {
    all: tab.recent.rain_in.length,
    name: (days) => `Rain each day over the last ${days} days`,
    standfirst: host.querySelector("[data-part=standfirst]"),
    lede,
    draw: (days, width) =>
      drawDailyBars(chart, lastDays(tab.recent, days), width, (t) => (readout.textContent = t)),
  });
  const todayStatus = rainStatus(snapshot, "today");
  const live = snapshot.current.rain_today_in;
  const liveNote =
    todayStatus.status === "invalid" && live !== null
      ? `The live reading has ${inches(live)} today; only the live total is known until the archive catches up.`
      : null;
  source(
    host,
    snapshot,
    "daily rain counter",
    [
      "The station archive, each day's highest daily rain counter, with today's live total.",
      liveNote,
      windowsNote(snapshot),
    ],
    gapReasons([
      [hatchedCount(tab.windows.map((w) => w.status)), "window", "windows"],
      [hatchedDays(tab), "day", "days"],
    ]),
  );
}

/** The dry-spell tile: the current spell against the record. */
function renderDry(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const spells = snapshot.rain_tab.dry_spells;
  heading(
    host,
    dryHeadline(snapshot),
    "How long it has been since it last rained, against the dry spells of the last 365 days, and the longest in the record.",
  );
  const strip = make("dl", "rain-stats rain-dry-stats");
  const lastRain = snapshot.current.last_rain_ms;
  stat(strip, "Last rain", lastRain === null ? null : formatStamp(lastRain));
  const calm = calmRule(snapshot);
  stat(strip, "Since it last rained", calm?.value == null ? null : formatDays(calm.value));
  const usualDd = stat(
    strip,
    "Usual dry spell",
    calm?.reference == null ? null : formatDayCount(calm.reference),
  );
  if (calm?.reference == null) hatchGap(usualDd, windowStatus(snapshot, "365d"));
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
  source(
    host,
    snapshot,
    "daily totals",
    [
      "The station archive's daily totals; the last rain is the station's own record.",
      `A rain day has at least ${WET_DAY_IN} in. Spells with a day missing from the record are left out, so a missing day is never read as dry.`,
      "The usual dry spell is the median completed spell in the last 365 days, the same comparison the Overview's calm lead makes.",
    ],
    gapReasons([[strip.querySelectorAll("dd.gap-hatch").length, "figure", "figures"]]),
  );
}

/** The year-over-year tile: one curve per water year. */
function renderYears(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const tab: RainTab = snapshot.rain_tab;
  heading(
    host,
    yearsHeadline(tab),
    "Rain since Oct 1, one line per water year, named by the year it ends in. The bold line is this year.",
  );
  const current = tab.water_years[tab.water_years.length - 1];
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
  const left = tab.water_years.filter((c) => c.cumulative_in.every((v) => v === null));
  source(
    host,
    snapshot,
    "daily totals with today's live total",
    [
      "The station archive's daily totals, with today's live total.",
      "Feb 29 is folded into Feb 28 so the years line up by date; ≈ marks a total scaled up over a few missing days.",
      ...left.map(yearLeftOut),
      yearsSkipped(tab).length
        ? `Not compared by this date, so left out of the headline: ${yearsSkipped(tab).join(", ")}.`
        : null,
    ],
    gapReasons([[left.length, "water year", "water years"]]),
  );
}

/** The heatmap tile: rain by the hour over the last 90 days. */
function renderHours(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const tab = snapshot.rain_tab;
  const lede = (days: number) =>
    `Each column is a day of the last ${days} and each row an hour, midnight at the top; the key shows the rain each shade stands for.`;
  heading(host, hoursHeadline(snapshot), lede(90));
  const { chart, readout } = figure(host, "Rain by the hour over the last 90 days");
  drawDays(chart, {
    all: tab.recent.rain_in.length,
    name: (days) => `Rain by the hour over the last ${days} days`,
    standfirst: host.querySelector("[data-part=standfirst]"),
    lede,
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
  source(
    host,
    snapshot,
    "archive and live reading, hourly",
    [
      "The station archive and the live reading. Each hour's rain is the rise in the station's daily counter within it; rain counted during a gap lands in the hour after it.",
    ],
    gapReasons([[hatchedDays(tab), "day", "days"]]),
  );
}

/** One tile of the Rain tab. */
export type RainSection = TileSection;

/**
 * The Rain tab's sections, in the desktop board's order: the windows and
 * the daily bars across the grid (the phone's fixed top), then the water
 * year beside the dry spells, then rain by the hour across the grid.
 */
export const SECTIONS: readonly RainSection[] = [
  {
    id: "daily",
    label: "Rain over each window, and each day of the last 90",
    kicker: "Rain, by window",
    span: 4,
    phoneTop: true,
    render: renderDaily,
  },
  {
    id: "years",
    label: "Rain since Oct 1, each water year",
    kicker: "The water year",
    span: 2,
    render: renderYears,
  },
  {
    id: "dry",
    label: "Dry spells",
    kicker: "Dry spells",
    span: 2,
    render: renderDry,
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
