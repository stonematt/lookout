/**
 * The Rain tab (#83): daily rain, dry spells, year-over-year accumulation
 * and the accumulation heatmap, everything read from the snapshot's
 * `rain_tab` (built by `lookout/snapshot/rain_tab.py`).
 *
 * Each section runs finding headline, standfirst, chart, readout, source
 * line (DESIGN.md). Every window value follows its gap status: hatched with
 * the reason when invalid, ≈ when prorated. Today's rain is the live total,
 * the same number the Overview shows.
 */
import "./rain.css";

import { formatStamp } from "../format.js";
import { hatchGap } from "../gaps.js";
import type { DashboardSnapshot, RainTab } from "../snapshot.js";
import { drawDailyBars, drawHourGrid, drawSized, drawYearCurves } from "./charts.js";
import {
  WINDOW_NAMES,
  dailyHeadline,
  dayText,
  daysText,
  dryHeadline,
  hoursHeadline,
  inches,
  windowStatus,
  windowsNote,
  windowValue,
  yearLeftOut,
  yearsHeadline,
} from "./text.js";

/** The rain steps the heatmap key decodes (`rainStep` at 60 minutes). */
const KEY_STEPS: [string, string][] = [
  ["light", "up to 0.05 in"],
  ["moderate", "up to 0.10 in"],
  ["steady", "up to 0.30 in"],
  ["heavy", "over 0.30 in"],
];

/** An element with a class and, optionally, text. */
function make<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** One section: headline, standfirst, and a body to fill. */
function section(id: string, headline: string, standfirst: string): HTMLElement {
  const host = make("section", "rain-section");
  host.dataset.section = id;
  const title = make("h2", "rain-title", headline);
  title.id = `rain-${id}-title`;
  host.setAttribute("aria-labelledby", title.id);
  host.append(title, make("p", "standfirst", standfirst));
  return host;
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

/** The source line. */
function source(host: HTMLElement, sentences: (string | null)[]): void {
  host.append(make("p", "rain-source", sentences.filter(Boolean).join(" ")));
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

/** The daily section: window totals, then 90 days of bars. */
function daily(snapshot: DashboardSnapshot, charts: (() => void)[]): HTMLElement {
  const tab = snapshot.rain_tab;
  const host = section(
    "daily",
    dailyHeadline(snapshot),
    "Rain over each window ending today, then each day of the last 90; the line marked today is the live total.",
  );
  const strip = make("dl", "rain-stats rain-windows");
  for (const window of tab.windows) {
    const dd = stat(strip, WINDOW_NAMES[window.key] ?? window.key, windowValue(snapshot, window));
    hatchGap(dd, windowStatus(snapshot, window.key));
  }
  host.append(strip);
  const { chart, readout } = figure(host, "Rain each day over the last 90 days");
  charts.push(() =>
    drawSized(chart, (width) => drawDailyBars(chart, tab.recent, width, (t) => (readout.textContent = t))),
  );
  const todayStatus = windowStatus(snapshot, "today");
  const live = snapshot.current.rain_today_in;
  const liveNote =
    todayStatus.status === "invalid" && live !== null
      ? `The live reading has ${inches(live)} today; only the live total is known until the archive catches up.`
      : null;
  source(host, [
    "Source: the station archive, each day's highest daily rain counter, with today's live total.",
    snapshot.archive_newest_ms === null ? null : `Archived through ${formatStamp(snapshot.archive_newest_ms)}.`,
    liveNote,
    windowsNote(snapshot),
  ]);
  return host;
}

/** The dry-spell section: the current spell against the record. */
function dry(snapshot: DashboardSnapshot): HTMLElement {
  const spells = snapshot.rain_tab.dry_spells;
  const host = section(
    "dry",
    dryHeadline(snapshot.rain_tab),
    "How long it has been since the last rain day, against dry spells that ended near this date in the record.",
  );
  const strip = make("dl", "rain-stats rain-dry");
  const lastRain = snapshot.current.last_rain_ms;
  stat(strip, "Last rain", lastRain === null ? null : formatStamp(lastRain));
  stat(strip, "Usual spell now", spells.usual_days === null ? null : daysText(spells.usual_days));
  const longest = spells.longest;
  stat(
    strip,
    "Longest on record",
    longest ? daysText(longest.days) : null,
    longest ? `${dayText(longest.first_day)} to ${dayText(longest.last_day, true)}` : undefined,
  );
  const wyStatus = windowStatus(snapshot, "water_year");
  const rainDays = spells.water_year_rain_days;
  const daysDd = stat(
    strip,
    "Rain days this water year",
    rainDays === null ? null : `${wyStatus.status === "prorated" ? "≈ " : ""}${Math.round(rainDays)}`,
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
  source(host, [
    "Source: the station archive's daily totals; the last rain is the station's own record.",
    "A rain day has at least 0.01 in. Spells with a day missing from the record are left out, so a missing day is never read as dry.",
    spells.usual_sample
      ? `The usual spell is the median of ${spells.usual_sample} that ended within 30 days of this date in any year.`
      : "No completed dry spell ended within 30 days of this date yet.",
  ]);
  return host;
}

/** The year-over-year section: one curve per water year. */
function years(snapshot: DashboardSnapshot, charts: (() => void)[]): HTMLElement {
  const tab: RainTab = snapshot.rain_tab;
  const host = section(
    "years",
    yearsHeadline(tab),
    "Rain since Oct 1, one line per water year, named by the year it ends in. The bold line is this year.",
  );
  const { chart, readout } = figure(host, "Rain since Oct 1, one line per water year");
  charts.push(() =>
    drawSized(chart, (width) => drawYearCurves(chart, tab, width, (t) => (readout.textContent = t))),
  );
  const left = tab.water_years
    .filter((c) => c.cumulative_in.every((v) => v === null))
    .map(yearLeftOut);
  source(host, [
    "Source: the station archive's daily totals, with today's live total.",
    "Feb 29 is folded into Feb 28 so the years line up by date; ≈ marks a total scaled up over a few missing days.",
    ...left,
  ]);
  return host;
}

/** The heatmap section: rain by the hour over the last 90 days. */
function hours(snapshot: DashboardSnapshot, charts: (() => void)[]): HTMLElement {
  const tab = snapshot.rain_tab;
  const host = section(
    "hours",
    hoursHeadline(tab),
    "Each column is a day of the last 90 and each row an hour, midnight at the top; the key shows the rain each shade stands for.",
  );
  const { chart, readout } = figure(host, "Rain by the hour over the last 90 days");
  charts.push(() =>
    drawSized(chart, (width) => drawHourGrid(chart, tab.recent, width, (t) => (readout.textContent = t))),
  );
  const key = make("ul", "rain-key");
  key.setAttribute("aria-label", "Key");
  KEY_STEPS.forEach(([word, amount], i) => {
    const item = make("li", "");
    item.append(make("span", `rain-swatch rain-step-${i + 1}`), document.createTextNode(`${word}, ${amount}`));
    key.append(item);
  });
  const gap = make("li", "");
  gap.append(make("span", "rain-swatch gap-hatch"), document.createTextNode("not in the record"));
  key.append(gap);
  readout.before(key);
  source(host, [
    "Source: the station archive and the live reading. Each hour's rain is the rise in the station's daily counter within it; rain counted during a gap lands in the hour after it.",
  ]);
  return host;
}

/**
 * Render the Rain tab into its panel, replacing what was there.
 *
 * @param panel - The Rain tab's panel.
 * @param snapshot - The dashboard snapshot.
 */
export function renderRain(panel: HTMLElement, snapshot: DashboardSnapshot): void {
  panel.replaceChildren();
  if (!snapshot.rain_tab) {
    panel.append(make("p", "standfirst", "The rain record is not in this snapshot yet."));
    return;
  }
  const charts: (() => void)[] = [];
  panel.append(
    daily(snapshot, charts),
    dry(snapshot),
    years(snapshot, charts),
    hours(snapshot, charts),
  );
  // Charts measure their width, so they draw once their sections are placed.
  for (const draw of charts) draw();
}
