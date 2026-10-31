/**
 * The Solar tab (#85): the last 24 hours against the usual range, daily
 * energy, a picked day's 15-minute energy, and the solar heatmaps, all read
 * from the snapshot's `solar_tab` (built by `lookout/snapshot/solar.py`).
 *
 * Each section runs finding headline, standfirst, chart, readout, source
 * line (DESIGN.md). Every value over a day, a month or a window follows its
 * gap status: hatched with the reason when invalid, ≈ when prorated.
 *
 * The selected day lives in the URL (`?solar_day=YYYY-MM-DD`), so a shared
 * link opens on it. Picking a day on the daily bars, either day heatmap or
 * the date field updates the URL in place and redraws what shows it.
 */
import "./solar.css";

import { formatStamp } from "../format.js";
import { gapElement, gapSentence, hatchGap } from "../gaps.js";
import { FALLBACK_WIDTH, dayStatus, drawSized } from "../rain/charts.js";
import { WINDOW_NAMES, addDays, dayText, windowStatus } from "../rain/text.js";
import type { DashboardSnapshot, SolarTab } from "../snapshot.js";
import {
  drawDailyBars,
  drawDayGrid,
  drawDaySlots,
  drawMonthDay,
  drawTrace,
  drawYearMonth,
  shadeKey,
} from "./charts.js";
import {
  FIRST_SLOT,
  LAST_SLOT,
  brightestDay,
  brightestMonth,
  dailyHeadline,
  dayHeadline,
  kwh,
  lastDay,
  monthText,
  openingDay,
  readDay,
  slotText,
  traceHeadline,
  wh,
  windowValue,
  withSolarDay,
} from "./text.js";

/** The heatmap views, as the segmented control names them. */
const VIEWS = [
  ["months", "Year by month"],
  ["days", "Month by day"],
  ["slots", "Day by 15 minutes"],
] as const;

type View = (typeof VIEWS)[number][0];

/** An element with a class and, optionally, text. */
function make<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** One section: headline, standfirst, and a body to fill. */
function section(id: string, headline: string, standfirst: string): HTMLElement {
  const host = make("section", "solar-section");
  host.dataset.section = id;
  const title = make("h2", "solar-title", headline);
  title.id = `solar-${id}-title`;
  host.setAttribute("aria-labelledby", title.id);
  host.append(title, make("p", "standfirst", standfirst));
  return host;
}

/** A chart figure with its live readout. */
function figure(name: string): { chart: HTMLElement; readout: HTMLElement } {
  const chart = make("figure", "solar-chart");
  chart.setAttribute("aria-label", `${name}. Use the arrow keys to read each mark.`);
  const readout = make("p", "solar-readout");
  readout.setAttribute("aria-live", "polite");
  return { chart, readout };
}

/** The source line. */
function source(sentences: (string | null)[]): HTMLElement {
  return make("p", "solar-source", sentences.filter(Boolean).join(" "));
}

/** Draw at the host's width now and on resize; returns a forced redraw. */
function sized(host: HTMLElement, draw: (width: number) => void): () => void {
  let width = 0;
  drawSized(host, (w) => {
    width = w;
    draw(w);
  });
  return () => draw(width || Math.floor(host.clientWidth) || FALLBACK_WIDTH);
}

/** The archive line for a source. */
function archived(snapshot: DashboardSnapshot): string | null {
  return snapshot.archive_newest_ms === null ? null : `Archived through ${formatStamp(snapshot.archive_newest_ms)}.`;
}

/** What the tab draws later, once its sections are placed. */
interface Mount {
  setup: (() => void)[];
  redraws: (() => void)[];
}

/** The 24-hour section. */
function lastDaySection(snapshot: DashboardSnapshot, tab: SolarTab, mount: Mount): HTMLElement {
  const trace = tab.last_24h;
  const live = snapshot.current.solar_wm2;
  const host = section(
    "now",
    trace ? traceHeadline(trace, live) : "No solar reading yet",
    "The bold line is solar radiation over the last 24 hours. The shaded band is the middle half of the 30 days before at each time of day, the dashed line their median.",
  );
  if (!trace) {
    host.append(source(["Source: the station's solar sensor. There is no observation in this snapshot yet."]));
    return host;
  }
  const { chart, readout } = figure("Solar radiation over the last 24 hours");
  host.append(chart, readout);
  mount.setup.push(() => {
    sized(chart, (w) => drawTrace(chart, trace, snapshot.archive_newest_ms, w, (t) => (readout.textContent = t)));
  });
  host.append(
    source([
      "Source: the station's five-minute solar readings, the live reading at the end; energy is the sum of the energy catalog's 15-minute periods.",
      archived(snapshot),
      trace.usual.status === "invalid" ? `No usual range: ${gapSentence(trace.usual)}` : null,
      trace.energy_kwh_m2 === null ? `The 24-hour energy is not totalled: ${gapSentence(trace.energy)}` : null,
    ]),
  );
  return host;
}

/** The daily section: window totals, then 90 days of bars. */
function dailySection(snapshot: DashboardSnapshot, tab: SolarTab, state: { day: string }, pick: (day: string) => void, mount: Mount): HTMLElement {
  const host = section(
    "daily",
    dailyHeadline(snapshot),
    "Energy over each window ending today, then each day of the last 90. Select a day to see its 15 minutes below.",
  );
  const strip = make("dl", "solar-stats");
  for (const window of tab.windows) {
    const item = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = WINDOW_NAMES[window.key] ?? window.key;
    const dd = document.createElement("dd");
    dd.textContent = windowValue(snapshot, window) ?? "";
    item.append(dt, dd);
    strip.append(item);
    hatchGap(dd, windowStatus(snapshot, window.key), { compact: true });
  }
  const { chart, readout } = figure("Solar energy each day over the last 90 days");
  host.append(strip, chart, readout);
  mount.setup.push(() => {
    mount.redraws.push(sized(chart, (w) => drawDailyBars(chart, tab, state.day, w, (t) => (readout.textContent = t), pick)));
  });
  const invalid = tab.windows
    .filter((w) => w.energy_kwh_m2 === null)
    .map((w) => gapSentence(windowStatus(snapshot, w.key)));
  host.append(
    source([
      "Source: the energy catalog, the station's solar readings summed into 15-minute periods; a day is the sum of its periods, today the energy so far.",
      archived(snapshot),
      "A day missing from the record is hatched, never read as a dark day.",
      invalid.length ? `Not shown: ${invalid.join(" ")}` : null,
    ]),
  );
  return host;
}

/** The selected day's section, redrawn in place on every pick. */
function daySection(tab: SolarTab, state: { day: string }, pick: (day: string) => void): { host: HTMLElement; render: () => void } {
  const host = make("section", "solar-section solar-day");
  host.dataset.section = "day";
  const title = make("h2", "solar-title");
  title.id = "solar-day-title";
  host.setAttribute("aria-labelledby", title.id);
  const standfirst = make("p", "standfirst", "Each bar is 15 minutes of the selected day, 4 am to 10 pm. The marked bar is its brightest.");

  const controls = make("div", "solar-controls");
  const prev = make("button", "solar-button", "Previous day");
  prev.type = "button";
  const field = make("label", "solar-field");
  field.append(document.createTextNode("Day "));
  const input = make("input", "solar-date");
  input.type = "date";
  input.min = tab.days.first_day;
  input.max = lastDay(tab);
  field.append(input);
  const next = make("button", "solar-button", "Next day");
  next.type = "button";
  controls.append(prev, field, next);
  prev.addEventListener("click", () => pick(addDays(state.day, -1)));
  next.addEventListener("click", () => pick(addDays(state.day, 1)));
  input.addEventListener("change", () => {
    if (input.value && input.value >= input.min && input.value <= input.max) pick(input.value);
  });

  const body = make("div", "solar-day-body");
  const note = make("p", "solar-source");
  host.append(title, standfirst, controls, body, note);

  function render(): void {
    const day = readDay(tab, state.day);
    title.textContent = dayHeadline(day);
    input.value = day.day;
    prev.disabled = day.day <= tab.days.first_day;
    next.disabled = day.day >= lastDay(tab);
    if (day.reason !== null || day.energy === null) {
      body.replaceChildren(gapElement(dayStatus(day.day, day.reason ?? "gap")));
      note.textContent = "";
      return;
    }
    if (day.slots === null) {
      body.replaceChildren(make("p", "solar-older", `${dayText(day.day, true)} gave ${kwh(day.energy)}. Its 15-minute energy is kept for the last 90 days only.`));
      note.textContent = "Source: the energy catalog's daily sum.";
      return;
    }
    const { chart, readout } = figure(`Solar energy each 15 minutes on ${dayText(day.day, true)}`);
    body.replaceChildren(chart, readout);
    const slots = day.slots;
    drawSized(chart, (w) => drawDaySlots(chart, day, w, (t) => (readout.textContent = t)));
    const outside = slots.slice(0, FIRST_SLOT).concat(slots.slice(LAST_SLOT + 1)).some((v) => v !== null && v > 0);
    note.textContent = [
      "Source: the energy catalog's 15-minute periods, in Wh per m².",
      day.isToday ? "Today's bars stop at the archive's newest period." : null,
      outside ? `Some energy fell before ${slotText(FIRST_SLOT)} or after ${slotText(LAST_SLOT + 1)}; the total counts it.` : null,
    ]
      .filter(Boolean)
      .join(" ");
  }

  return { host, render };
}

/** The heatmap section, with a control to switch between its views. */
function heatmapSection(snapshot: DashboardSnapshot, tab: SolarTab, state: { day: string }, pick: (day: string) => void, mount: Mount): HTMLElement {
  const host = make("section", "solar-section");
  host.dataset.section = "heatmaps";
  const title = make("h2", "solar-title");
  title.id = "solar-heatmaps-title";
  host.setAttribute("aria-labelledby", title.id);
  const standfirst = make("p", "standfirst");
  const control = make("div", "solar-seg");
  control.setAttribute("role", "group");
  control.setAttribute("aria-label", "Heatmap");
  const { chart, readout } = figure("Solar heatmap");
  const key = make("ul", "solar-key");
  key.setAttribute("aria-label", "Key");
  const note = make("p", "solar-source");
  host.append(title, standfirst, control, chart, key, readout, note);

  let view: View = "days";
  const buttons = VIEWS.map(([id, name]) => {
    const button = make("button", "", name);
    button.type = "button";
    button.addEventListener("click", () => {
      view = id;
      show();
    });
    control.append(button);
    return [id, button] as const;
  });

  function fillKey(max: number, unit: string): void {
    key.replaceChildren();
    for (const [cls, words] of shadeKey(max, unit)) {
      const item = make("li", "");
      item.append(make("span", `solar-swatch ${cls}`), document.createTextNode(words));
      key.append(item);
    }
    const gap = make("li", "");
    gap.append(make("span", "solar-swatch gap-hatch"), document.createTextNode("not in the record"));
    key.append(gap);
  }

  let width = 0;
  function draw(w: number): void {
    width = w;
    const write = (t: string) => (readout.textContent = t);
    if (view === "months") drawYearMonth(chart, tab, snapshot.gaps.months, w, write);
    else if (view === "days") drawMonthDay(chart, tab, state.day, w, write, pick);
    else drawDayGrid(chart, tab, state.day, w, write, pick);
  }

  function show(): void {
    for (const [id, button] of buttons) button.setAttribute("aria-pressed", String(id === view));
    readout.textContent = "";
    if (view === "months") {
      const best = brightestMonth(tab);
      title.textContent = best ? `${monthText(best.key)} is the brightest month on record, at ${kwh(best.energy)}` : "Solar energy each month";
      standfirst.textContent = "One row per year, one column per month; the deeper the shade, the more energy. The month under way is not compared yet.";
      fillKey(Math.max(1, ...tab.months.map((m) => m.energy_kwh_m2 ?? 0)), "kWh/m²");
      note.textContent = "Source: the energy catalog's daily sums. ≈ marks a month scaled up over a few missing days; a hatched month has too many.";
    } else if (view === "days") {
      const best = brightestDay(tab);
      title.textContent = best ? `The brightest day on record was ${dayText(best.day, true)}, at ${kwh(best.energy)}` : "Solar energy each day";
      standfirst.textContent = "One row per month, newest at the top, one column per day; select a day to see its 15 minutes.";
      fillKey(Math.max(1, ...tab.days.energy_kwh_m2.map((v) => v ?? 0)), "kWh/m²");
      note.textContent = "Source: the energy catalog's daily sums. The outlined day is the one selected.";
    } else {
      const peak = recentPeak(tab);
      title.textContent = peak
        ? `The brightest 15 minutes of the last 90 days: ${dayText(peak.day, true)}, ${slotText(peak.slot)}, ${wh(peak.energy)}`
        : "Solar energy each 15 minutes over the last 90 days";
      standfirst.textContent = "One column per day of the last 90, one row per 15 minutes from 4 am at the top to 10 pm; select a day to see it below.";
      fillKey(Math.max(1, ...tab.recent.energy_wh_m2.flat().map((v) => v ?? 0)), "Wh/m²");
      note.textContent = "Source: the energy catalog's 15-minute periods. The outlined column is the day selected.";
    }
    if (width) draw(width);
  }

  mount.setup.push(() => {
    show();
    mount.redraws.push(sized(chart, draw));
  });
  return host;
}

/** The brightest 15 minutes over the recent days, or null. */
function recentPeak(tab: SolarTab): { day: string; slot: number; energy: number } | null {
  let best: { day: string; slot: number; energy: number } | null = null;
  tab.recent.energy_wh_m2.forEach((slots, i) => {
    slots.forEach((v, s) => {
      if (v !== null && v > 0 && (best === null || v > best.energy)) best = { day: addDays(tab.recent.first_day, i), slot: s, energy: v };
    });
  });
  return best;
}

/**
 * Render the Solar tab into its panel, replacing what was there.
 *
 * @param panel - The Solar tab's panel.
 * @param snapshot - The dashboard snapshot.
 * @param search - The page query the selected day is read from.
 * @param setSearch - Writes a new page query (the URL, in the browser).
 */
export function renderSolar(
  panel: HTMLElement,
  snapshot: DashboardSnapshot,
  search: () => string = () => location.search,
  setSearch: (query: string) => void = (query) =>
    history.replaceState(history.state, "", `${location.pathname}${query}${location.hash}`),
): void {
  panel.replaceChildren();
  const tab = snapshot.solar_tab;
  if (!tab || !tab.days.energy_kwh_m2.length) {
    panel.append(make("p", "standfirst", "The solar record is not in this snapshot yet."));
    return;
  }
  const state = { day: openingDay(tab, search()) };
  const mount: Mount = { setup: [], redraws: [] };
  let dayPart: { host: HTMLElement; render: () => void } | null = null;

  const pick = (day: string) => {
    if (day < tab.days.first_day || day > lastDay(tab) || day === state.day) return;
    state.day = day;
    setSearch(withSolarDay(search(), day));
    dayPart?.render();
    for (const redraw of mount.redraws) redraw();
  };

  const daily = dailySection(snapshot, tab, state, pick, mount);
  dayPart = daySection(tab, state, pick);
  const heat = heatmapSection(
    snapshot,
    tab,
    state,
    (day) => {
      pick(day);
      const top = dayPart?.host.getBoundingClientRect().top ?? 0;
      if (top < 0) dayPart?.host.scrollIntoView({ block: "start" });
    },
    mount,
  );
  panel.append(lastDaySection(snapshot, tab, mount), daily, dayPart.host, heat);
  // Charts measure their width, so they draw once their sections are placed.
  dayPart.render();
  for (const setup of mount.setup) setup();
}
