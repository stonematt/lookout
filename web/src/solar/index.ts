/**
 * The Solar tab (#85): the last 24 hours against the usual range, daily
 * energy, a picked day's 15-minute energy, and the solar heatmaps, all read
 * from the snapshot's `solar_tab` (built by `lookout/snapshot/solar.py`).
 *
 * Each section is a tile (`web/src/tiles.ts`, spec #332): kicker, finding,
 * standfirst, chart or stats, then the folded source note, whose chip names
 * the archive lag and what the chart hatches. The sections share the
 * selected day, so `solarSections` builds the registry, in the Overview's
 * shape, for one mount. Every value over a day, a month or a window follows
 * its gap status: hatched with the reason when invalid, ≈ when prorated.
 *
 * The selected day lives in the URL (`?solar_day=YYYY-MM-DD`), so a shared
 * link opens on it. Picking a day on the daily bars, either day heatmap or
 * the date field updates the URL in place and redraws what shows it.
 */
import "./solar.css";

import { addDays, dayText, daysBetween, kwhPerM2 } from "../format.js";
import { drawDays } from "../charts/days-switch.js";
import { FALLBACK_WIDTH, drawSized, missingDay } from "../charts/svg.js";
import { make } from "../dom.js";
import { dayStatus, gapElement, gapSentence, hatchGap } from "../gaps.js";
import { WINDOW_NAMES, listText } from "../rain/text.js";
import { staleCatalogNote } from "../stale.js";
import type { DashboardSnapshot, SolarTab } from "../snapshot.js";
import type { TileSection } from "../tiles.js";
import {
  gapReasons,
  hatchedCount,
  notCompared,
  renderTiles,
  sourceNote,
  tileFinding,
  tileFolds,
} from "../tiles.js";
import {
  BAR_DAYS,
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
  ["months", "Months"],
  ["days", "Days"],
  ["slots", "15 minutes"],
] as const;

type View = (typeof VIEWS)[number][0];

/** The tile's finding and standfirst, its first parts after the kicker. */
function heading(
  host: HTMLElement,
  finding: string,
  standfirst: string,
): {
  title: HTMLElement;
  lede: HTMLElement;
} {
  const title = tileFinding(finding);
  const lede = make("p", "standfirst", standfirst);
  lede.dataset.part = "standfirst";
  host.append(title, lede);
  return { title, lede };
}

/** A chart figure with its live readout. */
function figure(name: string): { chart: HTMLElement; readout: HTMLElement } {
  const chart = make("figure", "solar-chart");
  chart.setAttribute("aria-label", `${name}. Use the arrow keys to read each mark.`);
  const readout = make("p", "solar-readout");
  readout.setAttribute("aria-live", "polite");
  return { chart, readout };
}

/**
 * The folded source note at a tile's foot: the archive time on its closed
 * line, the method inside, and a chip when the archive lags or a gap hides
 * part of the chart.
 *
 * @param snapshot - The dashboard snapshot.
 * @param what - What the numbers come from, short.
 * @param sentences - The method paragraph's sentences; nulls are dropped.
 * @param gaps - What the chart hatches, as the chip's reason; null for none.
 * @returns The folded lines' group.
 */
function source(
  snapshot: DashboardSnapshot,
  what: string,
  sentences: (string | null)[],
  gaps: string | null = null,
): HTMLElement {
  const method = sentences.filter(Boolean).join(" ");
  return tileFolds(sourceNote(snapshot, { what, method, gaps }));
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

/** Days a row of day columns hatches: the ones not in the record. */
function hatchedDays(reasons: readonly (SolarTab["days"]["reason"][number] | null)[]): number {
  return reasons.filter((r) => missingDay(r ?? null)).length;
}

/** What the sections of one mount share: the selected day and its pickers. */
interface Mount {
  tab: SolarTab;
  state: { day: string };
  /** Select a day: every view that shows it follows. */
  pick: (day: string) => void;
  /** Redraw the charts that mark the selected day. */
  redraws: (() => void)[];
  /** Redraw the selected day's tile; set when it renders. */
  renderDay: () => void;
  /** The selected day's tile, once rendered. */
  dayHost: HTMLElement | null;
}

/** The 24-hour tile. */
function renderNow(host: HTMLElement, snapshot: DashboardSnapshot, tab: SolarTab): void {
  const trace = tab.last_24h;
  const live = snapshot.current.solar_wm2;
  heading(
    host,
    trace ? traceHeadline(trace, live) : "No solar reading yet",
    "The bold line is solar radiation over the last 24 hours; 1 kWh/m² is an hour of full midday sun. The shaded band is the middle half of the 30 days before at each time of day, the dashed line their median.",
  );
  if (!trace) {
    host.append(
      source(snapshot, "the station's solar sensor", [
        "The station's solar sensor. There is no observation in this snapshot yet.",
      ]),
    );
    return;
  }
  const { chart, readout } = figure("Solar radiation over the last 24 hours");
  host.append(chart, readout);
  sized(chart, (w) =>
    drawTrace(chart, trace, snapshot.archive_newest_ms, w, (t) => (readout.textContent = t)),
  );
  const noUsual = trace.usual.status === "invalid";
  host.append(
    source(
      snapshot,
      "five-minute solar readings",
      [
        "The station's five-minute solar readings, the live reading at the end; energy is the sum of the energy catalog's 15-minute periods.",
        noUsual ? `No usual range: ${gapSentence(trace.usual)}` : null,
        trace.energy_kwh_m2 === null
          ? `The 24-hour energy is not totalled: ${gapSentence(trace.energy)}`
          : null,
      ],
      noUsual ? "usual range not compared" : null,
    ),
  );
}

/** The daily tile: window totals, then 90 days of bars. */
function renderDaily(host: HTMLElement, snapshot: DashboardSnapshot, mount: Mount): void {
  const { tab, state, pick } = mount;
  const standfirst = (days: number) =>
    `Energy over each window ending today, then each day of the last ${days}. Select a day to see its 15 minutes below.`;
  const { lede } = heading(host, dailyHeadline(snapshot), standfirst(BAR_DAYS));
  const strip = make("dl", "solar-stats");
  for (const window of tab.windows) {
    const item = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = WINDOW_NAMES[window.key] ?? window.key;
    const dd = document.createElement("dd");
    dd.textContent = windowValue(window) ?? "";
    item.append(dt, dd);
    strip.append(item);
    hatchGap(dd, window.status, { compact: true });
  }
  // On the four-column grid the strip stands beside the bars (tiles.css).
  const split = make("div", "tile-split");
  const bars = make("div", "tile-split-chart");
  const { chart, readout } = figure("Solar energy each day over the last 90 days");
  bars.append(chart, readout);
  split.append(strip, bars);
  host.append(split);
  // Under 520px of chart width, the last 30 days and the switch to 90.
  mount.redraws.push(
    drawDays(chart, {
      all: BAR_DAYS,
      name: (days) => `Solar energy each day over the last ${days} days`,
      standfirst: lede,
      lede: standfirst,
      draw: (days, w) =>
        drawDailyBars(chart, tab, state.day, w, (t) => (readout.textContent = t), pick, days),
    }),
  );
  const windows = hatchedCount(tab.windows.map((w) => w.status));
  host.append(
    source(
      snapshot,
      "energy catalog, daily sums, kWh/m²",
      [
        "The energy catalog, the station's solar readings summed into 15-minute periods; a day is the sum of its periods, today the energy so far.",
        "A day missing from the record is hatched, never read as a dark day.",
        notShown(tab),
      ],
      gapReasons([
        [windows, "window", "windows"],
        [hatchedDays(tab.days.reason.slice(-BAR_DAYS)), "day", "days"],
      ]),
    ),
  );
}

/** Why each window not shown is left out, one sentence per reason. */
const REASON_WORDS: Record<string, string> = {
  before_archive: "start before the archive began",
  not_archived: "reach past the archive's newest reading",
  in_progress: "are not over yet",
  gap: "miss more than a tenth of their days",
  low_coverage: "have too many days with under 80% of their readings",
};

function notShown(tab: SolarTab): string | null {
  const byReason = new Map<string, string[]>();
  for (const w of tab.windows) {
    if (w.energy_kwh_m2 !== null) continue;
    const reason = w.status.reason ?? "gap";
    byReason.set(reason, [
      ...(byReason.get(reason) ?? []),
      (WINDOW_NAMES[w.key] ?? w.key).toLowerCase(),
    ]);
  }
  const parts = [...byReason].map(
    ([reason, names]) => `${listText(names)} ${REASON_WORDS[reason] ?? "cannot be compared"}`,
  );
  return parts.length ? `Not shown: ${parts.join("; ")}.` : null;
}

/** The selected day's tile, redrawn in place on every pick. */
function renderDay(host: HTMLElement, snapshot: DashboardSnapshot, mount: Mount): void {
  const { tab, state, pick } = mount;
  mount.dayHost = host;
  host.classList.add("solar-day");
  const { title } = heading(
    host,
    "",
    "Each bar is 15 minutes of the selected day, 4 am to 10 pm. The marked bar is its brightest.",
  );

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
  const today = make("button", "solar-button", "Today");
  today.type = "button";
  today.addEventListener("click", () => pick(lastDay(tab)));
  controls.append(prev, field, next, today);
  prev.addEventListener("click", () => pick(addDays(state.day, -1)));
  next.addEventListener("click", () => pick(addDays(state.day, 1)));
  input.addEventListener("change", () => {
    if (input.value && input.value >= input.min && input.value <= input.max) pick(input.value);
  });

  const body = make("div", "solar-day-body");
  let folds = source(snapshot, "energy catalog, 15-minute periods", []);
  host.append(controls, body, folds);
  const note = (what: string, sentences: (string | null)[], gaps: string | null = null) => {
    const next = source(snapshot, what, sentences, gaps);
    folds.replaceWith(next);
    folds = next;
  };

  function render(): void {
    const day = readDay(tab, state.day);
    title.textContent = dayHeadline(day);
    input.value = day.day;
    prev.disabled = day.day <= tab.days.first_day;
    next.disabled = day.day >= lastDay(tab);
    if (day.reason !== null || day.energy === null) {
      const status = dayStatus(day.day, day.reason ?? "gap");
      body.replaceChildren(gapElement(status));
      note(
        "energy catalog, daily sums",
        [`The energy catalog's daily sums. ${gapSentence(status)}`],
        notCompared(1, "day", "days"),
      );
      return;
    }
    if (day.slots === null) {
      body.replaceChildren(
        make(
          "p",
          "solar-older",
          `${dayText(day.day, true)} gave ${kwhPerM2(day.energy)}. Its 15-minute energy is kept for the last 90 days only.`,
        ),
      );
      note("energy catalog, daily sums", [
        "The energy catalog's daily sum; 15-minute periods are kept for the last 90 days.",
      ]);
      return;
    }
    const { chart, readout } = figure(`Solar energy each 15 minutes on ${dayText(day.day, true)}`);
    body.replaceChildren(chart, readout);
    const slots = day.slots;
    drawSized(chart, (w) => drawDaySlots(chart, day, w, (t) => (readout.textContent = t)));
    const outside = slots
      .slice(0, FIRST_SLOT)
      .concat(slots.slice(LAST_SLOT + 1))
      .some((v) => v !== null && v > 0);
    note("energy catalog, 15-minute periods", [
      "The energy catalog's 15-minute periods, in Wh per m².",
      day.isToday ? "Today's bars stop at the archive's newest period." : null,
      outside
        ? `Some energy fell before ${slotText(FIRST_SLOT)} or after ${slotText(LAST_SLOT + 1)}; the total counts it.`
        : null,
    ]);
  }

  mount.renderDay = render;
  render();
}

/** The heatmap tile, with a control to switch between its views. */
function renderHeatmaps(host: HTMLElement, snapshot: DashboardSnapshot, mount: Mount): void {
  const { tab, state } = mount;
  // Picking a day here scrolls its tile into view when it is above.
  const pick = (day: string) => {
    mount.pick(day);
    const top = mount.dayHost?.getBoundingClientRect().top ?? 0;
    if (top < 0) mount.dayHost?.scrollIntoView({ block: "start" });
  };
  const { title, lede: standfirst } = heading(host, "", "");
  const control = make("div", "solar-seg");
  control.setAttribute("role", "group");
  control.setAttribute("aria-label", "Heatmap");
  const { chart, readout } = figure("Solar heatmap");
  const key = make("ul", "solar-key");
  key.setAttribute("aria-label", "Key");
  let folds = source(snapshot, "energy catalog, daily sums", []);
  host.append(control, chart, key, readout, folds);
  const note = (what: string, sentence: string, gaps: string | null) => {
    const next = source(snapshot, what, [sentence], gaps);
    folds.replaceWith(next);
    folds = next;
  };

  // On a phone a day cell is too small to tap; the year-by-month grid is not.
  let view: View = (document.documentElement.clientWidth || 1024) < 520 ? "months" : "days";
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
    gap.append(
      make("span", "solar-swatch gap-hatch"),
      document.createTextNode("not in the record"),
    );
    key.append(gap);
  }

  let width = 0;
  function draw(w: number): void {
    width = w;
    const write = (t: string) => (readout.textContent = t);
    if (view === "months") drawYearMonth(chart, tab, w, write);
    else if (view === "days") drawMonthDay(chart, tab, state.day, w, write, pick);
    else drawDayGrid(chart, tab, state.day, w, write, pick);
  }

  function show(): void {
    for (const [id, button] of buttons) button.setAttribute("aria-pressed", String(id === view));
    readout.textContent = "";
    if (view === "months") {
      const best = brightestMonth(tab);
      title.textContent = best
        ? `${monthText(best.key)} is the brightest month on record, at ${kwhPerM2(best.energy)}`
        : "Solar energy each month";
      standfirst.textContent =
        "One row per year, one column per month; the deeper the shade, the more energy. The month under way is not compared yet.";
      fillKey(Math.max(1, ...tab.months.map((m) => m.energy_kwh_m2 ?? 0)), "kWh/m²");
      // The grid hatches a month with no value, the one under way aside.
      const hatched = tab.months.filter(
        (m) =>
          m.energy_kwh_m2 === null &&
          m.status.reason !== "before_archive" &&
          m.status.reason !== "in_progress",
      ).length;
      note(
        "energy catalog, daily sums by month",
        "The energy catalog's daily sums. ≈ marks a month scaled up over a few missing days; a hatched month has too many.",
        notCompared(hatched, "month", "months"),
      );
    } else if (view === "days") {
      const best = brightestDay(tab);
      title.textContent = best
        ? `The brightest day on record was ${dayText(best.day, true)}, at ${kwhPerM2(best.energy)}`
        : "Solar energy each day";
      standfirst.textContent =
        "One row per month, newest at the top, one column per day; select a day to see its 15 minutes.";
      fillKey(Math.max(1, ...tab.days.energy_kwh_m2.map((v) => v ?? 0)), "kWh/m²");
      note(
        "energy catalog, daily sums",
        "The energy catalog's daily sums. The outlined day is the one selected.",
        notCompared(hatchedDays(tab.days.reason), "day", "days"),
      );
    } else {
      const peak = recentPeak(tab);
      title.textContent = peak
        ? `The brightest 15 minutes of the last 90 days: ${dayText(peak.day, true)}, ${slotText(peak.slot)}, ${wh(peak.energy)}`
        : "Solar energy each 15 minutes over the last 90 days";
      standfirst.textContent =
        "One column per day of the last 90, one row per 15 minutes from 4 am at the top to 10 pm; select a day to see it below.";
      fillKey(Math.max(1, ...tab.recent.energy_wh_m2.flat().map((v) => v ?? 0)), "Wh/m²");
      const offset = daysBetween(tab.days.first_day, tab.recent.first_day);
      const reasons = tab.recent.energy_wh_m2.map((_, i) => tab.days.reason[offset + i] ?? null);
      note(
        "energy catalog, 15-minute periods",
        "The energy catalog's 15-minute periods. The outlined column is the day selected.",
        notCompared(hatchedDays(reasons), "day", "days"),
      );
    }
    if (width) draw(width);
  }

  show();
  mount.redraws.push(sized(chart, draw));
}

/** The brightest 15 minutes over the recent days, or null. */
function recentPeak(tab: SolarTab): { day: string; slot: number; energy: number } | null {
  let best: { day: string; slot: number; energy: number } | null = null;
  tab.recent.energy_wh_m2.forEach((slots, i) => {
    slots.forEach((v, s) => {
      if (v !== null && v > 0 && (best === null || v > best.energy))
        best = { day: addDays(tab.recent.first_day, i), slot: s, energy: v };
    });
  });
  return best;
}

/**
 * The Solar tab's sections for one mount, in the desktop board's order:
 * sunlight now across the grid (the phone's fixed top), the daily bars
 * across the grid, then the selected day beside the heatmaps.
 *
 * @param mount - The selected day and its pickers, shared by the sections.
 * @returns The registry.
 */
function solarSections(mount: Mount): TileSection[] {
  return [
    {
      id: "now",
      label: "Solar radiation over the last 24 hours",
      kicker: "Sunlight now",
      span: 4,
      phoneTop: true,
      render: (host, snapshot) => renderNow(host, snapshot, mount.tab),
    },
    {
      id: "daily",
      label: "Solar energy over each window, and each day of the last 90",
      kicker: "Sun each day",
      span: 4,
      render: (host, snapshot) => renderDaily(host, snapshot, mount),
    },
    {
      id: "day",
      label: "The selected day, by the quarter hour",
      kicker: "One day, by the quarter hour",
      span: 2,
      render: (host, snapshot) => renderDay(host, snapshot, mount),
    },
    {
      id: "heatmaps",
      label: "Solar heatmaps",
      kicker: "Solar almanac",
      span: 2,
      render: (host, snapshot) => renderHeatmaps(host, snapshot, mount),
    },
  ];
}

/**
 * Render the Solar tab into its panel as a grid of tiles, replacing what
 * was there.
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
  const stale = staleCatalogNote(snapshot, "energy");
  const staleNote = stale ? make("p", "solar-source solar-stale", stale) : null;
  staleNote?.setAttribute("role", "status");
  const tab = snapshot.solar_tab;
  if (!tab || !tab.days.energy_kwh_m2.length) {
    if (staleNote) panel.append(staleNote);
    panel.append(make("p", "standfirst", "The solar record is not in this snapshot yet."));
    return;
  }
  const mount: Mount = {
    tab,
    state: { day: openingDay(tab, search()) },
    pick: (day) => {
      if (day < tab.days.first_day || day > lastDay(tab) || day === mount.state.day) return;
      mount.state.day = day;
      setSearch(withSolarDay(search(), day));
      mount.renderDay();
      for (const redraw of mount.redraws) redraw();
    },
    redraws: [],
    renderDay: () => {},
    dayHost: null,
  };
  const grid = renderTiles(panel, snapshot, solarSections(mount));
  // The catalog note stands above the tiles: it holds for every one of them.
  if (staleNote) grid.before(staleNote);
}
