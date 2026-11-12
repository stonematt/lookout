/**
 * The almanac grid and rolling-window rain table (#82), read from the
 * snapshot's `almanac` (built by `lookout/snapshot/almanac.py`). They left
 * the Overview with the accordion (#472) for the rain zoom (#475).
 *
 * The grid follows DESIGN.md's "Almanac grid": water years as rows (Oct
 * through Sep, the ticket's season), months as columns, 44px cells with 4px
 * corners and 3px gaps, filled from the rain ramp on a square-root scale
 * with the value in the cell when there is room. ≈ marks a prorated month,
 * the hatch an invalid one, a dashed outline a month before the archive
 * began, and the `now` outline over its ramp fill the current month, which
 * is not yet comparable (hatched only when it has no total so far). Tap, hover, focus or the arrow keys write a month's
 * full sentence to the live readout below the grid. Below 520px of chart
 * width the values and the season column go; the readout keeps both.
 *
 * The table follows DESIGN.md's "Tables": one row per window, one column per
 * year, newest first; an invalid cell is hatched with its reason.
 *
 * History grows (#470): what the record does not cover yet (`shown` false)
 * is not hatched. A grid month it does not cover draws as one before the
 * archive, its season cell stays empty; a table cell it does not cover
 * stays empty, and a row or column with no cell it covers is not drawn. A
 * gap inside a shown period stays hatched, worded without "n/a" ("gap").
 *
 * Offers the two as sections (`almanacGrid`, `rollingRain`), and their
 * renders for the rain zoom, which draws them without their source notes:
 * `renderGrid`, `renderRolling`, and `rollingHasRows` (whether the table
 * has a row to draw).
 */
import "./almanac.css";

import { WATER_MONTHS, dayText, inches } from "../format.js";
import { make } from "../dom.js";
import { hatchGap } from "../gaps.js";
import type { HatchOptions } from "../gaps.js";
import { NAMED_YEARS } from "../rain/text.js";
import type {
  Almanac,
  AlmanacMonth,
  AlmanacYear,
  DashboardSnapshot,
  RollingWindow,
} from "../snapshot.js";
import {
  ROLLING_NAMES,
  cellKind,
  cellText,
  type CellKind,
  gridHeadline,
  monthSentence,
  rampBounds,
  rampStep,
  rollingHeadline,
  seasonSentence,
  seasonSoFar,
  tableValue,
  wettestMonth,
} from "./almanac-text.js";
import { gapReasons, hatchedCount, sourceNote, tileFinding, tileFolds } from "../tiles.js";
import type { SectionOptions } from "../rain/index.js";
import { isShown } from "./history.js";
import type { OverviewSection } from "./index.js";

/** A shown gap's hatch on the Overview: compact, without "n/a" (#470). */
const CELL_GAP: HatchOptions = { compact: true, plain: true };

/**
 * A month's cell kind, with a month the record does not cover yet drawn
 * as one before the archive, never hatched (#470). The current month is
 * judged by its days so far, and keeps any total it has.
 */
function monthKind(month: AlmanacMonth | null): CellKind {
  const kind = cellKind(month);
  if (!month) return kind;
  if (kind !== "current") return isShown(month.status) ? kind : "before";
  const covered = !month.to_date || isShown(month.to_date) || month.to_date_in !== null;
  return covered ? kind : "before";
}

/**
 * The folded source note: the method, with the archive time on its closed
 * line. No lag chip: the header states freshness once (#458). None in the
 * rain zoom (prototype decision 22).
 */
function source(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  options: SectionOptions,
  what: string,
  method: string,
  gaps: string | null,
): void {
  if (options.notes === false) return;
  host.append(tileFolds(sourceNote(snapshot, { what, method, gaps, lag: false })));
}

// ---------- the almanac grid ----------

/** The ramp key: five swatches with where each starts, then the marks. */
function rampKey(max: number): HTMLElement {
  const key = make("div", "alm-key");
  key.dataset.part = "key";
  key.setAttribute("aria-hidden", "true");
  if (max > 0) {
    const ramp = make("span", "alm-key-ramp");
    rampBounds(max).forEach((bound, step) => {
      const item = make("span", "alm-key-step");
      const swatch = make("span", "alm-key-swatch");
      swatch.dataset.step = String(step);
      item.append(swatch, make("span", "alm-key-text", step === 0 ? "0" : cellText(bound)));
      ramp.append(item);
    });
    ramp.append(make("span", "alm-key-unit", "in"));
    key.append(ramp);
  }
  for (const [kind, text] of [
    ["invalid", "not compared"],
    ["before", "before the archive"],
    ["current", "this month, so far"],
  ]) {
    const item = make("span", "alm-key-mark");
    const swatch = make("span", "alm-key-swatch");
    swatch.dataset.kind = kind!;
    item.append(swatch, make("span", "alm-key-text", text));
    key.append(item);
  }
  return key;
}

/** One month's mark: a button whose name is the month's full sentence. */
function monthMark(month: AlmanacMonth | null, max: number): HTMLElement {
  const kind = monthKind(month);
  if (!month) {
    const empty = make("span", "alm-cell");
    empty.dataset.kind = kind;
    return empty;
  }
  const mark = make("button", "alm-cell");
  mark.type = "button";
  mark.tabIndex = -1;
  mark.dataset.kind = kind;
  mark.dataset.month = month.status.key;
  const sentence = monthSentence(month);
  mark.setAttribute("aria-label", sentence);
  if (kind === "valid" || kind === "prorated") {
    mark.dataset.step = String(rampStep(month.rain_in!, max));
    const prefix = kind === "prorated" ? "≈" : "";
    mark.append(make("span", "alm-value", `${prefix}${cellText(month.rain_in!)}`));
    mark.title = sentence;
  } else if (kind === "current") {
    if (month.to_date_in !== null) {
      mark.dataset.step = String(rampStep(month.to_date_in, max));
      mark.append(make("span", "alm-value", cellText(month.to_date_in)));
    }
    mark.title = sentence;
  } else if (kind === "before") {
    mark.title = sentence;
  } else {
    hatchGap(mark, month.status, CELL_GAP);
  }
  return mark;
}

/** A row's season cell: the total, or the running season so far. */
function seasonCell(year: AlmanacYear, almanac: Almanac): HTMLElement {
  const cell = make("td", "alm-season");
  const soFar = year.season.reason === "in_progress" ? seasonSoFar(almanac) : null;
  const sentence = seasonSentence(year, soFar);
  cell.title = sentence;
  if (soFar) {
    cell.dataset.kind = "current";
    if (soFar.rain_in !== null) {
      cell.append(make("span", "alm-value", tableValue(soFar.status, soFar.rain_in)!));
    } else if (isShown(soFar.status)) hatchGap(cell, soFar.status, CELL_GAP);
  } else if (year.season_in === null) {
    if (isShown(year.season)) hatchGap(cell, year.season, CELL_GAP);
  } else {
    cell.append(make("span", "alm-value", tableValue(year.season, year.season_in)!));
  }
  return cell;
}

/** The grid as a table: water years down, months across, season last. */
function gridTable(almanac: Almanac, max: number): HTMLTableElement {
  const table = make("table", "alm-grid");
  const caption = make("caption", "ov-sr", "Rain each month, inches, by water year, newest first");
  const head = make("thead", "");
  const headRow = document.createElement("tr");
  headRow.append(make("td", "alm-corner"));
  for (const name of WATER_MONTHS) {
    const th = make("th", "alm-month");
    th.scope = "col";
    th.append(make("span", "alm-long", name), make("span", "alm-short", name.slice(0, 1)));
    th.setAttribute("aria-label", name);
    headRow.append(th);
  }
  const season = make("th", "alm-season", "Season");
  season.scope = "col";
  headRow.append(season);
  head.append(headRow);

  const body = document.createElement("tbody");
  for (const year of almanac.years) {
    const tr = document.createElement("tr");
    tr.dataset.waterYear = String(year.water_year);
    const th = make("th", "alm-year", String(year.water_year));
    th.scope = "row";
    tr.append(th);
    year.months.forEach((month, column) => {
      const td = make("td", "alm-slot");
      const mark = monthMark(month, max);
      mark.dataset.column = String(column);
      td.append(mark);
      tr.append(td);
    });
    tr.append(seasonCell(year, almanac));
    body.append(tr);
  }
  table.append(caption, head, body);
  return table;
}

/** The mark the readout starts on: the newest month with a value. */
function startingMark(rows: HTMLButtonElement[][]): HTMLButtonElement | undefined {
  for (const row of rows) {
    const found = [...row]
      .reverse()
      .find((m) => m.dataset.kind === "valid" || m.dataset.kind === "prorated");
    if (found) return found;
  }
  const marks = rows.flat();
  return marks.find((m) => m.dataset.kind === "current") ?? marks[0];
}

/**
 * Make the grid's marks one tab stop that tap, hover, focus or the arrow
 * keys move through, each writing its sentence to the readout.
 *
 * @param table - The grid.
 * @param readout - The live readout below it.
 * @param almanac - The snapshot's almanac, for each row's season line.
 */
function wire(table: HTMLTableElement, readout: HTMLElement, almanac: Almanac): void {
  const rows = [...table.querySelectorAll("tbody tr")].map((tr) => [
    ...tr.querySelectorAll<HTMLButtonElement>("button.alm-cell"),
  ]);
  const marks = rows.flat();
  if (!marks.length) {
    readout.textContent = "No month is in the record yet.";
    return;
  }
  const years = new Map(almanac.years.map((y) => [String(y.water_year), y]));
  const soFar = seasonSoFar(almanac);

  const show = (mark: HTMLButtonElement, focus = false) => {
    for (const other of marks) {
      other.tabIndex = other === mark ? 0 : -1;
      other.toggleAttribute("data-selected", other === mark);
    }
    const year = years.get(mark.closest("tr")!.dataset.waterYear!)!;
    readout.replaceChildren(
      make("span", "alm-readout-month", mark.getAttribute("aria-label")!),
      " ",
      make("span", "alm-readout-season", seasonSentence(year, soFar)),
    );
    if (focus) mark.focus();
  };

  const move = (mark: HTMLButtonElement, key: string): HTMLButtonElement | undefined => {
    const r = rows.findIndex((row) => row.includes(mark));
    const row = rows[r]!;
    const column = (m: HTMLButtonElement) => Number(m.dataset.column);
    const at = column(mark);
    const nearest = (target: HTMLButtonElement[]) =>
      target.reduce<HTMLButtonElement | undefined>(
        (best, m) => (!best || Math.abs(column(m) - at) < Math.abs(column(best) - at) ? m : best),
        undefined,
      );
    switch (key) {
      case "ArrowRight":
        return row[row.indexOf(mark) + 1];
      case "ArrowLeft":
        return row[row.indexOf(mark) - 1];
      case "ArrowUp":
        return r > 0 ? nearest(rows[r - 1]!) : undefined;
      case "ArrowDown":
        return r < rows.length - 1 ? nearest(rows[r + 1]!) : undefined;
      case "Home":
        return row[0];
      case "End":
        return row[row.length - 1];
    }
    return undefined;
  };

  for (const mark of marks) {
    mark.addEventListener("click", () => show(mark));
    mark.addEventListener("focus", () => show(mark));
    mark.addEventListener("pointerenter", () => show(mark));
    mark.addEventListener("keydown", (event) => {
      const next = move(mark, event.key);
      if (!next) return;
      event.preventDefault();
      show(next, true);
    });
  }
  show(startingMark(rows)!);
}

/**
 * The almanac grid under its finding, with its source note unless told
 * otherwise (the rain zoom).
 *
 * @param host - The tile, or the zoom's panel.
 * @param snapshot - The dashboard snapshot.
 * @param options - Whether it ends in its source note.
 */
export function renderGrid(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  options: SectionOptions = {},
): void {
  const { almanac } = snapshot;
  const max = wettestMonth(almanac);
  host.append(tileFinding(gridHeadline(almanac)));
  const figure = make("figure", "alm-figure");
  figure.dataset.part = "chart";
  figure.append(rampKey(max));
  const table = gridTable(almanac, max);
  figure.append(table);
  const readout = make("p", "alm-readout");
  readout.dataset.part = "readout";
  readout.setAttribute("aria-live", "polite");
  figure.append(readout);
  host.append(figure);
  wire(table, readout, almanac);
  const hatched = almanac.years
    .flatMap((year) => year.months)
    .filter((month) => month !== null && monthKind(month) === "invalid").length;
  source(
    host,
    snapshot,
    options,
    "daily rain counter, summed by month",
    "Each row is a water year, Oct through Sep, named by the year it ends in. " +
      "The more rain, the stronger a cell's fill, on a square-root scale so drier months still differ; the key reads it. " +
      "The station archive, each day's highest daily rain counter summed by month, with today's live total. " +
      "≈ marks a month scaled up over a few missing days.",
    gapReasons([[hatched, "month", "months"]]),
  );
}

/** The almanac grid, as an Overview section. */
export const almanacGrid: OverviewSection = {
  id: "almanac",
  label: "Rain by month, each water year",
  kicker: "Rain almanac",
  span: 3,
  render: (host, snapshot) => renderGrid(host, snapshot),
};

// ---------- the rolling-window table ----------

/** The id of the season row's water-year heading over column `i`. */
function waterYearId(i: number): string {
  return `alm-roll-wy-${i}`;
}

/**
 * The season row's own heading row, naming each cell's water year, when any
 * differs from its calendar column (an end date in Oct to Dec).
 *
 * @param window - The rolling table's season row.
 * @param columns - The table's calendar-year columns, newest first.
 * @returns The heading row, or null when every season ends in its column's
 *   year.
 */
function waterYearHead(window: RollingWindow, columns: number[]): HTMLTableRowElement | null {
  const years = columns.map((c) => window.years.find((y) => y.year === c));
  if (years.every((y, i) => !y || y.water_year === columns[i])) return null;
  const tr = document.createElement("tr");
  tr.dataset.part = "water-years";
  tr.className = "alm-wy-head";
  const label = make("th", "", "Water year");
  label.scope = "row";
  tr.append(label);
  years.forEach((y, i) => {
    const th = make("th", i === 0 ? "alm-now" : "", y ? String(y.water_year) : "");
    th.scope = "col";
    th.id = waterYearId(i);
    tr.append(th);
  });
  return tr;
}

/**
 * The rolling table's rows and columns the record covers: a row (window) or
 * a column (year) with no cell it covers is not drawn (#470). In a snapshot
 * written before Release A every cell is covered.
 *
 * @param almanac - The snapshot's almanac.
 * @returns The windows to draw and the years, newest first.
 */
function rollingShown(almanac: Almanac): { windows: RollingWindow[]; columns: number[] } {
  const years = (almanac.rolling[0]?.years ?? []).slice(0, NAMED_YEARS + 1).map((y) => y.year);
  const covered = (window: RollingWindow, year: number) =>
    window.years.some((y) => y.year === year && isShown(y.status));
  const windows = almanac.rolling.filter((w) => years.some((year) => covered(w, year)));
  const columns = years.filter((year) => windows.some((w) => covered(w, year)));
  return { windows, columns };
}

/**
 * Whether the rolling table has a row the record covers; no rolling tile is
 * drawn without one (#470).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns True when at least one cell is shown.
 */
export function rollingHasRows(snapshot: DashboardSnapshot): boolean {
  return rollingShown(snapshot.almanac).windows.length > 0;
}

/**
 * The rolling-window table under its finding, with its source note unless
 * told otherwise (the rain zoom).
 *
 * @param host - The tile, or the zoom's panel.
 * @param snapshot - The dashboard snapshot.
 * @param options - Whether it ends in its source note.
 */
export function renderRolling(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  options: SectionOptions = {},
): void {
  const { almanac } = snapshot;
  const { windows, columns } = rollingShown(almanac);
  const today = almanac.rolling[0]?.years[0]?.status.last_day;
  host.append(tileFinding(rollingHeadline(almanac)));
  // The day the windows end, once the standfirst is gone (#458): in the
  // table's caption and the method paragraph.
  const ending = today ? `ending ${dayText(today)}` : "ending today";

  const table = make("table", "alm-table");
  table.dataset.part = "table";
  table.append(
    make("caption", "ov-sr", `Rain over rolling windows ${ending}, inches, by year, newest first`),
  );
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  const corner = make("th", "", "Window");
  corner.scope = "col";
  headRow.append(corner);
  for (const year of columns) {
    const th = make("th", "", String(year));
    th.scope = "col";
    headRow.append(th);
  }
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const window of windows) {
    const tr = document.createElement("tr");
    tr.dataset.window = window.key;
    const th = make("th", "", ROLLING_NAMES[window.key] ?? window.key);
    th.scope = "row";
    tr.append(th);
    // The season is named by its water year; under the calendar columns
    // it is a year ahead from Oct through Dec, so it gets its own heading
    // row then, and its cells are read by it (#156).
    const waterYears = window.key === "water_year" ? waterYearHead(window, columns) : null;
    if (waterYears) {
      th.id = "alm-roll-season";
      body.append(waterYears);
    }
    columns.forEach((columnYear, i) => {
      const year = window.years.find((y) => y.year === columnYear);
      const td = make("td", i === 0 ? "alm-now" : "");
      if (waterYears) td.setAttribute("headers", `${th.id} ${waterYearId(i)}`);
      if (!year || !isShown(year.status)) {
        tr.append(td);
        return;
      }
      const value = tableValue(year.status, year.rain_in);
      if (value === null) hatchGap(td, year.status, CELL_GAP);
      else {
        td.textContent = value;
        hatchGap(td, year.status);
      }
      tr.append(td);
    });
    body.append(tr);
  }
  table.append(head, body);
  const frame = make("div", "alm-table-frame");
  frame.append(table);
  host.append(frame);

  const live = snapshot.current.rain_today_in;
  const statuses = windows
    .flatMap((window) => window.years.filter((y) => columns.includes(y.year)))
    .map((year) => year.status)
    .filter(isShown);
  const hatched = hatchedCount(statuses);
  source(
    host,
    snapshot,
    options,
    "daily rain counter with today's live total",
    `Rain in inches over each window ${ending}, against the same days in earlier years; ` +
      "since Oct 1 is the water year so far. " +
      "The station archive, each day's highest daily rain counter, with today's live total" +
      (live === null ? "." : ` (${inches(live)}).`) +
      " ≈ marks a window scaled up over a few missing days.",
    gapReasons([[hatched, "window", "windows"]]),
  );
}

/** The rolling-window rain table, as an Overview section. */
export const rollingRain: OverviewSection = {
  id: "rolling-rain",
  label: "Rain over the last week, month and season",
  kicker: "Rain, rolling windows",
  // Not drawn while the record covers none of its windows (#470).
  shown: rollingHasRows,
  render: (host, snapshot) => renderRolling(host, snapshot),
};
