/**
 * The Overview's almanac grid and rolling-window rain table (#82), read from
 * the snapshot's `almanac` (built by `lookout/snapshot/almanac.py`).
 *
 * The grid follows DESIGN.md's "Almanac grid": water years as rows (Oct
 * through Sep, the ticket's season), months as columns, 44px cells with 4px
 * corners and 3px gaps, filled from the rain ramp on a square-root scale
 * with the value in the cell when there is room. ≈ marks a prorated month,
 * the hatch an invalid one, a dashed outline a month before the archive
 * began, and the `note` outline over a light hatch the current month, which
 * is not yet comparable. Tap, hover, focus or the arrow keys write a month's
 * full sentence to the live readout below the grid. Below 520px of chart
 * width the values and the season column go; the readout keeps both.
 *
 * The table follows DESIGN.md's "Tables": one row per window, one column per
 * year, newest first; an invalid cell is hatched and reads "n/a".
 */
import "./almanac.css";

import { formatStamp } from "../format.js";
import { hatchGap } from "../gaps.js";
import { NAMED_YEARS, dayText, inches } from "../rain/text.js";
import type { Almanac, AlmanacMonth, AlmanacYear, DashboardSnapshot } from "../snapshot.js";
import {
  ROLLING_NAMES,
  WATER_MONTHS,
  cellKind,
  cellText,
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
import type { OverviewSection } from "./index.js";

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

/** Headline and standfirst, the section's first two parts. */
function heading(host: HTMLElement, headline: string, standfirst: string): void {
  const title = make("h2", "alm-headline", headline);
  title.dataset.part = "headline";
  const lede = make("p", "standfirst", standfirst);
  lede.dataset.part = "standfirst";
  host.append(title, lede);
}

/** The source line: method first, then the archive lag. */
function source(host: HTMLElement, snapshot: DashboardSnapshot, method: string): void {
  const lag =
    snapshot.archive_newest_ms === null
      ? null
      : `Archived through ${formatStamp(snapshot.archive_newest_ms)}.`;
  const line = make("p", "ov-source", [method, lag].filter(Boolean).join(" "));
  line.dataset.part = "source";
  host.append(line);
}

// ---------- the almanac grid ----------

/** The ramp key: five swatches with where each starts, then the marks. */
function rampKey(max: number): HTMLElement {
  const key = make("div", "alm-key");
  key.dataset.part = "key";
  key.setAttribute("aria-hidden", "true");
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
  const kind = cellKind(month);
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
    if (month.to_date_in !== null) mark.append(make("span", "alm-value", cellText(month.to_date_in)));
    mark.title = sentence;
  } else if (kind === "before") {
    mark.title = sentence;
  } else {
    hatchGap(mark, month.status, { compact: true });
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
    if (soFar.rain_in === null) hatchGap(cell, soFar.status, { compact: true });
    else cell.append(make("span", "alm-value", tableValue(soFar.status, soFar.rain_in)!));
  } else if (year.season_in === null) {
    hatchGap(cell, year.season, { compact: true });
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
    const found = [...row].reverse().find((m) => m.dataset.kind === "valid" || m.dataset.kind === "prorated");
    if (found) return found;
  }
  return rows.flat()[0];
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
  const rows = [...table.querySelectorAll("tbody tr")].map((tr) =>
    [...tr.querySelectorAll<HTMLButtonElement>("button.alm-cell")],
  );
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

function renderGrid(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const { almanac } = snapshot;
  const max = wettestMonth(almanac);
  heading(
    host,
    gridHeadline(almanac),
    "Each row is a water year, Oct through Sep, named by the year it ends in. " +
      "The more rain, the stronger the blue, on a square-root scale so drier months still differ.",
  );
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
  source(
    host,
    snapshot,
    "Source: the station archive, each day's highest daily rain counter summed by month, with today's live total. " +
      "≈ marks a month scaled up over a few missing days.",
  );
}

/** The almanac grid, as an Overview section. */
export const almanacGrid: OverviewSection = {
  id: "almanac",
  label: "Rain by month, each water year",
  render: renderGrid,
};

// ---------- the rolling-window table ----------

function renderRolling(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const { almanac } = snapshot;
  const columns = (almanac.rolling[0]?.years ?? []).slice(0, NAMED_YEARS + 1).map((y) => y.year);
  const today = almanac.rolling[0]?.years[0]?.status.last_day;
  heading(
    host,
    rollingHeadline(almanac),
    today
      ? `Rain in inches over each window ending ${dayText(today)}, against the same days in earlier years. ` +
          "Since Oct 1 is the water year so far."
      : "Rain in inches over each window ending today, against the same days in earlier years.",
  );

  const table = make("table", "alm-table");
  table.dataset.part = "table";
  table.append(make("caption", "ov-sr", "Rain over rolling windows, inches, by year, newest first"));
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
  for (const window of almanac.rolling) {
    const tr = document.createElement("tr");
    tr.dataset.window = window.key;
    const th = make("th", "", ROLLING_NAMES[window.key] ?? window.key);
    th.scope = "row";
    tr.append(th);
    columns.forEach((columnYear, i) => {
      const year = window.years.find((y) => y.year === columnYear);
      const td = make("td", i === 0 ? "alm-now" : "");
      if (!year) {
        tr.append(td);
        return;
      }
      const value = tableValue(year.status, year.rain_in);
      if (value === null) hatchGap(td, year.status, { compact: true });
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
  source(
    host,
    snapshot,
    "Source: the station archive, each day's highest daily rain counter, with today's live total" +
      (live === null ? "." : ` (${inches(live)}).`) +
      " ≈ marks a window scaled up over a few missing days.",
  );
}

/** The rolling-window rain table, as an Overview section. */
export const rollingRain: OverviewSection = {
  id: "rolling-rain",
  label: "Rain over the last week, month and season",
  render: renderRolling,
};
