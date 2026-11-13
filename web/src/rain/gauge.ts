/**
 * The rain gauge (#619): rain over each window as vertical bars, in the rain
 * zoom's "Rain over each window" panel, next to the water year. It replaces
 * the window strip there; the Rain tab keeps its strip.
 *
 * Three panels, each on one shared axis topped by the largest value drawn
 * in it, with 0 and that top printed: Today and yesterday, Last 7 and 30
 * days, Last 90 and 365 days. This year's total (`RainWindow.total_in`) is a
 * bar in the rain colour, its total on top. For 7, 30, 90 and 365 days the
 * same days in earlier years (`almanac.rolling`) stand left of it as thin
 * grey bars, oldest first, each year under its bar and this year's in bold;
 * an earlier year with no total keeps its label over a hatched column, so a
 * missing year never reads as a dry one. Today and Yesterday, and every
 * window of a station with no earlier year, are plain bars. A window whose
 * status is not ok (no total, or not shown under the history-grows rule) is
 * left out, and a panel with none left is too. When neither today nor
 * yesterday had rain, their panel folds to one line.
 *
 * Across from 560 px of width, the Today and yesterday panel a third
 * narrower than the others, and no panel wider than its share of a full
 * row, so one drawn alone keeps its bars close; stacked below that. Window
 * names are the short ones ("7 days"), so nothing breaks.
 *
 * The title leads with the window it speaks to, as that panel's heading
 * names it ("Last 90 days: ..."), and states numbers the snapshot holds,
 * with no comparison of its own: deciding "wetter" or "drier" is a rule,
 * and rules are Python's (ADR-0003).
 *
 * Offers `gaugeView`, `gaugeShown`, `gaugeHeadline` and `renderGauge`.
 */
import "./gauge.css";

import { canvas, drawSized } from "../charts/svg.js";
import { make, svgEl } from "../dom.js";
import { inches } from "../format.js";
import { gapValue } from "../gaps.js";
import { isShown } from "../overview/history.js";
import type { DashboardSnapshot, GapStatus } from "../snapshot.js";
import { tileFinding } from "../tiles.js";
import { NAMED_YEARS, WINDOW_NAMES, listText } from "./text.js";

/** One earlier year of a window: null when the record holds no total for it. */
export interface GaugeYear {
  year: number;
  status: GapStatus;
  value: number | null;
}

/** One window's bar, and its earlier years. */
export interface GaugeColumn {
  key: string;
  status: GapStatus;
  total: number;
  /** The year the window ends in. */
  year: number;
  /** Oldest first; empty for Today, Yesterday and a station with no earlier year. */
  prior: GaugeYear[];
}

/** What the gauge draws: its columns, and whether today and yesterday folded to a line. */
export interface GaugeView {
  columns: GaugeColumn[];
  /** Today and yesterday were both drawn at 0, so their panel is one line instead. */
  quiet: boolean;
}

/** The panels, each on one axis, and the windows each holds. */
const PANELS: readonly { keys: readonly string[]; weight: number }[] = [
  { keys: ["today", "yesterday"], weight: 2 },
  { keys: ["7d", "30d"], weight: 3 },
  { keys: ["90d", "365d"], weight: 3 },
];

/** The windows with earlier years in `almanac.rolling`. */
const ROLLING = new Set(["7d", "30d", "90d", "365d"]);
/** The day windows, which share the first panel. */
const DAYS = new Set(["today", "yesterday"]);

/**
 * The windows the gauge draws, in panel order, with their earlier years.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns One column per window whose status is ok, less today and
 *   yesterday when both are 0 (`quiet`).
 */
export function gaugeView(snapshot: DashboardSnapshot): GaugeView {
  const rolling = snapshot.almanac?.rolling ?? [];
  const out: GaugeColumn[] = [];
  for (const key of PANELS.flatMap((p) => p.keys)) {
    const window = snapshot.rain_tab.windows.find((w) => w.key === key);
    if (!window || window.total_in === null || !isShown(window.status)) continue;
    const years = ROLLING.has(key) ? (rolling.find((r) => r.key === key)?.years ?? []) : [];
    const prior = years
      .slice(1)
      .map((y) => ({ year: y.year, status: y.status, value: y.rain_in }))
      .sort((a, b) => a.year - b.year);
    out.push({
      key,
      status: window.status,
      total: window.total_in,
      // The window's own last day is station-local (ADR-0002), unlike the browser's clock.
      year: years[0]?.year ?? Number(window.status.last_day.slice(0, 4)),
      prior,
    });
  }
  const days = out.filter((c) => DAYS.has(c.key));
  const quiet = days.length > 0 && days.every((c) => c.total === 0);
  return { columns: quiet ? out.filter((c) => !DAYS.has(c.key)) : out, quiet };
}

/**
 * Whether the gauge has a window to draw.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns False when no window's status is ok.
 */
export function gaugeShown(snapshot: DashboardSnapshot): boolean {
  const view = gaugeView(snapshot);
  return view.columns.length > 0 || view.quiet;
}

/** A total as its status shows it: "0.44 in", "≈ 0.44 in". */
function shown(status: GapStatus, value: number, unit = true): string {
  const text = inches(value, { unit });
  return gapValue(status, text) ?? text;
}

/** A rolling window's name as its panel's heading reads it: "Last 90 days". */
const lastName = (key: string): string => `Last ${WINDOW_NAMES[key] ?? key}`;

/**
 * The gauge's finding: the longest window drawn, named first as its panel
 * names it, with its earlier years' totals, newest first. With only today
 * and yesterday drawn, both days.
 *
 * @param view - What the gauge draws.
 * @returns E.g. "Last 365 days: 38.2 in, against 41.0 in (2025) and 30.4 in
 *   (2024) over the same days"; "Today 0.30 in, yesterday 0.24 in".
 */
export function gaugeHeadline(view: GaugeView): string {
  const last = view.columns.at(-1);
  if (!last) return view.quiet ? "No rain today or yesterday" : "Rain over each window";
  if (DAYS.has(last.key)) {
    const days = view.columns.map((c) => `${c.key} ${shown(c.status, c.total)}`).join(", ");
    return days.charAt(0).toUpperCase() + days.slice(1);
  }
  const lead = `${lastName(last.key)}: ${shown(last.status, last.total)}`;
  const earlier = last.prior
    .filter((y) => y.value !== null)
    .reverse()
    .slice(0, NAMED_YEARS)
    .map((y) => `${shown(y.status, y.value!)} (${y.year})`);
  return earlier.length ? `${lead}, against ${listText(earlier)} over the same days` : lead;
}

/** A panel's heading, naming only the windows drawn in it. */
function heading(columns: GaugeColumn[]): string {
  const keys = columns.map((c) => c.key);
  if (keys[0] === "today" || keys[0] === "yesterday") {
    return keys.length === 2 ? "Today and yesterday" : (WINDOW_NAMES[keys[0]] ?? keys[0]);
  }
  if (keys.length === 1) return lastName(keys[0]!);
  return `Last ${keys.map((k) => k.replace("d", "")).join(" and ")} days`;
}

/** An axis number: short, no trailing zeros. */
const tick = (v: number): string => String(+v.toFixed(v >= 10 ? 0 : v >= 1 ? 1 : 2));
/** A year as its label: '25. */
const yy = (year: number): string => `'${String(year).slice(2)}`;

/** The rows of one panel. */
const HEAD = 20;
const VALUE = 18;
const PLOT = 76;
const YEAR_ROW = 16;
const NAME_ROW = 18;
const PANEL_H = HEAD + VALUE + PLOT + YEAR_ROW + NAME_ROW + 6;
/** The axis gutter, the gap between panels across, and between them stacked. */
const AXIS = 38;
const GAP = 20;
const STACK_GAP = 12;
/** From this width the panels sit across. */
const ACROSS = 560;
/** The smallest axis top: a dry panel still has a scale. */
const MIN_TOP = 0.1;
/** The widest a panel grows, per unit of its weight. */
const MAX_PER_WEIGHT = 150;

/** Append a text. */
function text(
  parent: SVGElement,
  x: number,
  y: number,
  value: string,
  cls: string,
  anchor = "middle",
): void {
  parent.append(
    svgEl("text", { x: +x.toFixed(1), y: +y.toFixed(1), class: cls, "text-anchor": anchor }, value),
  );
}

/** A bar from the baseline up to a value; at least a hairline, so a 0 still shows. */
function bar(x: number, width: number, height: number, base: number, cls: string): SVGElement {
  const h = Math.max(1, height);
  return svgEl("rect", {
    x: +x.toFixed(1),
    y: +(base - h).toFixed(1),
    width: +width.toFixed(1),
    height: +h.toFixed(1),
    class: cls,
  });
}

/** One panel at (px, py), pw wide, on one axis. */
function panel(
  svg: SVGSVGElement,
  hatch: string,
  px: number,
  py: number,
  pw: number,
  cols: GaugeColumn[],
): void {
  const g = svgEl("g", { class: "gauge-panel" });
  const top = Math.max(
    MIN_TOP,
    ...cols.flatMap((c) => [c.total, ...c.prior.map((p) => p.value ?? 0)]),
  );
  const plotTop = py + HEAD + VALUE;
  const base = plotTop + PLOT;
  const h = (v: number) => (v / top) * PLOT;
  text(g, px, py + 13, heading(cols), "gauge-head", "start");
  const ax = px + AXIS - 4;
  g.append(svgEl("line", { x1: ax, x2: ax, y1: plotTop, y2: base, class: "gauge-axis" }));
  g.append(svgEl("line", { x1: ax, x2: px + pw, y1: base, y2: base, class: "gauge-axis" }));
  text(g, ax - 4, plotTop + 4, tick(top), "gauge-tick", "end");
  text(g, ax - 4, base + 4, "0", "gauge-tick", "end");
  const gw = (pw - AXIS) / cols.length;
  cols.forEach((c, i) => {
    const col = svgEl("g", { "data-window": c.key });
    const gx = px + AXIS + gw * i;
    const total = shown(c.status, c.total, false);
    if (!c.prior.length) {
      const bw = Math.min(40, gw * 0.5);
      const bx = gx + (gw - bw) / 2;
      col.append(bar(bx, bw, h(c.total), base, "gauge-bar"));
      text(col, bx + bw / 2, base - Math.max(1, h(c.total)) - 5, total, "gauge-total");
      text(col, gx + gw / 2, base + 16, WINDOW_NAMES[c.key] ?? c.key, "gauge-name");
    } else {
      const n = c.prior.length;
      const unit = Math.min(22, (gw - 8) / (n + 2));
      const thin = unit - 3;
      const full = unit * 2 - 3;
      let x = gx + (gw - unit * (n + 2)) / 2;
      for (const p of c.prior) {
        // A year with no total is hatched the plot's height: not known, not 0.
        const rect =
          p.value === null
            ? svgEl("rect", {
                x: +x.toFixed(1),
                y: plotTop,
                width: +thin.toFixed(1),
                height: PLOT,
                fill: hatch,
                class: "gauge-missing",
              })
            : bar(x, thin, h(p.value), base, "gauge-prior");
        rect.dataset.year = String(p.year);
        col.append(rect);
        text(col, x + thin / 2, base + 14, yy(p.year), "gauge-year");
        x += unit;
      }
      col.append(bar(x, full, h(c.total), base, "gauge-bar"));
      text(col, x + full / 2, base - Math.max(1, h(c.total)) - 5, total, "gauge-total");
      text(col, x + full / 2, base + 14, yy(c.year), "gauge-year gauge-this");
      text(col, gx + gw / 2, base + YEAR_ROW + 15, WINDOW_NAMES[c.key] ?? c.key, "gauge-name");
    }
    g.append(col);
  });
  svg.append(g);
}

/** Draw the panels at a width: across from `ACROSS`, stacked under it. */
function draw(host: HTMLElement, columns: GaugeColumn[], width: number): void {
  const panels = PANELS.map((p) => ({
    weight: p.weight,
    cols: columns.filter((c) => p.keys.includes(c.key)),
  })).filter((p) => p.cols.length);
  const across = width >= ACROSS;
  const n = panels.length;
  const height = across ? PANEL_H : n * PANEL_H + (n - 1) * STACK_GAP;
  const label = `Rain over each window: ${columns
    .map((c) => `${WINDOW_NAMES[c.key] ?? c.key} ${shown(c.status, c.total)}`)
    .join(", ")}`;
  const { svg, hatch } = canvas(width, height, label, "gauge-svg");
  // Weights over the full row of panels, so a panel drawn alone keeps its share.
  const weights = PANELS.reduce((sum, p) => sum + p.weight, 0);
  const room = width - GAP * (PANELS.length - 1);
  let px = 0;
  panels.forEach((p, i) => {
    const cap = p.weight * MAX_PER_WEIGHT;
    if (across) {
      const pw = Math.min(cap, (room * p.weight) / weights);
      panel(svg, hatch, px, 0, pw, p.cols);
      px += pw + GAP;
    } else {
      panel(svg, hatch, 0, i * (PANEL_H + STACK_GAP), Math.min(cap, width), p.cols);
    }
  });
  host.replaceChildren(svg);
}

/**
 * Draw the gauge into a zoom panel: its finding, the chart, and the key when
 * earlier years are drawn.
 *
 * @param host - The panel.
 * @param snapshot - The dashboard snapshot.
 */
export function renderGauge(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const view = gaugeView(snapshot);
  const { columns } = view;
  host.append(tileFinding(gaugeHeadline(view), { zoom: true }));
  if (view.quiet) {
    const quiet = make("p", "gauge-quiet", "No rain today or yesterday.");
    quiet.dataset.part = "quiet";
    host.append(quiet);
  }
  if (!columns.length) return;
  const fig = make("figure", "gauge-fig");
  host.append(fig);
  if (columns.some((c) => c.prior.length)) {
    const missing = columns.some((c) => c.prior.some((p) => p.value === null));
    const key = make(
      "p",
      "gauge-key",
      `Grey: the same window in earlier years.${missing ? " Hatched: no total that year." : ""}`,
    );
    key.dataset.part = "key";
    host.append(key);
  }
  drawSized(fig, (width) => draw(fig, columns, width));
}
