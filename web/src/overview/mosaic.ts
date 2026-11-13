/**
 * The mosaic (#472, spec #460 stories 18–22, 28, 52, 56): the Overview's
 * tiles under the feature, each a title, a number, a short delta and a mini
 * chart, and each one link to its zoom at `#overview/<slug>`.
 *
 * A tile is a definition, not markup: `MosaicTile` names its zoom slug, its
 * record-depth key, its title, the name of the zoom its link opens, and reads its number
 * and delta and draws its mini chart from the snapshot. `mosaicSection`
 * turns a definition into an Overview section; the tile definitions live in
 * `mosaic-tiles.ts`, in the mosaic's fixed order.
 *
 * The tile's markup, from `mosaicSection`:
 *
 * - the tile (`section.tile[data-mosaic]`, id `overview-<slug>`), with no
 *   accessible name of its own;
 * - inside it one link, `a.mosaic-link[href="#overview/<slug>"]`, holding the
 *   title (`h2`, the tile's heading), the number and its unit, the delta, the
 *   mini chart (`aria-hidden`, with its "usual range" legend swatch when it
 *   draws a band under bars) and, for screen readers only, "Open the …
 *   view". The link's name is its own text, so a screen reader hears the
 *   finding once and no `aria-label` repeats it.
 *
 * A tile whose record-depth entry (`record_depth.tiles`, #463) says it is not
 * shown is not drawn, and the later tiles move up. A snapshot without record
 * depth (N−1, ADR-0008) draws every tile.
 *
 * Every value is the snapshot's: a tile words, formats and draws; it never
 * compares or decides.
 *
 * A tile with rows (Inside, #473) draws one row per entry under its title in
 * place of one number: a name, a reading with an aside, and an
 * `aria-hidden` line. A tile whose own `shown` says the snapshot holds
 * nothing for it is not drawn either.
 *
 * Offers `MosaicTile`, `MosaicRow`, `TileReading`, `mosaicSection`,
 * `miniChart` (the mini chart's SVG root) and `ROW_LINE_HEIGHT`.
 */
import "./mosaic.css";

import { drawSized } from "../charts/svg.js";
import { spoken, svgEl } from "../dom.js";
import type { DashboardSnapshot } from "../snapshot.js";
import type { TileSource } from "../tiles.js";
import type { OverviewSection } from "./index.js";

/** A mini chart's height, the same on every tile so a row lines up. */
export const MINI_HEIGHT = 64;

/** What a tile reads from the snapshot: its number and its delta. */
export interface TileReading {
  /** The number, formatted ("72.0°", "1 d 14 h"); null when there is none. */
  value: string | null;
  /** A word unit set smaller after the number ("in"); none by default. */
  unit?: string | null;
  /** The short delta under the number; null for none. */
  delta: string | null;
  /**
   * Words for screen readers alone, after the delta: what the mini chart
   * shows that the number and delta do not ("sunrise 7:15 am, ...").
   */
  spoken?: string;
}

/** One mosaic tile, declared once; `mosaicSection` draws it. */
export interface MosaicTile {
  /**
   * The zoom's slug: the link goes to `#overview/<slug>`, the tile's
   * `data-section` is the slug and its id `overview-<slug>`.
   */
  slug: string;
  /** The tile's key in `record_depth.tiles` ("last_10_days"). */
  depth: string;
  /** The tile's title, its heading: "Rain since Oct 1". */
  title: string;
  /** The zoom's name, in the link's "Open the <zoomName> view": "temperature". */
  zoomName: string;
  /** The number and the delta. */
  read(snapshot: DashboardSnapshot): TileReading;
  /**
   * The mini chart at a width, a crop of the zoom's lead chart; null when
   * there is nothing to draw. It is `aria-hidden`: the number and delta
   * carry the finding.
   */
  chart?(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null;
  /**
   * Whether the chart draws a usual band under its bars for this snapshot:
   * it then carries the "usual range" legend swatch above the chart (spec
   * #460 Legends), and none when no day has a band to draw.
   */
  bandUnderBars?(snapshot: DashboardSnapshot): boolean;
  /**
   * The tile's entry in the Overview's sources footer (`sources.ts`): its
   * short line (source and comparison, at most 12 words) and its method
   * paragraph for the footer's disclosure. The tile itself folds no source.
   */
  source(snapshot: DashboardSnapshot): TileSource;
  /**
   * Whether the snapshot holds what the tile draws, beside its record
   * depth: Inside without an indoor sensor, Daylight without a location.
   * Shown by default.
   */
  shown?(snapshot: DashboardSnapshot): boolean;
  /**
   * Why the tile is not drawn when its own `shown` says no, in a sentence:
   * the sources footer gives it in the tile's place. None by default.
   */
  absent?(snapshot: DashboardSnapshot): string;
  /**
   * Rows in place of one number (Inside: one per sensor), each a name, a
   * reading, an aside and a line; drawn under the title, before the delta.
   */
  rows?(snapshot: DashboardSnapshot): MosaicRow[];
}

/** One row of a tile that has rows in place of one number. */
export interface MosaicRow {
  /** The row's name: "Console". */
  name: string;
  /** Its reading, formatted: "68.0°". */
  value: string;
  /** A second reading set after it, with words for screen readers alone. */
  aside: { text: string; spoken: string } | null;
  /** Its line at a width (`aria-hidden`); null when there is nothing to draw. */
  line(width: number): SVGSVGElement | null;
}

/** A row's line height. */
export const ROW_LINE_HEIGHT = 26;

/**
 * Whether the record covers a tile, by `record_depth.tiles`.
 *
 * @param snapshot - The dashboard snapshot.
 * @param key - The tile's record-depth key.
 * @returns False only when the snapshot lists the tile as not shown.
 */
function depthShown(snapshot: DashboardSnapshot, key: string): boolean {
  const tile = snapshot.record_depth?.tiles.find((t) => t.key === key);
  return tile?.shown ?? true;
}

/**
 * A mini chart's SVG root, `aria-hidden` and sized to its box.
 *
 * @param width - The width to draw at.
 * @param height - The height; `MINI_HEIGHT` by default.
 * @returns The `svg`.
 */
export function miniChart(width: number, height = MINI_HEIGHT): SVGSVGElement {
  const el = svgEl("svg", {
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    class: "mosaic-svg",
  });
  el.setAttribute("aria-hidden", "true");
  el.setAttribute("focusable", "false");
  return el;
}

/** The "usual range" legend: the band's swatch, then its name. */
function bandLegend(): HTMLElement {
  const legend = document.createElement("p");
  legend.className = "mosaic-legend";
  legend.dataset.part = "legend";
  const swatch = document.createElement("span");
  swatch.className = "mosaic-swatch";
  legend.append(swatch, "usual range");
  return legend;
}

/**
 * A tile's rows: each its name, its reading and aside, then its line.
 *
 * @param rows - The rows.
 * @returns The list, and a call that draws each row's line once the list
 *   is in the page and has a width.
 */
function rowList(rows: MosaicRow[]): [HTMLElement, () => void] {
  const draws: (() => void)[] = [];
  const list = document.createElement("div");
  list.className = "mosaic-rows";
  list.dataset.part = "rows";
  for (const row of rows) {
    const item = document.createElement("div");
    item.className = "mosaic-row";
    item.dataset.part = "row";
    const name = document.createElement("p");
    name.className = "mosaic-row-name";
    name.textContent = row.name;
    const reading = document.createElement("p");
    reading.className = "mosaic-row-reading";
    const value = document.createElement("span");
    value.className = "mosaic-row-value";
    value.textContent = row.value;
    reading.append(value);
    if (row.aside) {
      const aside = document.createElement("span");
      aside.className = "mosaic-row-aside";
      aside.append(spoken(` ${row.aside.spoken} `), row.aside.text);
      reading.append(aside);
    }
    const plot = document.createElement("div");
    plot.className = "mosaic-row-line";
    plot.setAttribute("aria-hidden", "true");
    item.append(name, reading, plot);
    list.append(item);
    draws.push(() =>
      drawSized(plot, (width) => {
        const svg = row.line(width);
        plot.replaceChildren(...(svg ? [svg] : []));
      }),
    );
  }
  return [list, () => draws.forEach((draw) => draw())];
}

/** Fill a mosaic tile: one link holding its title, number, delta and chart. */
function render(tile: MosaicTile, host: HTMLElement, snapshot: DashboardSnapshot): void {
  const link = document.createElement("a");
  link.className = "mosaic-link";
  link.href = `#overview/${tile.slug}`;
  link.dataset.part = "link";
  // The kicker renderTiles put first, the tile's h2, moves into the link.
  const title = host.querySelector<HTMLElement>(":scope > [data-part=kicker]");
  if (title) {
    title.classList.add("mosaic-title");
    link.append(title);
  }

  const rows = tile.rows?.(snapshot) ?? [];
  let drawRows = () => {};
  if (rows.length) {
    const [list, draw] = rowList(rows);
    link.append(list);
    drawRows = draw;
  }
  const reading = tile.read(snapshot);
  if (reading.value !== null) {
    const value = document.createElement("p");
    value.className = "mosaic-value";
    value.dataset.part = "value";
    value.append(reading.value);
    if (reading.unit) {
      const unit = document.createElement("span");
      unit.className = "mosaic-unit";
      // A no-break space, so the unit never wraps from its number.
      unit.textContent = `${String.fromCharCode(160)}${reading.unit}`;
      value.append(unit);
    }
    link.append(value);
  }
  if (reading.delta !== null) {
    const delta = document.createElement("p");
    delta.className = "mosaic-delta";
    delta.dataset.part = "delta";
    delta.textContent = reading.delta;
    link.append(delta);
  }
  if (reading.spoken) link.append(spoken(reading.spoken));
  const chart = document.createElement("div");
  chart.className = "mosaic-chart";
  chart.dataset.part = "mini-chart";
  chart.setAttribute("aria-hidden", "true");
  link.append(chart, spoken(`Open the ${tile.zoomName} view`));
  host.replaceChildren(link);
  drawRows();
  const draw = tile.chart;
  if (!draw) return;
  const plot = document.createElement("div");
  chart.append(plot);
  const band = tile.bandUnderBars?.(snapshot) ?? false;
  drawSized(plot, (width) => {
    const svg = draw(snapshot, width);
    plot.replaceChildren(...(svg ? [svg] : []));
    // A band under bars carries its legend, never a label inside the plot.
    // The legend goes above the chart, so the chart keeps the tile's foot,
    // the baseline its row's charts share.
    const legend = chart.querySelector("[data-part=legend]");
    if (svg && band && !legend) chart.prepend(bandLegend());
    if (!svg) legend?.remove();
  });
}

/**
 * A mosaic tile as an Overview section: one column on the grids, half the
 * width on a phone, drawn only when the record covers it.
 *
 * @param tile - The tile's definition.
 * @returns The section.
 */
export function mosaicSection(tile: MosaicTile): OverviewSection {
  return {
    id: tile.slug,
    label: tile.title,
    kicker: tile.title,
    mosaic: true,
    shown: (snapshot) =>
      depthShown(snapshot, tile.depth) && (tile.shown ? tile.shown(snapshot) : true),
    source: (snapshot) => tile.source(snapshot),
    absent: (snapshot) =>
      tile.absent && tile.shown && depthShown(snapshot, tile.depth) && !tile.shown(snapshot)
        ? tile.absent(snapshot)
        : null,
    render: (host, snapshot) => render(tile, host, snapshot),
  };
}
