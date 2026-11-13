/**
 * The wind zoom (#476, spec #460; boards `WindZoom`, `WindZoomDark`): under
 * the Wind tile's speed, direction and gust, the wind rose, then the live
 * wind sentence: where it blows from and its gust, in full.
 *
 * The rose panel reads the snapshot's `wind_rose` (#465):
 *
 * - a segmented control picks the 24 h, 7 day or 30 day window. It only
 *   switches which snapshot window is drawn; a window the record does not
 *   cover (`status.shown` false) is not offered, and with one window left
 *   there is no control.
 * - the rose: 16 directions, each bin's highest gust as a pale edged wedge
 *   and its average wind as a dark wedge over it, rings every 5 mph, and
 *   now (the live speed and direction) as the one orange dot with a leader
 *   to its label. The SVG is hidden from screen readers; a table of each
 *   direction's average and gust stands in for it.
 * - the calm line: the calm share, named by what the snapshot counts (under
 *   `calm_below_mph`, or no direction).
 * - the share of the readings that were not calm by 8 compass points, as
 *   bars with their percent in words.
 *
 * An invalid window is hatched with its reason in place of the rose and
 * bars. A snapshot without `wind_rose` (before Release A) hides the panel.
 * No label names a speed threshold such as "windy" (#484, 2026-11-12): the
 * snapshot counts calm and not calm, nothing else.
 *
 * Offers `windZoom`.
 */
import "./wind.css";

import { segmented } from "../../charts/days-switch.js";
import { drawSized } from "../../charts/svg.js";
import { make, svgEl } from "../../dom.js";
import { mph } from "../../format.js";
import { beyondRecord, gapElement, isComparable } from "../../gaps.js";
import type {
  DashboardSnapshot,
  SnapshotConditions,
  WindRose,
  WindRoseWindow,
} from "../../snapshot.js";
import { calmGust, calmThreshold, isCalm, windClauses, windTile } from "../mosaic-conditions.js";
import type { ZoomDefinition } from "../zoom.js";

/** The window the rose opens on when the record covers it (the board's). */
const FIRST_KEY = "7d";

/** Miles per hour between the rose's rings. */
const RING_MPH = 5;

/** The rose's widest diameter, px, so it never fills a desktop row. */
const MAX_SIDE = 400;

/** Room around the outer ring for the compass labels, px. */
const RIM = 26;

/** Degrees left clear between neighbouring wedges. */
const WEDGE_GAP = 2;

/** The now dot's radius (DESIGN.md: an r5.5 `now` dot). */
const NOW_R = 5.5;

/** The compass points labelled round the rim. */
const RIM_POINTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] as const;

/** The window last pressed, kept across a refresh of the snapshot. */
let chosenKey: string | null = null;

/**
 * The windows the record covers, shortest first: one the archive holds none
 * of is left out, not drawn as a hatched rose (#606).
 */
function shownWindows(rose: WindRose | undefined): WindRoseWindow[] {
  return rose?.windows.filter((w) => !beyondRecord(w.status)) ?? [];
}

/** A window's length in its unit: 24 hours, or 7 days from 48 hours. */
function windowLength(w: WindRoseWindow): { count: number; unit: "hours" | "days" } {
  return w.hours < 48 ? { count: w.hours, unit: "hours" } : { count: w.hours / 24, unit: "days" };
}

/** A window's segment label: "24 h", "7 days". */
function segmentLabel(w: WindRoseWindow): string {
  const { count, unit } = windowLength(w);
  return `${count} ${unit === "hours" ? "h" : unit}`;
}

/** A window as a caption's tail: "last 24 hours", "last 7 days". */
function spanShort(w: WindRoseWindow): string {
  const { count, unit } = windowLength(w);
  return `last ${count} ${unit}`;
}

/** A window as a phrase: "the last 24 hours", "the last 7 days". */
function spanPhrase(w: WindRoseWindow): string {
  return `the ${spanShort(w)}`;
}

// ---------- the control ----------

/**
 * The segmented control over the windows (DESIGN.md "Segmented control and select",
 * the 30 / 90 switch's look, 44px tall at every width).
 */
function windowControl(
  windows: WindRoseWindow[],
  chosen: () => string,
  choose: (key: string) => void,
): HTMLElement {
  return segmented(
    "days-seg wind-seg",
    "Window shown",
    "window",
    windows.map((w) => ({ key: w.key, text: segmentLabel(w) })),
    chosen,
    choose,
  );
}

// ---------- the rose ----------

/** A point on the rose: `deg` clockwise from north, `r` px from the centre. */
function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const a = (deg * Math.PI) / 180;
  return [cx + r * Math.sin(a), cy - r * Math.cos(a)];
}

/** A wedge from the centre out to `r`, between two bearings. */
function wedge(cx: number, cy: number, r: number, from: number, to: number): string {
  const [x1, y1] = polar(cx, cy, r, from);
  const [x2, y2] = polar(cx, cy, r, to);
  const f = (v: number) => v.toFixed(1);
  return `M${f(cx)},${f(cy)} L${f(x1)},${f(y1)} A${f(r)},${f(r)} 0 0 1 ${f(x2)},${f(y2)} Z`;
}

/**
 * The rose for one valid window at a width.
 *
 * @param w - The window.
 * @param c - The live conditions, for the now dot.
 * @param width - The panel's width, px.
 * @returns The SVG, hidden from screen readers (the table stands in).
 */
function roseSvg(w: WindRoseWindow, c: SnapshotConditions, width: number): SVGSVGElement {
  const side = Math.min(width, MAX_SIDE);
  const cx = side / 2;
  const cy = side / 2;
  const outer = side / 2 - RIM;
  // A calm wind has no bearing to plot (#598).
  const nowSpeed = c.wind_dir_deg === null || isCalm(c) ? null : c.wind_speed_mph;
  // The scale reaches the next ring past the highest gust (or now's speed).
  const top = Math.max(
    RING_MPH,
    nowSpeed ?? 0,
    ...w.bins.map((b) => Math.max(b.gust_mph ?? 0, b.average_mph ?? 0)),
  );
  const rings = Math.ceil(top / RING_MPH);
  const radius = (speed: number) => (speed / (rings * RING_MPH)) * outer;

  const svg = svgEl("svg", {
    width: side,
    height: side,
    viewBox: `0 0 ${side} ${side}`,
    "aria-hidden": "true",
    focusable: "false",
  });
  RIM_POINTS.forEach((_, i) => {
    const [x, y] = polar(cx, cy, outer, i * 45);
    svg.append(svgEl("line", { class: "wind-spoke", x1: cx, y1: cy, x2: x, y2: y }));
  });
  for (let i = 1; i <= rings; i++) {
    svg.append(svgEl("circle", { class: "wind-ring", cx, cy, r: radius(i * RING_MPH) }));
  }
  // Ring speeds up the N spoke, every other ring when they would crowd.
  const every = outer / rings < 18 ? 2 : 1;
  for (let i = every; i <= rings; i += every) {
    svg.append(
      svgEl(
        "text",
        { class: "wind-tick", x: cx + 4, y: cy - radius(i * RING_MPH) + 13 },
        String(i * RING_MPH),
      ),
    );
  }
  RIM_POINTS.forEach((point, i) => {
    const [x, y] = polar(cx, cy, outer + RIM / 2, i * 45);
    svg.append(
      svgEl("text", { class: "wind-compass", x, y: y + 4, "text-anchor": "middle" }, point),
    );
  });
  const half = 360 / w.bins.length / 2 - WEDGE_GAP / 2;
  for (const bin of w.bins) {
    if (bin.readings === 0) continue;
    const from = bin.centre_deg - half;
    const to = bin.centre_deg + half;
    if (bin.gust_mph !== null) {
      svg.append(
        svgEl("path", { class: "wind-gust", d: wedge(cx, cy, radius(bin.gust_mph), from, to) }),
      );
    }
    if (bin.average_mph !== null) {
      svg.append(
        svgEl("path", {
          class: "wind-average",
          d: wedge(cx, cy, radius(bin.average_mph), from, to),
        }),
      );
    }
  }
  if (nowSpeed !== null && c.wind_dir_deg !== null) {
    const [dx, dy] = polar(cx, cy, radius(nowSpeed), c.wind_dir_deg);
    const text = `now ${mph(nowSpeed)}`;
    // The label sits up and out on the side away from the dot's bearing,
    // past the rings' centre, kept inside the rose's square.
    const right = dx <= cx;
    const textWidth = text.length * 7.6;
    const lx = right
      ? Math.min(side - textWidth - 2, Math.max(dx + 28, cx + 24))
      : Math.max(textWidth + 2, Math.min(dx - 28, cx - 24));
    const ly = Math.max(16, Math.min(side - 6, dy - 22));
    const leader = { x1: dx, y1: dy, x2: right ? lx - 3 : lx + 3, y2: ly - 4 };
    svg.append(
      // The leader crosses the wedges on a halo of the surface, so the now
      // mark never touches a data fill (DESIGN.md "The Now Stands Off Rule").
      svgEl("line", { class: "wind-now-halo", ...leader }),
      svgEl("line", { class: "wind-now-leader", ...leader }),
      svgEl("circle", { class: "wind-now-dot", cx: dx, cy: dy, r: NOW_R }),
      svgEl(
        "text",
        { class: "wind-now-label", x: lx, y: ly, "text-anchor": right ? "start" : "end" },
        text,
      ),
    );
  }
  return svg;
}

/** The table that stands in for the rose for a screen reader. */
function roseTable(w: WindRoseWindow): HTMLElement {
  const table = make("table", "");
  table.append(
    make("caption", "", `Average wind and highest gust from each direction, ${spanShort(w)}`),
  );
  const head = make("thead", "");
  const row = make("tr", "");
  for (const name of ["Direction", "Average wind", "Highest gust"]) {
    const th = make("th", "", name);
    th.scope = "col";
    row.append(th);
  }
  head.append(row);
  const body = make("tbody", "");
  for (const bin of w.bins) {
    const tr = make("tr", "");
    const th = make("th", "", bin.direction);
    th.scope = "row";
    tr.append(th);
    if (bin.readings === 0) {
      const td = make("td", "", "no readings");
      td.colSpan = 2;
      tr.append(td);
    } else {
      tr.append(
        make("td", "", bin.average_mph === null ? "not known" : mph(bin.average_mph)),
        make("td", "", bin.gust_mph === null ? "not known" : mph(bin.gust_mph)),
      );
    }
    body.append(tr);
  }
  table.append(head, body);
  const box = make("div", "wind-sr");
  box.append(
    make(
      "p",
      "",
      `Wind rose for the ${spanShort(w)}: each direction's average wind and highest gust, ` +
        "listed in the table that follows.",
    ),
    table,
  );
  return box;
}

/** The key under the rose: the two wedges and the rings. */
function roseKey(): HTMLElement {
  const key = make("p", "wind-key");
  key.setAttribute("aria-hidden", "true");
  const item = (cls: string, text: string) => {
    const span = make("span", "wind-key-item");
    span.append(make("span", `wind-swatch ${cls}`), text);
    return span;
  };
  key.append(
    item("wind-swatch-average", "average wind"),
    item("wind-swatch-gust", "highest gust"),
    make("span", "wind-key-item", `rings every ${RING_MPH} mph`),
  );
  return key;
}

/** The calm line: the calm share, named by what the snapshot counts. */
function calmLine(w: WindRoseWindow, calmBelow: number): string {
  if (w.calm_pct === null) return `No wind readings in ${spanPhrase(w)}.`;
  return (
    `Calm ${w.calm_pct}% of readings in ${spanPhrase(w)}: ` +
    `under ${calmThreshold(calmBelow)} or no direction. ` +
    "Calm readings have no direction, so the rose and the bars leave them out."
  );
}

/**
 * The share of the readings that were not calm, by compass point, each bar
 * drawn against 100%; null for a window with no readings, which the calm
 * line already says.
 */
function pointBars(w: WindRoseWindow): HTMLElement | null {
  if (w.readings === 0) return null;
  const block = make("div", "wind-bars");
  block.append(make("p", "wind-bars-title", "Readings that were not calm, by direction"));
  if (w.points.every((p) => p.share_pct === null)) {
    block.append(make("p", "wind-bars-none", `Every reading in ${spanPhrase(w)} was calm.`));
    return block;
  }
  const list = make("ul", "wind-bar-list");
  for (const point of w.points) {
    const item = make("li", "wind-bar");
    item.dataset.point = point.direction;
    // The percent follows the bar's end, as on the board.
    const track = make("span", "wind-bar-track");
    const fill = make("span", "wind-bar-fill");
    fill.setAttribute("aria-hidden", "true");
    fill.style.width = `${point.share_pct ?? 0}%`;
    track.append(
      fill,
      make("span", "wind-bar-value", point.share_pct === null ? "n/a" : `${point.share_pct}%`),
    );
    item.append(make("span", "wind-bar-label", point.direction), track);
    list.append(item);
  }
  block.append(list);
  return block;
}

/**
 * The rose panel: the control, then the rose, its key and the calm line,
 * then the bars, for the chosen window.
 *
 * @param host - The panel, in the page.
 * @param snapshot - The dashboard snapshot; its `wind_rose` is present.
 */
function renderRose(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const rose = snapshot.wind_rose!;
  const windows = shownWindows(rose);
  const keys = windows.map((w) => w.key);
  let key =
    chosenKey !== null && keys.includes(chosenKey)
      ? chosenKey
      : keys.includes(FIRST_KEY)
        ? FIRST_KEY
        : keys[0]!;

  const body = make("div", "wind-body");
  const figure = make("div", "wind-rose");
  const side = make("div", "wind-side");
  body.append(figure, side);

  if (windows.length > 1) {
    host.append(
      windowControl(
        windows,
        () => key,
        (next) => {
          key = next;
          chosenKey = next;
          draw();
        },
      ),
    );
  }
  host.append(body);

  const current = () => windows.find((w) => w.key === key)!;
  const redrawRose = drawSized(figure, (width) => {
    const w = current();
    figure.querySelector("svg")?.remove();
    if (isComparable(w.status)) figure.prepend(roseSvg(w, snapshot.current, width));
  });

  function draw(): void {
    const w = current();
    figure.replaceChildren();
    side.replaceChildren();
    if (!isComparable(w.status)) {
      figure.append(gapElement(w.status));
      return;
    }
    redrawRose();
    figure.append(
      roseKey(),
      make("p", "wind-calm", calmLine(w, rose.calm_below_mph)),
      roseTable(w),
    );
    const bars = pointBars(w);
    if (bars) side.append(bars);
  }
  draw();
}

// ---------- the live wind ----------

/**
 * The live wind sentence: where it blows from and its gust.
 *
 * @param c - The current conditions.
 * @returns E.g. "From the SE, gusts to 2.5 mph."
 */
function windSentence(c: SnapshotConditions): string {
  const { calm, from: fromClause, gust } = windClauses(c);
  // Calm: no direction (#598).
  if (calm) {
    const rest = calmGust(c, gust);
    return rest === "no gusts" ? "Calm." : `Calm; ${rest}.`;
  }
  const from = fromClause && `F${fromClause.slice(1)}`;
  let rest: string;
  if (from && gust) rest = `${from}, ${gust}.`;
  else if (from) rest = `${from}; gust not known.`;
  else if (gust) rest = `Direction not known; ${gust}.`;
  else rest = "Direction and gust not known.";
  return c.wind_speed_mph === null ? `Wind speed not known. ${rest}` : rest;
}

/** The live wind panel: the sentence. */
function renderNow(host: HTMLElement, snapshot: DashboardSnapshot): void {
  host.append(make("p", "wind-now-sentence", windSentence(snapshot.current)));
}

/** The wind zoom. */
export const windZoom: ZoomDefinition = {
  slug: windTile.slug,
  title: windTile.title,
  lead: (snapshot) => windTile.read(snapshot),
  panels: [
    {
      id: "rose",
      title: "Where the wind comes from",
      render: renderRose,
      wide: true,
      shown: (snapshot) => shownWindows(snapshot.wind_rose).length > 0,
    },
    { id: "now", title: "Right now", render: renderNow },
  ],
};
