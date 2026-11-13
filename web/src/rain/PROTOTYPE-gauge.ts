/**
 * PROTOTYPE (throwaway, issue #619, round 3): rain over each window as plain
 * bars, one per window, replacing the window strip in the rain zoom when
 * `?variant=B` is set. Today and Yesterday share one scale (one panel); each
 * longer window's axis runs to the larger of its total and its earlier years'
 * totals, with the earlier years as grey ticks. Also the hero's rain today and
 * yesterday (`?hero=1`; two placements drawn, CSS picks by width).
 *
 * Dev-only `?rainwindows=a,b,c,d,e,f` (Today, Yesterday, 7, 30, 90, 365 days)
 * feeds fake totals, and fake earlier-year ticks, because the goldens' windows
 * are near zero or invalid.
 *
 * Offers `protoVariant`, `protoOverride`, `protoHeadline`, `renderProtoGauge`,
 * `protoHero`, `protoHeroRainNode`.
 */
import { drawSized } from "../charts/svg.js";
import { make, svgEl } from "../dom.js";
import { isShown } from "../overview/history.js";
import type { DashboardSnapshot } from "../snapshot.js";

/** PROTOTYPE: the keys, in the order the columns draw. */
const KEYS = ["today", "yesterday", "7d", "30d", "90d", "365d"];
/** PROTOTYPE: the column names, in two lines at most. */
const NAMES: Record<string, string[]> = {
  today: ["Today"],
  yesterday: ["Yesterday"],
  "7d": ["Last 7", "days"],
  "30d": ["Last 30", "days"],
  "90d": ["Last 90", "days"],
  "365d": ["Last 365", "days"],
};
/** PROTOTYPE: fake earlier years, as multiples of this year's total. */
const FAKE_PRIOR = [0.55, 1.6, 0.9];

/** PROTOTYPE: one column's data. */
interface Col {
  key: string;
  total: number;
  prior: { year: number; v: number }[];
}

/** PROTOTYPE: the gauge is always drawn (no ?variant= gate any more). */
export function protoVariant(): string | null {
  return "B";
}

/** PROTOTYPE: the `?rainwindows=` override, or null. */
export function protoOverride(): number[] | null {
  const raw = new URLSearchParams(location.search).get("rainwindows");
  if (!raw) return null;
  const nums = raw.split(",").map(Number);
  return nums.length === KEYS.length && nums.every((n) => Number.isFinite(n)) ? nums : null;
}

/** PROTOTYPE: the columns, hiding every window the record does not cover. */
function columns(snapshot: DashboardSnapshot): Col[] {
  const over = protoOverride();
  const rolling = snapshot.almanac?.rolling ?? [];
  const thisYear = rolling[0]?.years[0]?.year ?? new Date(snapshot.built_at_ms).getFullYear();
  const out: Col[] = [];
  KEYS.forEach((key, i) => {
    const w = snapshot.rain_tab.windows.find((x) => x.key === key);
    let total: number | null;
    if (over) {
      total = over[i] ?? null;
    } else {
      if (!w || !isShown(w.status)) return;
      total = w.total_in;
    }
    if (total === null) return;
    let prior: Col["prior"] = [];
    if (key !== "today" && key !== "yesterday") {
      if (over) {
        prior = FAKE_PRIOR.map((m, j) => ({ year: thisYear - 1 - j, v: total * m }));
      } else {
        const row = rolling.find((r) => r.key === key);
        prior = (row?.years ?? [])
          .slice(1)
          .filter((y) => y.rain_in !== null && isShown(y.status))
          .map((y) => ({ year: y.year, v: y.rain_in as number }));
      }
    }
    out.push({ key, total, prior });
  });
  return out;
}

/**
 * PROTOTYPE: a finding for the zoom's title, from the totals against earlier
 * years. Null when the gauge is not asked for or has nothing to say.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "1.2 in in the last 7 days, 38 in the last 365, wetter than 2023 and 2025".
 */
export function protoHeadline(snapshot: DashboardSnapshot): string | null {
  if (!protoVariant()) return null;
  const cols = columns(snapshot);
  const short = cols.find((c) => c.key === "7d") ?? cols[0];
  const long = [...cols].reverse().find((c) => c.prior.length > 0) ?? cols[cols.length - 1];
  if (!short) return null;
  const label = (c: Col) =>
    c.key === "today" ? "today" : `in the last ${c.key.replace("d", "")} days`;
  let text = `${tickFmt(short.total)} in ${label(short)}`;
  if (long && long !== short) {
    text = `${tickFmt(short.total)} in the last ${short.key.replace("d", "")} days, ${tickFmt(long.total)} in the last ${long.key.replace("d", "")}`;
    const wetter = long.prior.filter((p) => long.total > p.v).map((p) => p.year);
    const drier = long.prior.filter((p) => long.total <= p.v).map((p) => p.year);
    if (wetter.length) text += `, wetter than ${wetter.join(" and ")}`;
    else if (drier.length) text += `, drier than ${drier.join(" and ")}`;
  }
  return text;
}

/** Axis numbers: short, no trailing zeros. */
const tickFmt = (v: number): string => String(+v.toFixed(v >= 10 ? 0 : v >= 1 ? 1 : 2));
/** Value labels: two decimals under 10. */
const fmt = (v: number): string => (v >= 10 ? v.toFixed(1) : v.toFixed(2));

/** PROTOTYPE: append a text node. */
function text(
  svg: SVGSVGElement,
  x: number,
  y: number,
  s: string,
  cls: string,
  anchor = "middle",
): void {
  svg.append(
    svgEl("text", { x: +x.toFixed(1), y: +y.toFixed(1), class: cls, "text-anchor": anchor }, s),
  );
}

/** PROTOTYPE: plain bars; Today and Yesterday on one scale, the rest each on their own. */
function bars(host: HTMLElement, cols: Col[], width: number): void {
  host.replaceChildren();
  const HEAD = 18;
  const T = HEAD + 18;
  const plotH = 84;
  const narrow = width / cols.length < 70;
  const NAME_H = narrow ? 52 : 34;
  const H = T + plotH + NAME_H;
  const AX = 24; // room left of each bar for its axis numbers
  const svg = svgEl("svg", {
    class: "pg-svg",
    width,
    height: H,
    viewBox: `0 0 ${width} ${H}`,
    role: "img",
    "aria-label": "Rain over each window, one bar each",
  });
  const cw = (width - 8) / cols.length;
  const bw = Math.max(16, Math.min(48, cw - AX - 2));
  const isPair = (c: Col) => c.key === "today" || c.key === "yesterday";
  const pairTop = Math.max(...cols.filter(isPair).map((c) => c.total), 0.01);
  let keyDone = false;
  cols.forEach((c, i) => {
    const cell = cw * i;
    const bx = cell + cw - bw - 2; // bar sits right in its cell, axis numbers left
    const top = isPair(c) ? pairTop : Math.max(c.total, ...c.prior.map((p) => p.v), 0.01);
    const y = (v: number) => T + plotH * (1 - v / top);
    // the axis: a thin line left of the bar, labelled at 0 and at its top
    // (the pair's second bar shares the first's axis, so has none)
    if (c.key !== "yesterday") {
      svg.append(svgEl("line", { x1: bx - 3, x2: bx - 3, y1: T, y2: T + plotH, class: "pg-axis" }));
      // the top number is dropped when the value label already says it
      if (top !== c.total) text(svg, bx - 6, T + 4, tickFmt(top), "pg-tick", "end");
      text(svg, bx - 6, T + plotH + 4, "0", "pg-tick", "end");
    }
    svg.append(
      svgEl("rect", {
        x: bx,
        y: y(c.total),
        width: bw,
        height: Math.max(1, T + plotH - y(c.total)),
        class: "pg-fill",
      }),
    );
    svg.append(
      svgEl("line", { x1: bx - 3, x2: bx + bw, y1: T + plotH, y2: T + plotH, class: "pg-floor" }),
    );
    for (const p of c.prior) {
      svg.append(
        svgEl("line", { x1: bx - 2, x2: bx + bw + 2, y1: y(p.v), y2: y(p.v), class: "pg-prior" }),
      );
    }
    if (c.prior.length && !keyDone) {
      keyDone = true;
      // the one key (a grey tick and its words), in the heading row's far end
      svg.append(
        svgEl("line", { x1: width - 98, x2: width - 84, y1: 8, y2: 8, class: "pg-prior" }),
      );
      text(svg, width, 12, "earlier years", "pg-key", "end");
    }
    text(svg, bx + bw / 2, y(c.total) - 4, fmt(c.total), "pg-total");
    const raw = NAMES[c.key] ?? [c.key];
    const lines = narrow
      ? raw.flatMap((l) => (l === "Yesterday" ? ["Yester-", "day"] : l.split(" ")))
      : raw;
    lines.forEach((line, k) => {
      text(svg, narrow ? cell + cw / 2 : bx + bw / 2, T + plotH + 16 + k * 14, line, "pg-name");
    });
  });
  // the shared panel's heading and bracket
  const yi = cols.findIndex((c) => c.key === "yesterday");
  const ti = cols.findIndex((c) => c.key === "today");
  if (yi >= 0 && ti >= 0) {
    const x0 = cw * ti + 2;
    const x1 = cw * (yi + 1) - 2;
    text(
      svg,
      x0,
      12,
      narrow ? "Today, yesterday" : "Today and yesterday, one scale",
      "pg-head",
      "start",
    );
    svg.append(svgEl("line", { x1: x0, x2: x1, y1: HEAD - 2, y2: HEAD - 2, class: "pg-axis" }));
  }
  host.append(svg);
}

/**
 * PROTOTYPE: draw the gauge into the host, in place of the window strip.
 *
 * @param host - The panel.
 * @param snapshot - The dashboard snapshot.
 */
export function renderProtoGauge(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const wrap = make("div", "pg-wrap");
  wrap.dataset.prototype = "rain-gauge";
  const cols = columns(snapshot);
  if (!cols.length) {
    wrap.append(
      make(
        "p",
        "pg-note",
        "PROTOTYPE: no window is covered by the record; nothing to draw (try ?rainwindows=0.1,0.3,1.2,4.5,12,38).",
      ),
    );
    host.append(wrap);
    return;
  }
  const over = protoOverride();
  const note = make(
    "p",
    "pg-note",
    `Each bar is its own scale (numbers at its axis); grey ticks are the same window in earlier years.${over ? ` PROTOTYPE: fake totals from ?rainwindows=${over.join(",")}.` : ""}`,
  );
  const fig = make("figure", "pg-fig");
  wrap.append(fig, note);
  host.append(wrap);
  drawSized(fig, (width) => bars(fig, cols, width));
}
