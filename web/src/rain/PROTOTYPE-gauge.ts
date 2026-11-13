/**
 * PROTOTYPE (throwaway, issue #619): rain over each window as vertical
 * rain-gauge columns. Four structurally different variants, picked by
 * `?variant=A|B|C|D` and mounted by `renderYears` in `rain/index.ts` in
 * place of the window strip. Not production code: no tests, no polish.
 *
 * - A: one shared linear axis, a column per window (the honest baseline).
 * - B: small multiples, a tube per window, each with its own scale.
 * - C: one shared symlog axis.
 * - D: one tall tube, nested "since" levels with a mark at each window.
 *
 * Dev-only `?rainwindows=a,b,c,d,e,f` (Today, Yesterday, 7, 30, 90, 365 days)
 * feeds fake totals, and fake earlier-year ticks, because the goldens' windows
 * are near zero or invalid.
 *
 * Offers `protoVariant`, `protoOverride`, `renderProtoGauge`, `protoHeroTotal`.
 */
import { drawSized } from "../charts/svg.js";
import { make, svgEl } from "../dom.js";
import { inches } from "../format.js";
import { isShown } from "../overview/history.js";
import type { DashboardSnapshot } from "../snapshot.js";
import { WINDOW_NAMES } from "./text.js";

/** PROTOTYPE: the keys, in the order the columns draw. */
const KEYS = ["today", "yesterday", "7d", "30d", "90d", "365d"];
/** PROTOTYPE: fake earlier years, as multiples of this year's total. */
const FAKE_PRIOR = [0.55, 1.6, 0.9];

/** PROTOTYPE: one column's data. */
interface Col {
  key: string;
  name: string;
  total: number;
  prior: { year: number; v: number }[];
}

/** PROTOTYPE: the variant asked for, or null. */
export function protoVariant(): string | null {
  const v = new URLSearchParams(location.search).get("variant");
  return v && v.length === 1 && "ABCD".includes(v.toUpperCase()) ? v.toUpperCase() : null;
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
    out.push({ key, name: WINDOW_NAMES[key] ?? key, total, prior });
  });
  return out;
}

/** PROTOTYPE: a "nice" axis top and ticks at 1, 2 or 5 times a power of ten. */
function niceAxis(max: number): { top: number; ticks: number[] } {
  const m = Math.max(max, 0.05);
  const pow = Math.pow(10, Math.floor(Math.log10(m / 3)));
  const step = [1, 2, 5, 10].map((k) => k * pow).find((s) => m / s <= 4) ?? pow * 10;
  const ticks: number[] = [];
  for (let v = 0; v <= m + step * 0.999; v += step) ticks.push(+v.toFixed(6));
  return { top: ticks[ticks.length - 1] as number, ticks };
}

const fmt = (v: number): string => (v >= 10 ? v.toFixed(1) : v.toFixed(2));
const tickFmt = (v: number): string => (v >= 1 ? String(+v.toFixed(1)) : String(+v.toFixed(2)));
const yy = (y: number): string => `'${String(y).slice(2)}`;
const shortName = (n: string): string => n.replace(" days", "d").replace("Yesterday", "Yest.");

type Svg = SVGSVGElement;

/** PROTOTYPE: append a text node. */
function text(svg: Svg, x: number, y: number, s: string, cls: string, anchor = "middle"): void {
  svg.append(
    svgEl("text", { x: +x.toFixed(1), y: +y.toFixed(1), class: cls, "text-anchor": anchor }, s),
  );
}

/** PROTOTYPE: a bare svg. */
function bare(width: number, height: number, label: string): Svg {
  return svgEl("svg", {
    class: "pg-svg",
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": label,
  });
}

/** PROTOTYPE: columns on a shared axis, A (linear) and C (symlog). */
function shared(host: HTMLElement, cols: Col[], width: number, log: boolean): void {
  host.replaceChildren();
  const H = 230;
  const L = 36;
  const T = 24;
  const B = 34;
  const R = 6;
  const max = Math.max(...cols.flatMap((c) => [c.total, ...c.prior.map((p) => p.v)]));
  const svg = bare(width, H, "Rain over each window, one shared axis");
  const plotH = H - T - B;
  let f: (v: number) => number;
  let ticks: number[];
  if (log) {
    const k = 0.1;
    const raw = (v: number) => Math.log10(1 + v / k);
    const cand = [0, 0.1, 0.5, 1, 5, 10, 50, 100, 500];
    ticks = cand.filter((t) => t <= max * 1.05);
    const top = raw(Math.max(max, ticks[ticks.length - 1] as number));
    f = (v) => raw(v) / top;
  } else {
    const ax = niceAxis(max);
    ticks = ax.ticks;
    f = (v) => v / ax.top;
  }
  const y = (v: number) => T + plotH * (1 - Math.min(1, f(v)));
  for (const t of ticks) {
    svg.append(
      svgEl("line", {
        x1: L,
        x2: width - R,
        y1: y(t),
        y2: y(t),
        class: t === 0 ? "pg-floor" : "pg-grid",
      }),
    );
    text(svg, L - 4, y(t) + 3, tickFmt(t), "pg-tick", "end");
  }
  const cw = (width - L - R) / cols.length;
  const bw = Math.min(40, cw * 0.55);
  cols.forEach((c, i) => {
    const cx = L + cw * (i + 0.5);
    const top = y(c.total);
    svg.append(
      svgEl("rect", {
        x: cx - bw / 2,
        y: top,
        width: bw,
        height: Math.max(1, y(0) - top),
        class: "pg-fill",
      }),
    );
    text(svg, cx, top - 4, fmt(c.total), "pg-total");
    text(svg, cx, H - B + 14, shortName(c.name), "pg-name");
    for (const p of c.prior) {
      const py = y(p.v);
      svg.append(
        svgEl("line", {
          x1: cx - cw * 0.45,
          x2: cx + cw * 0.45,
          y1: py,
          y2: py,
          class: "pg-prior",
        }),
      );
      text(svg, cx + cw * 0.45, py - 2, yy(p.year), "pg-year", "end");
    }
  });
  host.append(svg);
}

/** PROTOTYPE: B, one tube per window with its own scale. */
function multiples(host: HTMLElement, cols: Col[], width: number): void {
  host.replaceChildren();
  const H = 220;
  const T = 24;
  const B = 34;
  const svg = bare(width, H, "Rain over each window, one tube each with its own scale");
  const plotH = H - T - B;
  const cw = width / cols.length;
  const narrow = cw < 70;
  const tw = narrow ? cw * 0.2 : Math.min(30, cw * 0.4);
  cols.forEach((c, i) => {
    const cx = cw * (i + 0.5);
    const ax = niceAxis(Math.max(c.total, ...c.prior.map((p) => p.v)));
    const y = (v: number) => T + plotH * (1 - v / ax.top);
    svg.append(
      svgEl("rect", { x: cx - tw / 2, y: T, width: tw, height: plotH, rx: 4, class: "pg-tube" }),
    );
    svg.append(
      svgEl("rect", {
        x: cx - tw / 2,
        y: y(c.total),
        width: tw,
        height: Math.max(1, y(0) - y(c.total)),
        class: "pg-fill",
      }),
    );
    for (const t of ax.ticks) {
      svg.append(
        svgEl("line", {
          x1: cx - tw / 2 - 3,
          x2: cx - tw / 2,
          y1: y(t),
          y2: y(t),
          class: "pg-floor",
        }),
      );
      if (!narrow || t === ax.top) {
        text(svg, cx - tw / 2 - 4, y(t) + 3, tickFmt(t), "pg-tick pg-tick-sm", "end");
      }
    }
    for (const p of c.prior) {
      svg.append(
        svgEl("line", {
          x1: cx - tw / 2,
          x2: cx + tw / 2 + 3,
          y1: y(p.v),
          y2: y(p.v),
          class: "pg-prior",
        }),
      );
      text(svg, cx + tw / 2 + 4, y(p.v) + 3, yy(p.year), "pg-year", "start");
    }
    text(svg, cx, y(c.total) - 4, fmt(c.total), "pg-total");
    text(svg, cx, H - B + 14, shortName(c.name), "pg-name");
  });
  host.append(svg);
}

/** PROTOTYPE: D, one tall tube; nested "since" levels. */
function tall(host: HTMLElement, cols: Col[], width: number): void {
  host.replaceChildren();
  const H = 300;
  const T = 14;
  const B = 12;
  const svg = bare(width, H, "Rain since each window began, one tall gauge");
  const plotH = H - T - B;
  const levels: { name: string; v: number }[] = [];
  let acc = 0;
  const byKey = new Map(cols.map((c) => [c.key, c]));
  const today = byKey.get("today");
  const yest = byKey.get("yesterday");
  if (today) {
    acc = today.total;
    levels.push({ name: "Today", v: acc });
  }
  if (yest) {
    acc = acc + yest.total;
    levels.push({ name: "Since yesterday", v: acc });
  }
  for (const k of ["7d", "30d", "90d", "365d"]) {
    const c = byKey.get(k);
    if (c) {
      acc = Math.max(acc, c.total);
      levels.push({ name: `Last ${c.name.toLowerCase()}`, v: acc });
    }
  }
  const ax = niceAxis(acc);
  const y = (v: number) => T + plotH * (1 - v / ax.top);
  const tx = 52;
  const tw = Math.min(56, width * 0.2);
  svg.append(svgEl("rect", { x: tx, y: T, width: tw, height: plotH, rx: 6, class: "pg-tube" }));
  svg.append(
    svgEl("rect", {
      x: tx,
      y: y(acc),
      width: tw,
      height: Math.max(1, y(0) - y(acc)),
      class: "pg-fill",
    }),
  );
  for (const t of ax.ticks) {
    svg.append(svgEl("line", { x1: tx - 4, x2: tx, y1: y(t), y2: y(t), class: "pg-floor" }));
    text(svg, tx - 6, y(t) + 3, tickFmt(t), "pg-tick", "end");
  }
  // Labels, pushed apart so they do not collide.
  const labelY: number[] = [];
  levels.forEach((lv, i) => {
    const want = y(lv.v);
    const prev = labelY[i - 1];
    labelY.push(prev === undefined ? want : Math.min(want, prev - 15));
  });
  levels.forEach((lv, i) => {
    const ly = labelY[i] as number;
    const x0 = tx + tw;
    svg.append(
      svgEl("line", { x1: tx, x2: x0 + 8, y1: y(lv.v), y2: y(lv.v), class: "pg-prior pg-level" }),
    );
    svg.append(
      svgEl("polyline", {
        points: `${x0 + 8},${y(lv.v)} ${x0 + 16},${ly - 3} ${x0 + 22},${ly - 3}`,
        class: "pg-lead",
      }),
    );
    text(svg, x0 + 26, ly, `${fmt(lv.v)} in  ${lv.name}`, "pg-total pg-total-l", "start");
  });
  host.append(svg);
}

/** PROTOTYPE: a note under the chart saying what the marks mean. */
const NOTES: Record<string, string> = {
  A: "A: one shared linear axis. Each column is the rain over the window ending today. Ticks are the same window ending today in earlier years.",
  B: "B: a tube per window, each with its own scale (see the tick labels): heights are not comparable across tubes. Ticks are earlier years.",
  C: "C: one shared log axis (0, 0.1, 1, 10 in): equal steps are equal ratios, so height reads as order of magnitude, not amount. Ticks are earlier years.",
  D: "D: one gauge filled to the longest window. Each mark is the rain since that window began: Today; Since yesterday = today + yesterday; then the last 7, 30, 90 and 365 days (each contains the shorter ones).",
};

/**
 * PROTOTYPE: draw the variant into the host, in place of the window strip.
 *
 * @param host - The panel.
 * @param snapshot - The dashboard snapshot.
 * @param variant - "A" to "D".
 */
export function renderProtoGauge(
  host: HTMLElement,
  snapshot: DashboardSnapshot,
  variant: string,
): void {
  const wrap = make("div", "pg-wrap");
  wrap.dataset.prototype = "rain-gauge";
  const cols = columns(snapshot);
  const over = protoOverride();
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
  const note = make(
    "p",
    "pg-note",
    `PROTOTYPE ${NOTES[variant] ?? ""}${over ? ` Fake totals from ?rainwindows=${over.join(",")}.` : ""}`,
  );
  const fig = make("figure", "pg-fig");
  wrap.append(fig, note);
  host.append(wrap);
  drawSized(fig, (width) => {
    if (variant === "A") shared(fig, cols, width, false);
    else if (variant === "B") multiples(fig, cols, width);
    else if (variant === "C") shared(fig, cols, width, true);
    else tall(fig, cols, width);
  });
}

/**
 * PROTOTYPE: the hero rain track's total, e.g. "0.42 in today".
 *
 * @param live - The live total today, or null.
 * @param slotsSum - The sum of the track's slots.
 * @returns The words.
 */
export function protoHeroTotal(live: number | null, slotsSum: number): string {
  return `${inches(live ?? slotsSum)} today`;
}
