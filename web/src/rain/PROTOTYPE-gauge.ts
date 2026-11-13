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
  const yIdx = cols.findIndex((c) => c.key === "yesterday");
  if (yIdx >= 0) {
    const dx = L + cw * (yIdx + 1);
    svg.append(svgEl("line", { x1: dx, x2: dx, y1: T, y2: H - B, class: "pg-grid" }));
  }
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
    // Today and Yesterday share one axis (one panel); the rest have their own.
    const pair = c.key === "today" || c.key === "yesterday";
    const pairMax = Math.max(
      ...cols.filter((x) => x.key === "today" || x.key === "yesterday").map((x) => x.total),
    );
    const ax = niceAxis(pair ? pairMax : Math.max(c.total, ...c.prior.map((p) => p.v)));
    const y = (v: number) => T + plotH * (1 - v / ax.top);
    if (c.key === "yesterday") {
      svg.append(
        svgEl("line", { x1: cx + cw / 2, x2: cx + cw / 2, y1: T, y2: T + plotH, class: "pg-grid" }),
      );
    }
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
      if (c.key !== "yesterday" && (!narrow || t === ax.top)) {
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

/** PROTOTYPE: a note under the chart saying what the marks mean. */
const NOTES: Record<string, string> = {
  B: "B: Today and Yesterday share one scale (left panel); each longer window is a tube with its own scale (see the tick labels), so heights are not comparable across tubes. Orange ticks are the same window in earlier years.",
  C: "C: one shared log axis (0, 0.1, 1, 10 in) with Today and Yesterday side by side as one panel: equal steps are equal ratios, so height reads as order of magnitude, not amount. Orange ticks are earlier years.",
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
    if (variant === "B") multiples(fig, cols, width);
    else shared(fig, cols, width, true);
  });
}

/** PROTOTYPE: the hero placement asked for by `?hero=1|2`, or null. */
export function protoHero(): "1" | "2" | null {
  const h = new URLSearchParams(location.search).get("hero");
  return h === "1" || h === "2" ? h : null;
}

/**
 * PROTOTYPE: rain today and yesterday for the hero, or null when neither is
 * above 0 (the old Streamlit rule: show only when it rained). `?rainwindows=`
 * drives both; else the live total today and the Yesterday window total.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The two totals, or null to hide the hero rain.
 */
export function protoHeroRain(snapshot: DashboardSnapshot): { today: number; yest: number } | null {
  const over = protoOverride();
  let today = snapshot.current.rain_today_in;
  const yw = snapshot.rain_tab.windows.find((w) => w.key === "yesterday");
  let yest = yw && isShown(yw.status) ? yw.total_in : null;
  if (over) {
    today = over[0] ?? null;
    yest = over[1] ?? null;
  }
  const t = today ?? 0;
  const y = yest ?? 0;
  return t > 0 || y > 0 ? { today: t, yest: y } : null;
}

/**
 * PROTOTYPE: the hero's rain, placement 1 (words under the temperature) or
 * 2 (a compact block on the far side of the ribbon with a tiny two-column gauge).
 *
 * @param snapshot - The dashboard snapshot.
 * @param placement - "1" or "2".
 * @returns The element, or null when it did not rain.
 */
export function protoHeroRainNode(
  snapshot: DashboardSnapshot,
  placement: "1" | "2",
): HTMLElement | null {
  const r = protoHeroRain(snapshot);
  if (!r) return null;
  if (placement === "1") {
    const p = make("p", "pg-hero1");
    p.dataset.prototype = "rain-gauge";
    p.textContent = `Rain today ${inches(r.today)} · yesterday ${inches(r.yest)}`;
    return p;
  }
  const box = make("div", "pg-hero2");
  box.dataset.prototype = "rain-gauge";
  const H = 40;
  const max = Math.max(r.today, r.yest, 0.1);
  const svg = svgEl("svg", {
    class: "pg-hero2-svg",
    width: 44,
    height: H + 2,
    viewBox: `0 0 44 ${H + 2}`,
    "aria-hidden": "true",
  });
  [r.yest, r.today].forEach((v, i) => {
    const h = v > 0 ? Math.max(2, (v / max) * H) : 0;
    svg.append(
      svgEl("rect", { x: 4 + i * 20, y: H - h, width: 16, height: h, class: "pg-hero2-bar" }),
    );
  });
  svg.append(svgEl("line", { x1: 0, x2: 44, y1: H, y2: H, class: "pg-floor" }));
  const nums = make("dl", "pg-hero2-nums");
  for (const [k, v] of [
    ["Today", r.today],
    ["Yesterday", r.yest],
  ] as const) {
    const row = make("div", "pg-hero2-row");
    row.append(make("dt", "pg-hero2-k", k), make("dd", "pg-hero2-v", inches(v)));
    nums.append(row);
  }
  box.append(svg, nums);
  return box;
}
