/**
 * PROTOTYPE (throwaway, issue #619, round 6): rain over each window in the
 * rain zoom, in place of the window strip. Three panels, each with one shared
 * axis: [Today, Yesterday], [Last 7 days, Last 30 days], [Last 90 days, Last
 * 365 days]; a panel's axis top is the largest thing drawn in it. The earlier
 * years of a window are drawn one of two ways, `?years=bars|dots`:
 * - bars: a small group of thin grey bars (one per earlier year, oldest left)
 *   then this year's full-width bar in the rain colour, a year under each;
 * - dots: this year's bar, and a grey dot beside it for each earlier year at
 *   its height, labelled with its year.
 * Today and Yesterday have no earlier years: plain bars.
 *
 * Dev-only `?rainwindows=a,b,c,d,e,f` (Today, Yesterday, 7, 30, 90, 365 days)
 * feeds fake totals, and fake earlier years that vary by window, because the
 * goldens' windows are near zero or invalid.
 *
 * Offers `protoVariant`, `protoOverride`, `protoYears`, `protoHeadline`,
 * `renderProtoGauge`.
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
/** PROTOTYPE: the panels, each with one shared axis. */
const PANELS = [
  { keys: ["today", "yesterday"], head: "Today and yesterday" },
  { keys: ["7d", "30d"], head: "Last 7 and 30 days" },
  { keys: ["90d", "365d"], head: "Last 90 and 365 days" },
];

/** PROTOTYPE: one column's data. */
interface Col {
  key: string;
  total: number;
  /** Earlier years, oldest first. */
  prior: { year: number; v: number }[];
}

/** PROTOTYPE: the gauge is always drawn. */
export function protoVariant(): string | null {
  return "B";
}

/** PROTOTYPE: how earlier years are drawn, `?years=bars|dots` (default bars). */
export function protoYears(): "bars" | "dots" {
  return new URLSearchParams(location.search).get("years") === "dots" ? "dots" : "bars";
}

/** PROTOTYPE: the `?rainwindows=` override, or null. */
export function protoOverride(): number[] | null {
  const raw = new URLSearchParams(location.search).get("rainwindows");
  if (!raw) return null;
  const nums = raw.split(",").map(Number);
  return nums.length === KEYS.length && nums.every((n) => Number.isFinite(n)) ? nums : null;
}

/** PROTOTYPE: a deterministic multiple, 0.3 to 1.9, per window and year. */
function jitter(windowIdx: number, yearIdx: number): number {
  const s = Math.sin(windowIdx * 12.9898 + yearIdx * 78.233 + 1.7) * 43758.5453;
  return 0.3 + 1.6 * (s - Math.floor(s));
}

/** PROTOTYPE: the columns, hiding every window the record does not cover. */
function columns(snapshot: DashboardSnapshot): { cols: Col[]; thisYear: number } {
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
        prior = [3, 2, 1].map((back, j) => ({
          year: thisYear - back,
          v: Math.max(0, total * jitter(i, j)),
        }));
      } else {
        const row = rolling.find((r) => r.key === key);
        prior = (row?.years ?? [])
          .slice(1)
          .filter((y) => y.rain_in !== null && isShown(y.status))
          .map((y) => ({ year: y.year, v: y.rain_in as number }))
          .sort((a, b) => a.year - b.year);
      }
    }
    out.push({ key, total, prior });
  });
  return { cols: out, thisYear };
}

/**
 * PROTOTYPE: a finding for the zoom's title, from the totals against earlier
 * years.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "1.2 in the last 7 days, 38 in the last 365, wetter than 2023 and 2025".
 */
export function protoHeadline(snapshot: DashboardSnapshot): string | null {
  const { cols } = columns(snapshot);
  const short = cols.find((c) => c.key === "7d") ?? cols[0];
  const long = [...cols].reverse().find((c) => c.prior.length > 0) ?? cols[cols.length - 1];
  if (!short) return null;
  let text = `${tickFmt(short.total)} in ${short.key === "today" ? "today" : `the last ${short.key.replace("d", "")} days`}`;
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
const yy = (y: number): string => `'${String(y).slice(2)}`;

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

const HEAD = 20;
const VAL = 16;
const PLOT = 76;
const FOOT = 52;
const PH = HEAD + VAL + PLOT + FOOT;
const AX = 26;
const GAP = 16;

/** PROTOTYPE: one panel with one shared axis, at (px, py), pw wide. */
function panel(
  svg: SVGSVGElement,
  px: number,
  py: number,
  pw: number,
  head: string,
  cols: Col[],
  thisYear: number,
  mode: "bars" | "dots",
): void {
  const top = Math.max(0.01, ...cols.flatMap((c) => [c.total, ...c.prior.map((p) => p.v)]));
  const T = py + HEAD + VAL;
  const base = T + PLOT;
  const y = (v: number) => base - (v / top) * PLOT;
  text(svg, px, py + 12, head, "pg-head", "start");
  svg.append(
    svgEl("line", { x1: px, x2: px + pw, y1: py + HEAD - 2, y2: py + HEAD - 2, class: "pg-axis" }),
  );
  const ax = px + AX - 3;
  svg.append(svgEl("line", { x1: ax, x2: ax, y1: T, y2: base, class: "pg-axis" }));
  svg.append(svgEl("line", { x1: ax, x2: px + pw, y1: base, y2: base, class: "pg-floor" }));
  text(svg, ax - 3, T + 4, tickFmt(top), "pg-tick", "end");
  text(svg, ax - 3, base + 4, "0", "pg-tick", "end");
  const gw = (pw - AX) / cols.length;
  cols.forEach((c, i) => {
    const gx = px + AX + gw * i;
    const has = c.prior.length > 0;
    const nameY = base + (has && mode === "bars" ? 30 : 14);
    if (!has) {
      const bw = Math.min(44, gw * 0.5);
      const bx = gx + (gw - bw) / 2;
      svg.append(
        svgEl("rect", {
          x: bx,
          y: y(c.total),
          width: bw,
          height: Math.max(1, base - y(c.total)),
          class: "pg-fill",
        }),
      );
      text(svg, bx + bw / 2, y(c.total) - 4, fmt(c.total), "pg-total");
      (NAMES[c.key] ?? [c.key]).forEach((l, k) =>
        text(svg, gx + gw / 2, nameY + k * 14, l, "pg-name"),
      );
      return;
    }
    if (mode === "bars") {
      const n = c.prior.length;
      const unit = Math.min(22, (gw - 6) / (n + 2));
      const thin = unit - 2;
      const full = unit * 2 - 2;
      let x = gx + (gw - unit * (n + 2)) / 2;
      c.prior.forEach((p) => {
        svg.append(
          svgEl("rect", {
            x,
            y: y(p.v),
            width: thin,
            height: Math.max(1, base - y(p.v)),
            class: "pg-prior-bar",
          }),
        );
        text(svg, x + thin / 2, base + 14, yy(p.year), "pg-name");
        x += unit;
      });
      svg.append(
        svgEl("rect", {
          x,
          y: y(c.total),
          width: full,
          height: Math.max(1, base - y(c.total)),
          class: "pg-fill",
        }),
      );
      text(svg, x + full / 2, y(c.total) - 4, fmt(c.total), "pg-total");
      text(svg, x + full / 2, base + 14, yy(thisYear), "pg-name pg-this");
    } else {
      const bw = Math.min(40, gw * 0.4);
      const bx = gx + 4;
      svg.append(
        svgEl("rect", {
          x: bx,
          y: y(c.total),
          width: bw,
          height: Math.max(1, base - y(c.total)),
          class: "pg-fill",
        }),
      );
      text(svg, bx + bw / 2, y(c.total) - 4, fmt(c.total), "pg-total");
      const dx = bx + bw + 10;
      let lastY = 1e9;
      [...c.prior]
        .sort((a, b) => y(b.v) - y(a.v))
        .forEach((p) => {
          svg.append(svgEl("circle", { cx: dx, cy: y(p.v), r: 4, class: "pg-dot" }));
          const ly = Math.min(y(p.v) + 4, lastY - 13);
          lastY = ly;
          text(svg, dx + 8, ly, yy(p.year), "pg-name", "start");
        });
      (NAMES[c.key] ?? [c.key]).forEach((l, k) =>
        text(svg, bx + bw / 2 + 10, nameY + k * 14, l, "pg-name"),
      );
    }
    if (mode === "bars") {
      (NAMES[c.key] ?? [c.key]).forEach((l, k) =>
        text(svg, gx + gw / 2, nameY + k * 14, l, "pg-name pg-win"),
      );
    }
  });
}

/** PROTOTYPE: the three panels, across on a wide screen, stacked on a phone. */
function draw(host: HTMLElement, cols: Col[], thisYear: number, width: number): void {
  host.replaceChildren();
  const mode = protoYears();
  const panels = PANELS.map((p) => ({
    head: p.head,
    cols: cols.filter((c) => p.keys.includes(c.key)),
  })).filter((p) => p.cols.length);
  const across = width >= 560;
  const n = panels.length;
  const H = across ? PH : n * PH + (n - 1) * 8;
  const svg = svgEl("svg", {
    class: "pg-svg",
    width,
    height: H,
    viewBox: `0 0 ${width} ${H}`,
    role: "img",
    "aria-label": "Rain over each window, in three panels with one axis each",
  });
  const pw = across ? (width - GAP * (n - 1)) / n : width;
  panels.forEach((p, i) => {
    const px = across ? i * (pw + GAP) : 0;
    const py = across ? 0 : i * (PH + 8);
    panel(svg, px, py, pw, p.head, p.cols, thisYear, mode);
  });
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
  const { cols, thisYear } = columns(snapshot);
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
    `Grey: the same window in earlier years. Each panel has one axis.${over ? ` PROTOTYPE: fake totals from ?rainwindows=${over.join(",")}.` : ""}`,
  );
  const fig = make("figure", "pg-fig");
  wrap.append(fig, note);
  host.append(wrap);
  drawSized(fig, (width) => draw(fig, cols, thisYear, width));
}
