/**
 * The Overview hero's rain (#619): rain today and yesterday, shown only when
 * either is above 0. Today is the live rain today (`current.rain_today_in`),
 * which stays today's during a rain event, as an event can span more than a
 * day; yesterday is the Yesterday window's total.
 *
 * Two forms, both in the DOM, and the layout's CSS shows one
 * (`hero-rain.css`): from 760 px up, a column headed "Rain" opposite the
 * temperature, two bars on one scale whose plot and baseline are the ribbon
 * chart's (`todayFrame`), names on its x-label line; below 760 px, one line
 * under the temperature's caption.
 *
 * Offers `heroRain`, `heroRainLine` and `heroRainColumn`.
 */
import "./hero-rain.css";

import { drawSized } from "../charts/svg.js";
import { make, svgEl } from "../dom.js";
import { inches } from "../format.js";
import { todayFrame } from "../overview/today-hero.js";
import type { DashboardSnapshot } from "../snapshot.js";
import { isShown } from "../overview/history.js";

/** Rain today and yesterday, in inches. */
export interface HeroRain {
  today: number;
  yesterday: number;
}

/**
 * Rain today and yesterday, or null when neither is above 0.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The two totals; a missing one reads 0.
 */
export function heroRain(snapshot: DashboardSnapshot): HeroRain | null {
  const window = snapshot.rain_tab.windows.find((w) => w.key === "yesterday");
  const today = snapshot.current.rain_today_in ?? 0;
  const yesterday = (window && isShown(window.status) ? window.total_in : null) ?? 0;
  return today > 0 || yesterday > 0 ? { today, yesterday } : null;
}

/**
 * The one-line form, under the temperature's caption below 760 px.
 *
 * @param rain - Rain today and yesterday.
 * @returns "Rain today 0.30 in · yesterday 0.12 in".
 */
export function heroRainLine(rain: HeroRain): HTMLElement {
  const node = make(
    "p",
    "hero-rain-line",
    `Rain today ${inches(rain.today)} · yesterday ${inches(rain.yesterday)}`,
  );
  node.dataset.part = "rain-line";
  return node;
}

/** The column's width, and each bar's width and centre. */
const WIDTH = 132;
const BAR_W = 40;
const BARS: readonly { name: string; key: keyof HeroRain; mid: number }[] = [
  { name: "Today", key: "today", mid: 36 },
  { name: "Yesterday", key: "yesterday", mid: 96 },
];
/** Room over a bar for its value. */
const VALUE_ROOM = 20;

/** Draw the column on the chart's rows at the chart's width. */
function paint(box: HTMLElement, snapshot: DashboardSnapshot, rain: HeroRain, width: number): void {
  const { height, top, floor, labelY } = todayFrame(snapshot, width);
  const label = `Rain today ${inches(rain.today)}, yesterday ${inches(rain.yesterday)}, on one scale`;
  const svg = svgEl("svg", {
    class: "hero-rain-svg",
    width: WIDTH,
    height,
    viewBox: `0 0 ${WIDTH} ${height}`,
    role: "img",
    "aria-label": label,
  });
  svg.append(
    svgEl(
      "text",
      { x: WIDTH / 2, y: top - 8, class: "hero-rain-head", "text-anchor": "middle" },
      "Rain",
    ),
  );
  const max = Math.max(rain.today, rain.yesterday);
  const room = floor - top - VALUE_ROOM;
  for (const bar of BARS) {
    const value = rain[bar.key];
    const h = value > 0 ? Math.max(2, Math.round((value / max) * room)) : 0;
    svg.append(
      svgEl("rect", {
        x: bar.mid - BAR_W / 2,
        y: floor - h,
        width: BAR_W,
        height: h,
        class: "hero-rain-bar",
      }),
      svgEl(
        "text",
        { x: bar.mid, y: floor - h - 6, class: "hero-rain-value", "text-anchor": "middle" },
        inches(value, { unit: false }),
      ),
      svgEl(
        "text",
        { x: bar.mid, y: labelY, class: "hero-rain-name", "text-anchor": "middle" },
        bar.name,
      ),
    );
  }
  svg.append(
    svgEl("line", { x1: 4, x2: WIDTH - 4, y1: floor, y2: floor, class: "hero-rain-floor" }),
  );
  box.replaceChildren(svg);
}

/**
 * The column form, opposite the temperature from 760 px up. It is drawn on
 * the ribbon chart's rows, and again whenever the chart's width changes them.
 *
 * @param snapshot - The dashboard snapshot.
 * @param rain - Rain today and yesterday.
 * @param chart - The ribbon chart's block, whose width sets its rows.
 * @returns The column; append it, then call `start` once it is in the page.
 */
export function heroRainColumn(
  snapshot: DashboardSnapshot,
  rain: HeroRain,
  chart: HTMLElement,
): { box: HTMLElement; start(): void } {
  const box = make("div", "hero-rain");
  box.dataset.part = "rain-column";
  return { box, start: () => void drawSized(chart, (w) => paint(box, snapshot, rain, w)) };
}
