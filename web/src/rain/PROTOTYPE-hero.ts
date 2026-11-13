/**
 * PROTOTYPE (throwaway, issue #619, round 5): the Overview hero's rain today
 * and yesterday. From 760 px up a vertical two-bar gauge (Today, Yesterday,
 * one scale) on the ribbon chart's baseline, headed "Rain"; below 760 px the
 * one-line text under the temperature. Hidden when neither is above 0. The
 * CSS (`PROTOTYPE-gauge.css`) shows one or the other by width.
 *
 * Offers `protoHeroRainNode`.
 */
import { make, svgEl } from "../dom.js";
import { inches } from "../format.js";
import { isShown } from "../overview/history.js";
import type { DashboardSnapshot } from "../snapshot.js";
import { protoOverride } from "./PROTOTYPE-gauge.js";

/**
 * PROTOTYPE: rain today and yesterday for the hero, or null when neither is
 * above 0. `?rainwindows=` drives both; else the live total today and the
 * Yesterday window total. Today is the live rain today, not the event's total.
 */
function heroRain(snapshot: DashboardSnapshot): { today: number; yest: number } | null {
  const over = protoOverride();
  const yw = snapshot.rain_tab.windows.find((w) => w.key === "yesterday");
  let today = snapshot.current.rain_today_in;
  let yest = yw && isShown(yw.status) ? yw.total_in : null;
  if (over) {
    today = over[0] ?? null;
    yest = over[1] ?? null;
  }
  const t = today ?? 0;
  const y = yest ?? 0;
  return t > 0 || y > 0 ? { today: t, yest: y } : null;
}

const two = (v: number): string => v.toFixed(2);

/** The chart's top gutter (`TOP` in today-hero.ts) and its label baseline offset. */
const CHART_TOP = 22;
const LABEL_DROP = 17;

/**
 * PROTOTYPE: the hero's rain.
 *
 * @param snapshot - The dashboard snapshot.
 * @param kind - "line" (below 760 px) or "gauge" (from 760 px up).
 * @param chart - For "gauge": the ribbon chart's block, whose plot height and
 *   baseline the gauge reads and tracks, so the bars stand on its x-axis line.
 * @returns The element, or null when it did not rain.
 */
export function protoHeroRainNode(
  snapshot: DashboardSnapshot,
  kind: "line" | "gauge",
  chart?: HTMLElement,
): HTMLElement | null {
  const r = heroRain(snapshot);
  if (!r) return null;
  if (kind === "line") {
    const p = make("p", "pg-hero1");
    p.dataset.prototype = "rain-gauge";
    p.textContent = `Rain today ${inches(r.today)} · yesterday ${inches(r.yest)}`;
    return p;
  }
  const box = make("div", "pg-hdesk");
  box.dataset.prototype = "rain-gauge";
  const max = Math.max(r.today, r.yest, 0.01);
  const W = 150;
  const paint = (): void => {
    // the chart's own geometry: its height, and the y of its rain-track floor
    const src = chart?.querySelector("svg");
    const floor = chart?.querySelector(".today-rain-floor");
    const H = src ? Number(src.getAttribute("height")) || 232 : 232;
    const base = floor ? Number(floor.getAttribute("y1")) || 192 : 192;
    const svg = svgEl("svg", {
      class: "pg-hb-svg",
      width: W,
      height: H,
      viewBox: `0 0 ${W} ${H}`,
      role: "img",
      "aria-label": `Rain today ${two(r.today)} in, yesterday ${two(r.yest)} in, one scale`,
    });
    // "Rain" on the chart's top row; the bars share one scale below it
    svg.append(
      svgEl(
        "text",
        { x: W / 2, y: CHART_TOP - 8, class: "pg-hb-head", "text-anchor": "middle" },
        "Rain",
      ),
    );
    const room = base - CHART_TOP - 22;
    const bars: [string, number, number][] = [
      ["Today", r.today, 20],
      ["Yesterday", r.yest, 90],
    ];
    for (const [name, v, x] of bars) {
      const h = v > 0 ? Math.max(2, (v / max) * room) : 0;
      svg.append(
        svgEl("rect", { x, y: base - h, width: 40, height: h, class: "pg-hb-bar" }),
        svgEl(
          "text",
          { x: x + 20, y: base - h - 6, class: "pg-hb-val", "text-anchor": "middle" },
          two(v),
        ),
        svgEl(
          "text",
          { x: x + 20, y: base + LABEL_DROP, class: "pg-hb-name", "text-anchor": "middle" },
          name,
        ),
      );
    }
    svg.append(svgEl("line", { x1: 4, x2: W - 4, y1: base, y2: base, class: "pg-hb-floor" }));
    box.replaceChildren(svg);
  };
  paint();
  if (chart && typeof ResizeObserver === "function") {
    new ResizeObserver(paint).observe(chart);
  }
  return box;
}
