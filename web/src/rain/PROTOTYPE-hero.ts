/**
 * PROTOTYPE (throwaway, issue #619, round 4): the Overview hero's rain today
 * and yesterday, three desktop column options (`?hero=a|b|c`) and the one-line
 * text under the temperature below 760 px. The CSS (`PROTOTYPE-gauge.css`)
 * shows one set or the other by width.
 *
 * Offers `HERO_OPTIONS`, `protoHero`, `protoHeroRainNode`.
 */
import { make, svgEl } from "../dom.js";
import { inches } from "../format.js";
import { isShown } from "../overview/history.js";
import type { DashboardSnapshot } from "../snapshot.js";
import { protoOverride } from "./PROTOTYPE-gauge.js";

/** PROTOTYPE: the hero options. */
export const HERO_OPTIONS = ["a", "b", "c"] as const;

/** PROTOTYPE: the hero option asked for (`?hero=a|b|c`; "1" reads as "c"), or null. */
export function protoHero(): string | null {
  const h = new URLSearchParams(location.search).get("hero");
  if (h === "1") return "c";
  return h && (HERO_OPTIONS as readonly string[]).includes(h) ? h : null;
}

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

/** PROTOTYPE: a number with its small " in" unit, in a paragraph. */
function numberLine(cls: string, v: number): HTMLElement {
  const n = make("p", cls, two(v));
  n.append(make("span", "pg-unit", " in"));
  return n;
}

/**
 * PROTOTYPE: the hero's rain. "line" is the one-line text under the
 * temperature (below 760 px); "a", "b" and "c" are the desktop columns.
 *
 * @param snapshot - The dashboard snapshot.
 * @param kind - "line", "a", "b" or "c".
 * @returns The element, or null when it did not rain.
 */
export function protoHeroRainNode(snapshot: DashboardSnapshot, kind: string): HTMLElement | null {
  const r = heroRain(snapshot);
  if (!r) return null;
  if (kind === "line") {
    const p = make("p", "pg-hero1");
    p.dataset.prototype = "rain-gauge";
    p.textContent = `Rain today ${inches(r.today)} · yesterday ${inches(r.yest)}`;
    return p;
  }
  const box = make("div", `pg-hdesk pg-h-${kind}`);
  box.dataset.prototype = "rain-gauge";
  const max = Math.max(r.today, r.yest, 0.01);
  if (kind === "a") {
    // mirrors the temperature: a big number, its unit, a caption line
    box.append(
      make("p", "pg-ha-label", "Rain today"),
      numberLine("pg-ha-num", r.today),
      make("p", "pg-ha-cap", `yesterday ${inches(r.yest)}`),
    );
    return box;
  }
  if (kind === "b") {
    // two bars on one scale; baseline and label line match the ribbon chart
    // (TOP 22 + plot 150 + gap 6 + track 14 = baseline 192; labels at 209)
    const band = snapshot.temperature_context.usual_band ?? null;
    const H = 192 + 22 + (band ? 18 : 0);
    const W = 150;
    const svg = svgEl("svg", {
      class: "pg-hb-svg",
      width: W,
      height: H,
      viewBox: `0 0 ${W} ${H}`,
      role: "img",
      "aria-label": `Rain today ${two(r.today)} in, yesterday ${two(r.yest)} in, one scale`,
    });
    const bars: [string, number, number][] = [
      ["Today", r.today, 20],
      ["Yesterday", r.yest, 90],
    ];
    for (const [name, v, x] of bars) {
      const h = v > 0 ? Math.max(2, (v / max) * 126) : 0;
      svg.append(
        svgEl("rect", { x, y: 192 - h, width: 40, height: h, class: "pg-hb-bar" }),
        svgEl(
          "text",
          { x: x + 20, y: 192 - h - 6, class: "pg-hb-val", "text-anchor": "middle" },
          two(v),
        ),
        svgEl("text", { x: x + 20, y: 209, class: "pg-hb-name", "text-anchor": "middle" }, name),
      );
    }
    svg.append(svgEl("line", { x1: 4, x2: W - 4, y1: 192, y2: 192, class: "pg-hb-floor" }));
    box.append(svg);
    return box;
  }
  // c: two enlarged rows, label above, a hero-style number, a long bar
  for (const [k, v] of [
    ["Rain today", r.today],
    ["Yesterday", r.yest],
  ] as const) {
    const row = make("div", "pg-hc-row");
    const track = make("span", "pg-hc-track");
    const bar = make("span", "pg-hc-bar");
    bar.style.width = `${v > 0 ? Math.max(3, (v / max) * 100) : 0}%`;
    track.append(bar);
    row.append(make("p", "pg-hc-k", k), numberLine("pg-hc-num", v), track);
    box.append(row);
  }
  return box;
}
