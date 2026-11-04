// @vitest-environment happy-dom
/**
 * The page layout switch (`layout.ts`, #334): the root's `data-layout` at
 * the 760px and 1180px thresholds, and the tile grid that follows it. happy-dom
 * evaluates no media queries, so the switch gets a stand-in `matchMedia`, and
 * the grid is read through the tile stylesheet with the layout set first.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { MatchMedia } from "../src/layout.js";
import { currentLayout, layoutFor, mountLayout, setLayout } from "../src/layout.js";
import { renderOverview } from "../src/overview/index.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const raining = parseDashboard(golden("raining", "dashboard"));

/** A `matchMedia` over a viewport width a test moves. */
function viewport(width: number): { match: MatchMedia; resize(to: number): void } {
  const lists: { min: number; list: MediaQueryList; listeners: Set<() => void> }[] = [];
  return {
    match(query) {
      const min = Number(/min-width:\s*(\d+)px/.exec(query)![1]);
      const listeners = new Set<() => void>();
      const list = {
        get matches() {
          return width >= min;
        },
        addEventListener: (_: string, fn: () => void) => listeners.add(fn),
        removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
      } as unknown as MediaQueryList;
      lists.push({ min, list, listeners });
      return list;
    },
    resize(to) {
      const before = lists.map(({ list }) => list.matches);
      width = to;
      lists.forEach(({ list, listeners }, i) => {
        if (list.matches !== before[i]) for (const fn of listeners) fn();
      });
    },
  };
}

afterEach(() => document.documentElement.removeAttribute("data-layout"));

describe("the layout switch", () => {
  it("names the layout for a width at the 760px and 1180px thresholds", () => {
    expect([390, 759, 760, 1179, 1180, 1440].map(layoutFor)).toEqual([
      "phone",
      "phone",
      "two",
      "two",
      "four",
      "four",
    ]);
  });

  it("sets the root's attribute from the viewport, and again as it crosses each threshold", () => {
    const screen = viewport(390);
    const layout = mountLayout(document.documentElement, screen.match);
    const seen = [document.documentElement.dataset.layout];
    for (const width of [760, 1179, 1180, 1440, 759]) {
      screen.resize(width);
      seen.push(document.documentElement.dataset.layout);
    }

    expect(seen).toEqual(["phone", "two", "two", "four", "four", "phone"]);
    expect(layout.layout()).toBe("phone");
  });

  it("tells its caller each new layout the viewport moves it to, not the first", () => {
    const screen = viewport(390);
    const seen: string[] = [];
    mountLayout(document.documentElement, screen.match, (layout) => seen.push(layout));
    for (const width of [500, 760, 1179, 1440, 759]) screen.resize(width);

    expect(seen).toEqual(["two", "four", "phone"]);
  });

  it("stops following the viewport once stopped", () => {
    const screen = viewport(1280);
    const layout = mountLayout(document.documentElement, screen.match);
    layout.stop();
    screen.resize(390);

    expect(currentLayout()).toBe("four");
  });

  it("reads phone with no attribute set", () => {
    expect(currentLayout()).toBe("phone");
  });
});

describe("the Overview's grid under each layout", () => {
  const css = readFileSync(resolve(import.meta.dirname, "../src/styles/tiles.css"), "utf8");

  /** The grid's columns and each tile's span, with the layout set before it renders. */
  function grid(layout: "phone" | "two" | "four"): { columns: string; spans: string[] } {
    document.head.innerHTML = "";
    const style = document.createElement("style");
    style.textContent = css;
    document.head.append(style);
    setLayout(layout);
    document.body.innerHTML = "<div id=panel></div>";
    renderOverview(document.getElementById("panel")!, raining);
    const tiles = [...document.querySelectorAll<HTMLElement>(".tile")];
    return {
      columns: getComputedStyle(document.querySelector(".tile-grid")!).gridTemplateColumns,
      spans: tiles.map((tile) => getComputedStyle(tile).gridColumn || "span 1"),
    };
  }

  it("is one column of tiles on a phone", () => {
    const { columns, spans } = grid("phone");
    expect(columns).toBe("minmax(0, 1fr)");
    expect(new Set(spans)).toEqual(new Set(["span 1"]));
  });

  it("is two columns of tiles at two-column, no tile wider than both", () => {
    const { columns, spans } = grid("two");
    expect(columns).toBe("repeat(2, minmax(0, 1fr))");
    // now, lead and periods full width; the two single tiles share a row;
    // the almanac full width, the rolling windows half.
    expect(spans).toEqual(["span 2", "span 2", "span 2", "span 1", "span 1", "span 2", "span 1"]);
  });

  it("is four columns of tiles at four-column, each tile at its declared span", () => {
    const { columns, spans } = grid("four");
    expect(columns).toBe("repeat(4, minmax(0, 1fr))");
    expect(spans).toEqual(["span 2", "span 2", "span 2", "span 1", "span 1", "span 3", "span 1"]);
  });
});
