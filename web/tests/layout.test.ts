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

import { currentLayout, layoutFor, mountLayout, setLayout } from "../src/layout.js";
import { renderOverview } from "../src/overview/index.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";
import { viewport } from "./viewport.js";

const raining = parseDashboard(golden("raining", "dashboard"));

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
    const view = viewport(390);
    const layout = mountLayout(document.documentElement, view.match);
    const seen = [document.documentElement.dataset.layout];
    for (const width of [760, 1179, 1180, 1440, 759]) {
      view.resize(width);
      seen.push(document.documentElement.dataset.layout);
    }

    expect(seen).toEqual(["phone", "two", "two", "four", "four", "phone"]);
    expect(layout.layout()).toBe("phone");
  });

  it("tells its caller each new layout the viewport moves it to, not the first", () => {
    const view = viewport(390);
    const seen: string[] = [];
    mountLayout(document.documentElement, view.match, (layout) => seen.push(layout));
    for (const width of [500, 760, 1179, 1440, 759]) view.resize(width);

    expect(seen).toEqual(["two", "four", "phone"]);
  });

  it("stops following the viewport once stopped", () => {
    const view = viewport(1280);
    const layout = mountLayout(document.documentElement, view.match);
    layout.stop();
    view.resize(390);

    expect(currentLayout()).toBe("four");
  });

  it("reads phone with no attribute set", () => {
    expect(currentLayout()).toBe("phone");
  });
});

describe("the Overview's grid under each layout", () => {
  // The grid's rules, then the mosaic's: the feature's two rows.
  const css = ["../src/styles/tiles.css", "../src/overview/mosaic.css"]
    .map((path) => readFileSync(resolve(import.meta.dirname, path), "utf8"))
    .join("\n");

  /** The grid's columns and each tile's span, with the layout set before it renders. */
  function grid(layout: "phone" | "two" | "four"): {
    columns: string;
    spans: string[];
    featureRows: string;
  } {
    document.head.innerHTML = "";
    const style = document.createElement("style");
    style.textContent = css;
    document.head.append(style);
    setLayout(layout);
    document.body.innerHTML = "<div id=panel></div>";
    renderOverview(document.getElementById("panel")!, raining);
    const tiles = [...document.querySelectorAll<HTMLElement>(".tile")];
    const feature = document.querySelector<HTMLElement>(".tile[data-section=feature]")!;
    return {
      columns: getComputedStyle(document.querySelector(".tile-grid")!).gridTemplateColumns,
      spans: tiles.map((tile) => getComputedStyle(tile).gridColumn || "span 1"),
      featureRows: getComputedStyle(feature).gridRow || "span 1",
    };
  }

  /** The mosaic's ten tiles. */
  const halves = Array<string>(10).fill("span 1");

  it("is two columns on a phone: now and feature full width, the mosaic two to a row", () => {
    const { columns, spans } = grid("phone");
    expect(columns).toBe("repeat(2, minmax(0, 1fr))");
    expect(spans).toEqual(["1 / -1", "1 / -1", ...halves]);
  });

  it("is two columns at two-column: now and feature full width, one column a tile", () => {
    const { columns, spans, featureRows } = grid("two");
    expect(columns).toBe("repeat(2, minmax(0, 1fr))");
    expect(spans).toEqual(["span 2", "span 2", ...halves]);
    expect(featureRows).toBe("span 1");
  });

  it("is four columns at four-column, the feature 2×2 at the left (Desktop board)", () => {
    const { columns, spans, featureRows } = grid("four");
    expect(columns).toBe("repeat(4, minmax(0, 1fr))");
    // The now tile takes the whole first row; the feature two columns and
    // two rows, so the first four mosaic tiles fill the 2×2 beside it, row
    // first, and the rest flow one column each in the phone's order.
    expect(spans).toEqual(["span 4", "span 2", ...halves]);
    expect(featureRows).toBe("span 2");
  });
});
