// @vitest-environment happy-dom
/**
 * The window range chart keeps its low and high value labels inside the
 * SVG (#534): a three-digit "≈ 102°" and a below-zero "≈ −12°" at the end of
 * the Full history row fit at a phone's width. Text cannot be measured in
 * happy-dom, so a label's extent is its characters at the chart's label size.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { LABEL_CHAR_PX } from "../src/charts/svg.js";
import { renderOverview } from "../src/overview/index.js";
import { temperaturePeriods } from "../src/overview/temperature.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

type Doc = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

let panel: HTMLElement;
let widthOf = 0;
let saved: PropertyDescriptor | undefined;

beforeEach(() => {
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
  saved = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: () => widthOf,
  });
});

afterEach(() => {
  if (saved) Object.defineProperty(HTMLElement.prototype, "clientWidth", saved);
  else delete (HTMLElement.prototype as { clientWidth?: number }).clientWidth;
});

/** The raining golden, its Full history stretched to 102.4 and -12 and marked approximate. */
function wide(): Doc {
  const doc = golden("raining", "dashboard") as Doc;
  const full = doc.temperature_context.full_history;
  full.low = -12;
  full.high = 102.4;
  full.q1 = 40;
  full.q3 = 70;
  full.status.status = "prorated";
  full.status.missing_days = 1;
  return doc;
}

describe("the window range chart's value labels", () => {
  it.each([256, 326, 1000, 1360])(
    "stay inside a %ipx SVG with 102° and −12° at the ends",
    (width) => {
      widthOf = width;
      // The temperature zoom's window chart, drawn alone as the zoom draws it.
      renderOverview(panel, parseDashboard(wide()), [temperaturePeriods]);

      const svg = panel.querySelector("[data-part=periods-chart] svg")!;
      expect(Number(svg.getAttribute("width"))).toBe(width);
      const values = [...svg.querySelectorAll<SVGTextElement>(".tc-value")];
      expect(values.map((v) => v.textContent)).toContain("≈\u00a0102°");
      expect(values.map((v) => v.textContent)).toContain("≈\u00a0−12°");

      // The plot group's left edge is its translate; a phone has none.
      const plot = svg.querySelector("g[transform]")!;
      const shift = Number(/translate\(([-\d.]+)/.exec(plot.getAttribute("transform")!)![1]);
      for (const v of values) {
        const extent = (v.textContent ?? "").length * LABEL_CHAR_PX;
        const x = shift + Number(v.getAttribute("x"));
        if (v.getAttribute("text-anchor") === "end") expect(x - extent).toBeGreaterThanOrEqual(0);
        else expect(x + extent).toBeLessThanOrEqual(width);
      }
    },
  );
});
