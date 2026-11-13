/**
 * The small parts the conditions zooms draw in their panels (#477): a
 * sentence, a large value line and a chart box drawn at the panel's width.
 * Every part is content only; the zoom shell (`../zoom.ts`) owns the
 * heading, Close, focus and the route. A zoom's link on to a tab is its
 * "More in" panel (`more.ts`).
 *
 * Offers `sentence`, `factValue`, `sizedChart` and `dayTrace` (a 24-hour
 * hourly series at full size: the Inside zoom's sensors, the Pressure and
 * Humidity zooms' lines).
 */
import "./parts.css";

import { drawSized, scale } from "../../charts/svg.js";
import { svgEl } from "../../dom.js";
import { lastIndex, traceMarks } from "../mosaic-conditions.js";
import { miniChart } from "../mosaic.js";

/** A day trace's height: the mini chart's, and room for its labels. */
const TRACE_HEIGHT = 112;

/** The strip under a day trace that holds its end labels. */
const TRACE_AXIS = 18;

/** The space at the left of a day trace for its low and high labels. */
const TRACE_LEFT = 36;

/**
 * The last 24 hours of an hourly series at a width: the line, its low and
 * high at the left, "24 h ago" and "now" under it, the newest hour as the
 * now dot.
 *
 * @param values - The series, hourly, oldest first.
 * @param width - The width to draw at.
 * @param label - A low or high value as its label ("68°", "30.01").
 * @param pad - The least room above and below the line, in its unit.
 * @returns The chart, or null with fewer than two hours or no reading.
 */
export function dayTrace(
  values: (number | null)[],
  width: number,
  label: (value: number) => string,
  pad: number,
): SVGSVGElement | null {
  if (values.length < 2 || lastIndex(values) < 0) return null;
  const numbers = values.filter((v): v is number => v !== null);
  const low = Math.min(...numbers);
  const high = Math.max(...numbers);
  const bottom = TRACE_HEIGHT - TRACE_AXIS;
  const x = scale(0, values.length - 1, TRACE_LEFT, width - 6);
  const y = scale(low - pad, high + pad, bottom - 6, 8);
  const svg = miniChart(width, TRACE_HEIGHT);
  svg.append(
    svgEl("line", { class: "mosaic-base", x1: TRACE_LEFT, x2: width, y1: bottom, y2: bottom }),
    svgEl("text", { class: "mosaic-label", x: 0, y: y(high) + 4 }, label(high)),
  );
  // A steady day has one value: its low is its high, labelled once.
  if (low !== high) {
    svg.append(svgEl("text", { class: "mosaic-label", x: 0, y: y(low) + 4 }, label(low)));
  }
  svg.append(
    svgEl("text", { class: "mosaic-label", x: TRACE_LEFT, y: TRACE_HEIGHT - 2 }, "24 h ago"),
    svgEl(
      "text",
      { class: "mosaic-label", x: width, y: TRACE_HEIGHT - 2, "text-anchor": "end" },
      "now",
    ),
  );
  svg.append(...traceMarks(values, x, y, 4));
  return svg;
}

/**
 * A sentence in a zoom panel, in Finding type.
 *
 * @param host - The panel.
 * @param text - The sentence.
 * @param part - Its `data-part`, for tests.
 * @returns The paragraph.
 */
export function sentence(host: HTMLElement, text: string, part: string): HTMLParagraphElement {
  return paragraph(host, "zoom-sentence", text, part);
}

/** A paragraph of one class appended to a panel. */
function paragraph(
  host: HTMLElement,
  className: string,
  text: string,
  part: string,
): HTMLParagraphElement {
  const p = document.createElement("p");
  p.className = className;
  p.dataset.part = part;
  p.textContent = text;
  host.append(p);
  return p;
}

/**
 * A large value in a zoom panel, in Fact type ("7:15 am to 4:38 pm").
 *
 * @param host - The panel.
 * @param text - The value.
 * @param part - Its `data-part`, for tests.
 * @returns The paragraph.
 */
export function factValue(host: HTMLElement, text: string, part: string): HTMLParagraphElement {
  return paragraph(host, "zoom-fact", text, part);
}

/**
 * A chart box that draws at its own width now and on resize; `aria-hidden`,
 * since the sentences beside it carry what it shows.
 *
 * @param host - The panel.
 * @param part - Its `data-part`, for tests.
 * @param draw - The chart at a width, or null when there is nothing to draw.
 * @returns The box.
 */
export function sizedChart(
  host: HTMLElement,
  part: string,
  draw: (width: number) => SVGSVGElement | null,
): HTMLDivElement {
  const box = document.createElement("div");
  box.className = "zoom-chart";
  box.dataset.part = part;
  box.setAttribute("aria-hidden", "true");
  host.append(box);
  drawSized(box, (width) => {
    const svg = draw(width);
    box.replaceChildren(...(svg ? [svg] : []));
  });
  return box;
}
