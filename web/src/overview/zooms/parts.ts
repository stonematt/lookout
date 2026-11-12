/**
 * The small parts the conditions zooms draw in their panels (#477): a
 * sentence, a large value line, a chart box drawn at the panel's width, and
 * a link on to a tab. Every part is content only; the zoom shell
 * (`../zoom.ts`) owns the heading, Close, focus and the route.
 *
 * Offers `sentence`, `factValue`, `sizedChart` and `onwardLink`.
 */
import "./parts.css";

import { drawSized } from "../../charts/svg.js";

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

/**
 * A link on from a zoom to the tab that holds more ("More in Solar").
 *
 * @param host - The panel.
 * @param href - The tab's hash: "#solar".
 * @param words - The link's words.
 * @returns The link.
 */
export function onwardLink(host: HTMLElement, href: string, words: string): HTMLAnchorElement {
  const link = document.createElement("a");
  link.className = "zoom-onward";
  link.dataset.part = "onward";
  link.href = href;
  link.textContent = words;
  const glyph = document.createElement("span");
  glyph.className = "zoom-onward-glyph";
  glyph.setAttribute("aria-hidden", "true");
  glyph.textContent = "›";
  link.append(glyph);
  host.append(link);
  return link;
}
