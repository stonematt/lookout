/**
 * The element helpers the views build with: `make` for HTML, `spoken` for
 * screen-reader-only text, `svgEl` for SVG, and `escapeHtml` for any value
 * written into an HTML template.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * An element with a class and, optionally, text.
 *
 * @param tag - The element name.
 * @param className - Its class.
 * @param text - Its text.
 * @returns The element.
 */
export function make<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Text for screen readers alone (`.ov-sr`, in `styles/tiles.css`): the one
 * way the views write hidden words.
 *
 * @param content - Its text, or the nodes it holds (a hidden list's items).
 * @param tag - The element name; a span by default.
 * @returns The element.
 */
export function spoken<K extends keyof HTMLElementTagNameMap = "span">(
  content: string | readonly Node[],
  tag: K = "span" as K,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = "ov-sr";
  if (typeof content === "string") node.textContent = content;
  else node.append(...content);
  return node;
}

/**
 * An SVG element with attributes and, optionally, text.
 *
 * @param tag - The element name.
 * @param attrs - Its attributes.
 * @param text - Its text.
 * @returns The element.
 */
export function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * A value as HTML text or attribute content: `& < > " '` become character
 * references. Every value interpolated into an `innerHTML` template goes
 * through it, numbers included, since a snapshot field typed as a number
 * is not checked at runtime.
 *
 * @param value - The text or number.
 * @returns The escaped text.
 */
export function escapeHtml(value: string | number): string {
  return String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}
