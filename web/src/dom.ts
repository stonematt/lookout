/**
 * The element helpers the views build with: `make` for HTML, `spoken` and
 * `spokenRunOn` for screen-reader-only text, `svgEl` for SVG, `escapeHtml`
 * for any value written into an HTML template, and `html`, the tagged
 * template that escapes each value for you.
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
 * Hidden words that run on from visible text (`.ov-sr-run`, in
 * `styles/tiles.css`): inline and zero-sized, so a screen reader hears
 * "on record, 57° now." with no space before the comma. An `.ov-sr` span
 * is laid out as a block, and a browser puts a space at a block's edge when
 * it names a link from its content. Append it inside the visible text's own
 * element; it becomes part of that element's text.
 *
 * @param text - The words, from their first stop or comma.
 * @returns The span.
 */
export function spokenRunOn(text: string): HTMLSpanElement {
  const node = document.createElement("span");
  node.className = "ov-sr-run";
  node.textContent = text;
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

declare const HTML_FRAGMENT: unique symbol;

/**
 * Markup built by `html`. It is a string to the type checker, so it goes
 * straight into `innerHTML` or `insertAdjacentHTML`; at runtime it is a
 * `String` object `html` remembers, which a snapshot's plain strings and
 * numbers can never be, so only `html`'s own output passes unescaped.
 */
export type Html = string & { readonly [HTML_FRAGMENT]: true };

/** What a `html` template takes: text and numbers (escaped), or fragments. */
export type HtmlValue = string | number | boolean | Html | readonly Html[];

/** Every fragment `html` built. */
const fragments = new WeakSet<object>();

/** One interpolation as markup: a fragment as is, anything else escaped. */
function htmlPart(value: HtmlValue): string {
  if (Array.isArray(value)) return value.map(htmlPart).join("");
  if (typeof value === "object" && fragments.has(value)) return String(value);
  return escapeHtml(String(value));
}

/**
 * The tagged template every `innerHTML` block is built with (#261): each
 * `${value}` is escaped as `escapeHtml` does, unless it is a fragment
 * `html` built, or a list of them, which is markup already. ESLint's
 * `no-unsanitized` knows the tag (`escape.taggedTemplates`), so a block
 * built through it needs no `eslint-disable`. Prettier would format an
 * `html` template as HTML and change its whitespace, which is text, so
 * `.prettierrc.json` turns embedded formatting off.
 *
 * @param strings - The template's literal markup.
 * @param values - Its interpolations.
 * @returns The markup, as a fragment another `html` template takes as is.
 */
export function html(strings: TemplateStringsArray, ...values: readonly HtmlValue[]): Html {
  let markup = strings[0]!;
  values.forEach((value, i) => {
    markup += htmlPart(value) + strings[i + 1]!;
  });
  const fragment = new String(markup);
  fragments.add(fragment);
  return fragment as unknown as Html;
}
