/**
 * The one HTML element helper the views build with.
 */

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
