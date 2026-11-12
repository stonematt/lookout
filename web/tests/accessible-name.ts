/**
 * An element's accessible name from its content, as a browser computes it
 * for a link: the text of every node not hidden by `aria-hidden`, element
 * by element.
 *
 * @param el - The element (a link, or a list inside one).
 * @returns Its name, whitespace collapsed.
 */
export function nameFrom(el: Element): string {
  const parts: string[] = [];
  const walk = (node: Node) => {
    if (node.nodeType === Node.TEXT_NODE) parts.push(node.textContent ?? "");
    else if (node instanceof Element && node.getAttribute("aria-hidden") !== "true") {
      for (const child of node.childNodes) walk(child);
      parts.push(" ");
    }
  };
  walk(el);
  return parts.join("").replace(/\s+/g, " ").trim();
}
