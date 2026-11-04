/**
 * The page layout switch (DESIGN.md "Layout"): one attribute on the root
 * element, `data-layout`, that every page layout rule keys off.
 *
 * - `phone`: under 760px of viewport, one column.
 * - `two`: from 760px, a two-column grid of tiles.
 * - `four`: from 1180px, a four-column grid of tiles.
 *
 * The switch reads `matchMedia` at the two thresholds and keeps the
 * attribute in step as the viewport crosses them. It is the tested seam:
 * happy-dom evaluates no media queries, so a test sets the layout through
 * `setLayout`, or hands `mountLayout` its own `matchMedia`, and asserts the
 * attribute, never the CSS. A chart does not follow this switch: it picks
 * its phone layout from its own width (DESIGN.md "The Chart Width Rule").
 *
 * Stylesheets select a layout as `:root[data-layout="four"]`, and both grid
 * layouts as `:root:is([data-layout="two"], [data-layout="four"])`. With no
 * attribute yet, the phone rules apply.
 */

/** The three page layouts. */
export type Layout = "phone" | "two" | "four";

/** The viewport width the two-column grid starts at. */
export const TWO_COLUMN_PX = 760;

/** The viewport width the four-column grid starts at. */
export const FOUR_COLUMN_PX = 1180;

/** The attribute the switch sets on the root element. */
export const LAYOUT_ATTRIBUTE = "data-layout";

/** `window.matchMedia`, or a stand-in a test hands over. */
export type MatchMedia = (query: string) => MediaQueryList;

/**
 * The layout for a viewport width.
 *
 * @param width - The viewport width, CSS px.
 * @returns "phone" under 760, "two" from 760, "four" from 1180.
 */
export function layoutFor(width: number): Layout {
  if (width >= FOUR_COLUMN_PX) return "four";
  if (width >= TWO_COLUMN_PX) return "two";
  return "phone";
}

/**
 * Set the page layout attribute.
 *
 * @param layout - The layout.
 * @param root - The element that carries it: the document's root element.
 */
export function setLayout(layout: Layout, root: HTMLElement = document.documentElement): void {
  root.setAttribute(LAYOUT_ATTRIBUTE, layout);
}

/**
 * The layout the root element carries now.
 *
 * @param root - The element that carries it.
 * @returns The layout; "phone" when none is set.
 */
export function currentLayout(root: HTMLElement = document.documentElement): Layout {
  const value = root.getAttribute(LAYOUT_ATTRIBUTE);
  return value === "two" || value === "four" ? value : "phone";
}

/** A mounted layout switch. */
export interface LayoutSwitch {
  /** The layout set now. */
  layout(): Layout;
  /** Stop following the viewport; the attribute keeps its last value. */
  stop(): void;
}

/**
 * Set the layout attribute from the viewport now, and again each time the
 * viewport crosses 760px or 1180px.
 *
 * @param root - The element that carries the attribute.
 * @param match - `matchMedia`; without one (no window) the layout is phone.
 * @returns The switch.
 */
export function mountLayout(
  root: HTMLElement = document.documentElement,
  match: MatchMedia | undefined = typeof window.matchMedia === "function"
    ? (query) => window.matchMedia(query)
    : undefined,
): LayoutSwitch {
  if (!match) {
    setLayout("phone", root);
    return { layout: () => currentLayout(root), stop: () => undefined };
  }
  const two = match(`(min-width: ${TWO_COLUMN_PX}px)`);
  const four = match(`(min-width: ${FOUR_COLUMN_PX}px)`);
  const apply = () => setLayout(four.matches ? "four" : two.matches ? "two" : "phone", root);
  apply();
  two.addEventListener("change", apply);
  four.addEventListener("change", apply);
  return {
    layout: () => currentLayout(root),
    stop() {
      two.removeEventListener("change", apply);
      four.removeEventListener("change", apply);
    },
  };
}
