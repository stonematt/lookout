/**
 * A stand-in `matchMedia` for tests: happy-dom evaluates no media queries,
 * so the page's layout switch (`layout.ts`) and an event chart's colour
 * scheme watch get this one instead.
 *
 * A `min-width` query matches while the viewport is at least that wide; any
 * other query (the colour scheme) always matches. Offers `viewport`.
 */
import type { MatchMedia } from "../src/layout.js";

/** A stand-in viewport a test moves. */
export interface Viewport {
  /** The stand-in `matchMedia`. */
  match: MatchMedia;
  /**
   * Set `window.matchMedia` to `match`; the test restores the real one.
   *
   * @returns This viewport.
   */
  install(): Viewport;
  /** Move the viewport to `to` px wide, firing each threshold it crosses. */
  resize(to: number): void;
  /** Fire every query that is not a width (a colour-scheme change). */
  scheme(): void;
}

/**
 * A stand-in viewport.
 *
 * @param width - The viewport's starting width in px.
 * @returns The viewport.
 */
export function viewport(width: number): Viewport {
  const lists: { min: number | null; list: MediaQueryList; listeners: Set<() => void> }[] = [];
  const match: MatchMedia = (query) => {
    const found = /min-width:\s*(\d+)px/.exec(query);
    const min = found ? Number(found[1]) : null;
    const listeners = new Set<() => void>();
    const list = {
      get matches() {
        return min === null || width >= min;
      },
      addEventListener: (_: string, fn: () => void) => listeners.add(fn),
      removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
    } as unknown as MediaQueryList;
    lists.push({ min, list, listeners });
    return list;
  };
  const screen: Viewport = {
    match,
    install() {
      window.matchMedia = match as typeof window.matchMedia;
      return screen;
    },
    resize(to) {
      const before = lists.map(({ list }) => list.matches);
      width = to;
      lists.forEach(({ list, listeners }, i) => {
        if (list.matches !== before[i]) for (const fn of listeners) fn();
      });
    },
    scheme() {
      for (const { min, listeners } of lists) if (min === null) for (const fn of listeners) fn();
    },
  };
  return screen;
}
