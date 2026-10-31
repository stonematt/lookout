/**
 * Light and dark follow the OS (`prefers-color-scheme`, in the tokens). An
 * explicit `?theme=light` or `?theme=dark` is honoured over it.
 */

/**
 * Apply an explicit theme from the page's query, if it names one.
 *
 * @param pageSearch - The page's `location.search`.
 * @param root - The element the tokens key off.
 */
export function applyThemeOverride(
  pageSearch: string,
  root: HTMLElement = document.documentElement,
): void {
  const theme = new URLSearchParams(pageSearch).get("theme");
  if (theme === "light" || theme === "dark") root.dataset.theme = theme;
}
