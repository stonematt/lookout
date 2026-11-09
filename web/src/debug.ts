/**
 * The debug page (`/debug.html`): dumps the raw `/current` answer, Ambient's
 * `lastData` with the context block, for one station.
 *
 * The station is the page's `?station=`, as on the dashboard, the default
 * station without it; so `/debug.html?station=<MAC>` shows that station.
 * One `/current` read, which the CDN caches for a minute, so the page costs
 * Ambient no more than the dashboard does. The body is shown as received,
 * not parsed, and reaches the page through `textContent`.
 *
 * Offers `renderDump`.
 */
import "./styles/tokens.css";
import "./styles/base.css";

import { make } from "./dom.js";
import { READ_TIMEOUT_MS, readUrl } from "./load.js";

/**
 * The dump: which URL was read and how it answered, then its body as
 * indented JSON (or as text when it is not JSON).
 *
 * @param url - The URL read.
 * @param status - Its HTTP status.
 * @param body - Its body text.
 * @returns The elements to show, in order.
 */
export function renderDump(url: string, status: number, body: string): HTMLElement[] {
  let text = body;
  try {
    text = JSON.stringify(JSON.parse(body), null, 2);
  } catch {
    // Not JSON: show it as it came.
  }
  return [make("p", "debug-meta", `GET ${url} → ${status}`), make("pre", "debug-json", text)];
}

async function main(): Promise<void> {
  const root = document.getElementById("dump");
  if (!root) return;
  const url = readUrl("/current", window.location.search);
  try {
    const response = await fetch(url, {
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(READ_TIMEOUT_MS),
    });
    root.append(...renderDump(url, response.status, await response.text()));
  } catch (error) {
    root.append(make("p", "debug-error", `GET ${url} failed: ${String(error)}`));
  }
}

if (typeof document !== "undefined") void main();
