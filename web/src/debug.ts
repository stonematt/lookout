/**
 * The debug page (`/debug.html`): dumps the raw `/current` answer, Ambient's
 * `lastData` with the context block, for one station.
 *
 * The station is the page's `?station=`, as on the dashboard, the default
 * station without it. One `/current` read, which the CDN caches for a
 * minute, so the page costs Ambient no more than the dashboard does.
 *
 * The body is never parsed against the snapshot contract: each top-level
 * block becomes a tile holding every key and value as received, epoch-ms
 * values get their local time beside them, and the whole body is kept as
 * indented JSON in a last, folded tile. Every value reaches the page through
 * `textContent`.
 *
 * A station selector in the header, drawn as the dashboard's, writes the
 * chosen `?station=` to the URL (a new history entry, so Back returns to the
 * last station) and reads `/current` again.
 *
 * Offers `renderDump`, `renderTree` and `stationPicker`.
 */
import "@fontsource-variable/lexend/wght.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/tiles.css";
import "./styles/debug.css";

import { make } from "./dom.js";
import { mountLayout } from "./layout.js";
import { READ_TIMEOUT_MS, fetchStations, readUrl } from "./load.js";
import type { StationIndex } from "./stations.js";
import { applyThemeOverride } from "./theme.js";

/** Blocks shown first, in this order; any other object block follows. */
const LEAD_BLOCKS = ["last_data", "context"];

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** An epoch-ms value worth a readable time: a `*_ms` key or `dateutc`. */
function isEpochKey(key: string, value: unknown): value is number {
  return typeof value === "number" && (key.endsWith("_ms") || key === "dateutc") && value > 1e12;
}

/**
 * A local time for an epoch-ms value, in the station's zone when known.
 *
 * @param ms - The time, epoch ms.
 * @param tz - The station's IANA zone, or undefined for the browser's.
 * @returns e.g. "Nov 9, 10:39:00 AM PST".
 */
function localTime(ms: number, tz: string | undefined): string {
  try {
    return new Date(ms).toLocaleString("en-US", {
      timeZone: tz,
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "short",
    });
  } catch {
    return new Date(ms).toISOString();
  }
}

/** One value's cell: the value as JSON writes it, styled by its type. */
function renderLeaf(key: string, value: unknown, tz: string | undefined): HTMLElement {
  const type = value === null ? "null" : typeof value;
  const cell = make("dd", `json-value json-${type}`, JSON.stringify(value));
  if (isEpochKey(key, value)) {
    cell.append(make("span", "json-time", localTime(value, tz)));
  }
  return cell;
}

/**
 * Every key and value of an object or array, nested ones folded open.
 *
 * @param value - The object or array.
 * @param tz - The station's zone, for epoch-ms values.
 * @returns A definition list, one row per key, in the body's order.
 */
export function renderTree(value: Record<string, unknown>, tz?: string): HTMLElement {
  const list = make("dl", "json-tree");
  for (const [key, child] of Object.entries(value)) {
    const row = make("div", "json-row");
    if (isObject(child)) {
      const fold = make("details", "json-fold");
      fold.open = true;
      const count = Object.keys(child).length;
      const summary = make("summary", "json-key", key);
      summary.append(
        make("span", "json-count", Array.isArray(child) ? `[${count}]` : `{${count}}`),
      );
      fold.append(summary, renderTree(child, tz));
      row.append(fold);
    } else {
      row.append(make("dt", "json-key", key), renderLeaf(key, child, tz));
    }
    list.append(row);
  }
  return list;
}

/** A tile with a kicker over its content. */
function tile(kicker: string, ...content: HTMLElement[]): HTMLElement {
  const section = make("section", "tile debug-tile");
  section.append(make("h2", "tile-kicker", kicker), ...content);
  return section;
}

/** The station's zone, from the context block, when the body carries one. */
function stationZone(body: Record<string, unknown>): string | undefined {
  const context = body.context;
  const station = isObject(context) ? context.station : undefined;
  const tz = isObject(station) ? station.tz : undefined;
  return typeof tz === "string" ? tz : undefined;
}

/** The station's name (or MAC) from the body, for the header. */
function stationName(body: unknown): string {
  const station = isObject(body) ? body.station : undefined;
  if (!isObject(station)) return "Raw /current";
  if (typeof station.name === "string") return station.name;
  return typeof station.mac === "string" ? station.mac : "Raw /current";
}

/** The MAC the body is for, upper case; null when it names none. */
function shownMac(body: unknown): string | null {
  const station = isObject(body) ? body.station : undefined;
  return isObject(station) && typeof station.mac === "string" ? station.mac.toUpperCase() : null;
}

/**
 * The station selector, as the dashboard's header draws it (the same
 * classes): one option per indexed station, the shown one selected.
 *
 * @param index - The station index.
 * @param shown - The MAC shown, or null.
 * @param onChoose - Called with the MAC chosen.
 * @returns The selector's wrapper; null for an index of one station.
 */
export function stationPicker(
  index: StationIndex,
  shown: string | null,
  onChoose: (mac: string) => void,
): HTMLElement | null {
  if (index.stations.length < 2) return null;
  const select = make("select", "station-select");
  select.setAttribute("aria-label", "Station");
  for (const entry of index.stations) {
    const option = make("option", "", entry.name ?? `Unnamed station …${entry.mac.slice(-5)}`);
    option.value = entry.mac;
    option.selected = entry.mac.toUpperCase() === shown;
    select.append(option);
  }
  select.addEventListener("change", () => onChoose(select.value));
  const wrap = make("span", "masthead-picks");
  wrap.append(select);
  return wrap;
}

/**
 * The page: a header naming the station (a selector when the index lists
 * two or more) and the read, then a tile for each object block (`last_data`
 * and `context` first), a tile of the document's own fields, and the whole
 * body as indented JSON in a folded tile.
 *
 * @param url - The URL read.
 * @param status - Its HTTP status.
 * @param bodyText - Its body text.
 * @param index - The station index, for the selector; null without one.
 * @param onChoose - Called with the MAC a visitor chooses.
 * @returns The header and the tiles' grid.
 */
export function renderDump(
  url: string,
  status: number,
  bodyText: string,
  index: StationIndex | null = null,
  onChoose: (mac: string) => void = () => undefined,
): HTMLElement[] {
  let body: unknown;
  let raw = bodyText;
  try {
    body = JSON.parse(bodyText);
    raw = JSON.stringify(body, null, 2);
  } catch {
    body = undefined;
  }

  const header = make("header", "masthead");
  const heading = make("h1", "masthead-station", stationName(body));
  const picker = index && stationPicker(index, shownMac(body), onChoose);
  if (picker) heading.replaceChildren(picker);
  header.append(heading);
  const line = make("p", "masthead-line debug-request", `GET ${url} · ${status}`);
  line.dataset.ok = String(status >= 200 && status < 300);
  header.append(line);

  const grid = make("main", "debug-grid");
  if (isObject(body) && !Array.isArray(body)) {
    const tz = stationZone(body);
    const blocks = Object.entries(body).filter(([, v]) => isObject(v));
    blocks.sort(([a], [b]) => rank(a) - rank(b));
    for (const [key, value] of blocks) {
      grid.append(tile(key, renderTree(value as Record<string, unknown>, tz)));
    }
    const fields = Object.fromEntries(Object.entries(body).filter(([, v]) => !isObject(v)));
    if (Object.keys(fields).length) grid.append(tile("document", renderTree(fields, tz)));
  }
  const fold = make("details", "json-raw");
  fold.append(make("summary", "", "Show the body as received"), make("pre", "json-pre", raw));
  grid.append(tile("raw JSON", fold));
  return [header, grid];
}

function rank(key: string): number {
  const at = LEAD_BLOCKS.indexOf(key);
  return at === -1 ? LEAD_BLOCKS.length : at;
}

/** The station index, read once; null when it cannot be read (no selector). */
let index: Promise<StationIndex | null> | null = null;

/** Read `/current` for the station the URL names, and draw it. */
async function show(root: HTMLElement): Promise<void> {
  index ??= fetchStations(location.search).catch(() => null);
  const url = readUrl("/current", location.search);
  let status = 0;
  let text: string;
  try {
    const response = await fetch(url, {
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(READ_TIMEOUT_MS),
    });
    status = response.status;
    text = await response.text();
  } catch (error) {
    text = `GET ${url} failed: ${String(error)}`;
  }
  // A later choice already started over: drop this answer.
  if (url !== readUrl("/current", location.search)) return;
  root.replaceChildren(...renderDump(url, status, text, await index, (mac) => choose(root, mac)));
}

/** A visitor chose a station: put it in the URL (Back returns) and show it. */
function choose(root: HTMLElement, mac: string): void {
  const url = new URL(location.href);
  url.searchParams.set("station", mac);
  history.pushState(null, "", `${url.pathname}${url.search}${url.hash}`);
  void show(root);
}

function main(): void {
  applyThemeOverride(location.search);
  mountLayout();
  const root = document.getElementById("dump");
  if (!root) return;
  window.addEventListener("popstate", () => void show(root));
  void show(root);
}

if (typeof document !== "undefined") main();
