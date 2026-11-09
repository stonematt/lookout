/**
 * The debug page (`/debug.html`): dumps Ambient's raw `lastData` for every
 * station in the station index, from one read of `/last-data`.
 *
 * Ambient allows one call a second per API key, shared with the dashboard's
 * `/current` and the refresh run. `/last-data` answers every station from
 * one Ambient `/devices` call and the CDN holds it for a minute, so opening
 * or reloading this page costs Ambient at most one call a minute. What it
 * shows is what Ambient holds now: the same `lastData` the refresh run reads
 * every five minutes. Every value reaches the page through `textContent`.
 *
 * Offers `parseLastData`, `renderStationDump` and `renderDump`.
 */
import "./styles/tokens.css";
import "./styles/base.css";

import { make } from "./dom.js";
import { READ_TIMEOUT_MS, ReadError, readUrl } from "./load.js";
import { SnapshotContractError, isRecord } from "./snapshot.js";

/** One station's raw reading, as `/last-data` serves it. */
export interface StationLastData {
  mac: string;
  name: string | null;
  /** Ambient's `lastData` as received; null when Ambient has none for it. */
  last_data: Record<string, unknown> | null;
}

/** The `/last-data` document (`server/handlers.ts`). */
export interface LastDataDocument {
  kind: "last_data";
  read_at_ms: number;
  stations: StationLastData[];
}

/**
 * Check a `/last-data` body's shape.
 *
 * @param doc - The parsed JSON.
 * @returns The typed document.
 * @throws SnapshotContractError when it is not one.
 */
export function parseLastData(doc: unknown): LastDataDocument {
  if (
    !isRecord(doc) ||
    doc.kind !== "last_data" ||
    typeof doc.read_at_ms !== "number" ||
    !Array.isArray(doc.stations) ||
    !doc.stations.every(
      (s) =>
        isRecord(s) &&
        typeof s.mac === "string" &&
        (s.name === null || typeof s.name === "string") &&
        (s.last_data === null || isRecord(s.last_data)),
    )
  ) {
    throw new SnapshotContractError("last_data: malformed document");
  }
  return doc as unknown as LastDataDocument;
}

/**
 * One station's section: its heading, when the reading was taken, and the
 * raw `lastData` as indented JSON.
 *
 * @param station - The station's entry in `/last-data`.
 * @returns The section element.
 */
export function renderStationDump(station: StationLastData): HTMLElement {
  const section = make("section", "debug-station");
  section.append(make("h2", "", `${station.name ?? "(unnamed)"} · ${station.mac}`));
  const data = station.last_data;
  const observed = data && typeof data.dateutc === "number" ? data.dateutc : null;
  section.append(
    make(
      "p",
      "debug-meta",
      observed === null ? "observed: unknown" : `observed: ${new Date(observed).toISOString()}`,
    ),
  );
  section.append(
    make(
      "pre",
      "debug-json",
      data === null
        ? "no lastData returned (Ambient is not listing this station)"
        : JSON.stringify(data, null, 2),
    ),
  );
  return section;
}

/**
 * The whole dump: when Ambient was read, then one section per station.
 *
 * @param doc - The `/last-data` document.
 * @returns The elements to show, in order.
 */
export function renderDump(doc: LastDataDocument): HTMLElement[] {
  const read = make(
    "p",
    "debug-meta",
    `Ambient read at ${new Date(doc.read_at_ms).toISOString()} (cached up to a minute)`,
  );
  return [read, ...doc.stations.map(renderStationDump)];
}

async function main(): Promise<void> {
  const root = document.getElementById("dump");
  if (!root) return;
  const url = readUrl("/last-data", window.location.search);
  try {
    const response = await fetch(url, {
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(READ_TIMEOUT_MS),
    });
    if (!response.ok) throw new ReadError(url, response.status);
    root.append(...renderDump(parseLastData(await response.json())));
  } catch (error) {
    root.append(make("p", "debug-error", `read failed: ${String(error)}`));
  }
}

if (typeof document !== "undefined") void main();
