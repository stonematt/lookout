/**
 * The debug page (`/debug.html`): dumps the raw Ambient `lastData` for every
 * station in the station index, as `/current` returns it.
 *
 * `/current` reads Ambient live (cached for a minute), so this is what Ambient
 * holds now, the same `lastData` the refresh run reads every five minutes.
 * Stations are read one after another, not in parallel, to stay inside
 * Ambient's rate limit. Every value reaches the page through `textContent`.
 *
 * Offers `debugSearch` and `renderStationDump`.
 */
import "./styles/tokens.css";
import "./styles/base.css";

import { make } from "./dom.js";
import { fetchCurrent, fetchStations } from "./load.js";
import type { CurrentReading } from "./snapshot.js";
import type { StationEntry } from "./stations.js";

/**
 * The query to read one station with: the page's own query (so a lane
 * server's `fixture` and `age` pass on) with `station` set to its MAC.
 *
 * @param pageSearch - The page's `location.search`.
 * @param mac - The station's MAC.
 * @returns A `?`-prefixed query.
 */
export function debugSearch(pageSearch: string, mac: string): string {
  const query = new URLSearchParams(pageSearch);
  query.set("station", mac);
  return `?${query.toString()}`;
}

/**
 * One station's section: its heading, when the reading was observed, and
 * the raw `last_data` (or the read error) as indented JSON.
 *
 * @param station - The station's index entry.
 * @param result - Its `/current` reading, or the error the read failed with.
 * @returns The section element.
 */
export function renderStationDump(
  station: StationEntry,
  result: CurrentReading | Error,
): HTMLElement {
  const section = make("section", "debug-station");
  section.append(make("h2", "", `${station.name ?? "(unnamed)"} · ${station.mac}`));
  if (result instanceof Error) {
    section.append(make("p", "debug-error", `read failed: ${result.message}`));
    return section;
  }
  const observed = result.observed_at_ms;
  section.append(
    make(
      "p",
      "debug-meta",
      observed === null
        ? "observed: unknown"
        : `observed: ${new Date(observed).toISOString()} (${station.tz})`,
    ),
  );
  section.append(
    make(
      "pre",
      "debug-json",
      result.last_data === null
        ? "no lastData returned (Ambient unreachable or not listing this station)"
        : JSON.stringify(result.last_data, null, 2),
    ),
  );
  return section;
}

async function main(): Promise<void> {
  const root = document.getElementById("dump");
  if (!root) return;
  const search = window.location.search;
  let stations: StationEntry[];
  try {
    stations = (await fetchStations(search)).stations;
  } catch (error) {
    root.append(make("p", "debug-error", `station index read failed: ${String(error)}`));
    return;
  }
  for (const station of stations) {
    let result: CurrentReading | Error;
    try {
      result = await fetchCurrent(debugSearch(search, station.mac));
    } catch (error) {
      result = error instanceof Error ? error : new Error(String(error));
    }
    root.append(renderStationDump(station, result));
  }
}

if (typeof document !== "undefined") void main();
