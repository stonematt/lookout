/**
 * Fetch the read functions from the browser.
 *
 * The page's own query parameter `station` (a MAC address) is passed on to
 * every station read (`/dashboard`, `/current`, `/event`) as written: the
 * read functions serve any station the station index lists, the default
 * station without it, answer 404 for a MAC the index lacks, and redirect
 * any other spelling of a MAC to their one URL, so a shared link keeps
 * working. `/stations`, the station index itself, takes no station.
 * `fixture` and `age` are passed on only by a lane server (see
 * `LANE_SERVER`): the deployed functions answer 400 to them.
 */
import type { CurrentReading, DashboardSnapshot, EventCurveDocument } from "./snapshot.js";
import { parseCurrent, parseDashboard, parseEventCurve } from "./snapshot.js";
import type { StationIndex } from "./stations.js";
import { parseStationIndex } from "./stations.js";

/**
 * True for `npm run dev`, and for a build made with `VITE_LANE_SERVER=1` to
 * be served by `vite preview`: both run the golden read functions that take
 * `fixture` and `age`. False in the deployed build.
 */
export const LANE_SERVER: boolean = import.meta.env.DEV || import.meta.env.VITE_LANE_SERVER === "1";

const PASSED_ON = ["station"] as const;
const LANE_PASSED_ON = ["fixture", "age"] as const;

/**
 * The URL of a read function, carrying the page's passed-on parameters.
 *
 * @param path - "/dashboard", "/current", "/event" or "/stations" (which is
 *   not passed `station`).
 * @param pageSearch - The page's `location.search`.
 * @param own - The call's own parameters, e.g. `/event`'s `start`.
 * @param laneServer - Whether the server takes `fixture` and `age`.
 * @returns The path with its query.
 */
export function readUrl(
  path: "/dashboard" | "/current" | "/event" | "/stations",
  pageSearch: string,
  own: Record<string, string> = {},
  laneServer: boolean = LANE_SERVER,
): string {
  const page = new URLSearchParams(pageSearch);
  const query = new URLSearchParams(own);
  const passed = path === "/stations" ? [] : PASSED_ON;
  for (const name of laneServer ? [...passed, ...LANE_PASSED_ON] : passed) {
    const value = page.get(name);
    if (value !== null) query.set(name, value);
  }
  const text = query.toString();
  return text ? `${path}?${text}` : path;
}

/** A read function answered with a non-2xx status. */
export class ReadError extends Error {
  constructor(
    readonly url: string,
    readonly status: number,
  ) {
    super(`${url} answered ${status}`);
    this.name = "ReadError";
  }
}

/**
 * How long one client read may take before it fails. The read functions give
 * Storj 5 s and Ambient 5 s, so a healthy call answers well inside 15 s even
 * on a cold start; past it the call is hung, and failing lets the page's
 * refresh cycle run again instead of waiting forever.
 */
export const READ_TIMEOUT_MS = 15_000;

/**
 * GET a URL as JSON, rejecting with a `ReadError` on a non-2xx status, and
 * with the abort error when the call (headers or body) outlasts `timeoutMs`.
 */
async function getJson(url: string, fetchImpl: typeof fetch, timeoutMs: number): Promise<unknown> {
  const response = await fetchImpl(url, {
    headers: { accept: "application/json" },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) throw new ReadError(url, response.status);
  return response.json();
}

/**
 * The snapshot, from `/dashboard`.
 *
 * @param pageSearch - The page's `location.search`.
 * @param fetchImpl - The fetch to use.
 * @param timeoutMs - How long the call may take before it fails.
 * @returns The typed snapshot.
 */
export async function fetchDashboard(
  pageSearch: string,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = READ_TIMEOUT_MS,
): Promise<DashboardSnapshot> {
  return parseDashboard(await getJson(readUrl("/dashboard", pageSearch), fetchImpl, timeoutMs));
}

/**
 * The live reading and context, from `/current`.
 *
 * @param pageSearch - The page's `location.search`.
 * @param fetchImpl - The fetch to use.
 * @param timeoutMs - How long the call may take before it fails.
 * @returns The typed reading.
 */
export async function fetchCurrent(
  pageSearch: string,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = READ_TIMEOUT_MS,
): Promise<CurrentReading> {
  return parseCurrent(await getJson(readUrl("/current", pageSearch), fetchImpl, timeoutMs));
}

/**
 * One completed event's curve file, from `/event`.
 *
 * @param startMs - The event's start, its key.
 * @param pageSearch - The page's `location.search`.
 * @param fetchImpl - The fetch to use.
 * @param timeoutMs - How long the call may take before it fails.
 * @returns The typed curve; rejects with a `ReadError` (404: not written yet).
 */
export async function fetchEventCurve(
  startMs: number,
  pageSearch: string,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = READ_TIMEOUT_MS,
): Promise<EventCurveDocument> {
  const url = readUrl("/event", pageSearch, { start: String(startMs) });
  return parseEventCurve(await getJson(url, fetchImpl, timeoutMs));
}

/**
 * The station index, from `/stations`: the stations the site can show, in
 * order, the first being the default station.
 *
 * @param pageSearch - The page's `location.search`.
 * @param fetchImpl - The fetch to use.
 * @param timeoutMs - How long the call may take before it fails.
 * @returns The typed index.
 */
export async function fetchStations(
  pageSearch: string,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = READ_TIMEOUT_MS,
): Promise<StationIndex> {
  return parseStationIndex(await getJson(readUrl("/stations", pageSearch), fetchImpl, timeoutMs));
}
