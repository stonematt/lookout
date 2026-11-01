/**
 * Fetch the two read functions from the browser.
 *
 * The page's own query parameter `station` (a MAC address) is passed on to
 * every call as written: the read functions accept any spelling of the
 * configured station, answer 404 for another, and redirect to their one URL
 * (which leaves the configured station out), so a shared link keeps working.
 * The client cannot know the configured station, so it cannot skip that
 * redirect. `fixture` and `age` are passed on only by a lane server (see
 * `LANE_SERVER`): the deployed functions answer 400 to them.
 */
import type {
  CurrentReading,
  DashboardSnapshot,
  EventCurveDocument,
} from "./snapshot.js";
import { parseCurrent, parseDashboard, parseEventCurve } from "./snapshot.js";

/**
 * True for `npm run dev`, and for a build made with `VITE_LANE_SERVER=1` to
 * be served by `vite preview`: both run the golden read functions that take
 * `fixture` and `age`. False in the deployed build.
 */
export const LANE_SERVER: boolean =
  import.meta.env.DEV || import.meta.env.VITE_LANE_SERVER === "1";

const PASSED_ON = ["station"] as const;
const LANE_PASSED_ON = ["fixture", "age"] as const;

/**
 * The URL of a read function, carrying the page's passed-on parameters.
 *
 * @param path - "/dashboard", "/current" or "/event".
 * @param pageSearch - The page's `location.search`.
 * @param own - The call's own parameters, e.g. `/event`'s `start`.
 * @param laneServer - Whether the server takes `fixture` and `age`.
 * @returns The path with its query.
 */
export function readUrl(
  path: "/dashboard" | "/current" | "/event",
  pageSearch: string,
  own: Record<string, string> = {},
  laneServer: boolean = LANE_SERVER,
): string {
  const page = new URLSearchParams(pageSearch);
  const query = new URLSearchParams(own);
  for (const name of laneServer ? [...PASSED_ON, ...LANE_PASSED_ON] : PASSED_ON) {
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

/** GET a URL as JSON, rejecting with a `ReadError` on a non-2xx status. */
async function getJson(url: string, fetchImpl: typeof fetch): Promise<unknown> {
  const response = await fetchImpl(url, { headers: { accept: "application/json" } });
  if (!response.ok) throw new ReadError(url, response.status);
  return response.json();
}

/**
 * The snapshot, from `/dashboard`.
 *
 * @param pageSearch - The page's `location.search`.
 * @param fetchImpl - The fetch to use.
 * @returns The typed snapshot.
 */
export async function fetchDashboard(
  pageSearch: string,
  fetchImpl: typeof fetch = fetch,
): Promise<DashboardSnapshot> {
  return parseDashboard(await getJson(readUrl("/dashboard", pageSearch), fetchImpl));
}

/**
 * The live reading and context, from `/current`.
 *
 * @param pageSearch - The page's `location.search`.
 * @param fetchImpl - The fetch to use.
 * @returns The typed reading.
 */
export async function fetchCurrent(
  pageSearch: string,
  fetchImpl: typeof fetch = fetch,
): Promise<CurrentReading> {
  return parseCurrent(await getJson(readUrl("/current", pageSearch), fetchImpl));
}

/**
 * One completed event's curve file, from `/event`.
 *
 * @param startMs - The event's start, its key.
 * @param pageSearch - The page's `location.search`.
 * @param fetchImpl - The fetch to use.
 * @returns The typed curve; rejects with a `ReadError` (404: not written yet).
 */
export async function fetchEventCurve(
  startMs: number,
  pageSearch: string,
  fetchImpl: typeof fetch = fetch,
): Promise<EventCurveDocument> {
  const url = readUrl("/event", pageSearch, { start: String(startMs) });
  return parseEventCurve(await getJson(url, fetchImpl));
}
