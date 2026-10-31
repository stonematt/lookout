/**
 * Fetch the two read functions from the browser.
 *
 * The page's own query parameters `station` (a MAC address), and on the lane
 * server `fixture` and `age`, are passed on to both calls, so a shared link
 * shows the same station and a lane can open either golden at any age.
 */
import type { CurrentReading, DashboardSnapshot } from "./snapshot.js";
import { parseCurrent, parseDashboard } from "./snapshot.js";

const PASSED_ON = ["station", "fixture", "age"] as const;

/**
 * The URL of a read function, carrying the page's passed-on parameters.
 *
 * @param path - "/dashboard" or "/current".
 * @param pageSearch - The page's `location.search`.
 * @returns The path with its query.
 */
export function readUrl(path: "/dashboard" | "/current", pageSearch: string): string {
  const page = new URLSearchParams(pageSearch);
  const query = new URLSearchParams();
  for (const name of PASSED_ON) {
    const value = page.get(name);
    if (value !== null) query.set(name, value);
  }
  const text = query.toString();
  return text ? `${path}?${text}` : path;
}

/** GET a URL as JSON, rejecting on a non-2xx status. */
async function getJson(url: string, fetchImpl: typeof fetch): Promise<unknown> {
  const response = await fetchImpl(url, { headers: { accept: "application/json" } });
  if (!response.ok) throw new Error(`${url} answered ${response.status}`);
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
