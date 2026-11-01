/**
 * The two read functions, free of Vercel and of `process.env`.
 *
 * - `/dashboard` returns the station's snapshot document as Storj holds it,
 *   cached by the CDN for 5 minutes (the refresh run's period).
 * - `/current` returns Ambient's live `lastData` with the context block, for
 *   the page header and any widget. Either half may be null when its source
 *   fails; only both failing is an error.
 *
 * The CDN keys its cache on the whole URL, so each read function reads its
 * sources only at one spelling of its URL: the bare path (and `/event`'s
 * `start`). An unknown or repeated query parameter is a 400 and another
 * spelling of the same request is a 308 to the one URL, both answered before
 * any read, so a caller cannot get past the cache with a query string.
 *
 * The optional `station` parameter (a MAC address) is kept for shared links:
 * the configured station, in any case, redirects to the URL without it; any
 * other MAC is a 404, never cached, so a caller cannot read another
 * station's snapshot or fill the CDN with keys.
 */
import type { CurrentReading, LastData, Station } from "../src/snapshot.js";
import {
  SCHEMA_VERSION,
  newestObservation,
  parseContext,
  parseDashboard,
} from "../src/snapshot.js";
import type { LiveReader } from "./ambient.js";
import { describeError } from "./errors.js";
import type { CurveReader, SnapshotReader } from "./storj.js";

/**
 * `/dashboard`: the CDN serves it; Storj is read about once per 5 minutes.
 * While a read fails, the CDN may serve the last good copy for 5 more.
 */
export const DASHBOARD_CACHE =
  "s-maxage=300, stale-while-revalidate=300, stale-if-error=300";

/** `/current`: short, so the live reading stays live and Ambient is spared. */
export const CURRENT_CACHE = "s-maxage=60, stale-while-revalidate=60, stale-if-error=300";

/**
 * A failed or partial read (an Ambient 429 included) is cached briefly, so
 * that during an outage or a rate-limit episode the sources are asked about
 * once per 15 seconds per region, not once per visitor. Vercel's CDN does
 * not cache a 5xx; for those, `stale-if-error` on the good copy stands in.
 */
export const FAILURE_CACHE = "s-maxage=15";

/** A redirect to a read function's one URL never changes. */
export const CANONICAL_CACHE = "s-maxage=3600";

/**
 * Check one query parameter's value.
 *
 * @returns Its one spelling, or the error response for a bad value.
 */
export type ParamCheck = (value: string) => string | Response;

/** What the read functions read from. */
export interface ReadSources {
  /** The configured station's MAC address, upper case. */
  defaultMac: string;
  readSnapshot: SnapshotReader;
  /** Only `/current` needs it. */
  readLive?: LiveReader;
  /** Only `/event` needs it (`server/event.ts`). */
  readCurve?: CurveReader;
}

const MAC_PATTERN = /^[0-9A-F]{2}(:[0-9A-F]{2}){5}$/;

/**
 * Normalise a MAC address to upper case, or null when it is not one.
 *
 * @param value - A MAC address in either case.
 * @returns The upper-case MAC, or null.
 */
export function normaliseMac(value: string): string | null {
  const mac = value.trim().toUpperCase();
  return MAC_PATTERN.test(mac) ? mac : null;
}

/**
 * A JSON response.
 *
 * @param body - Anything JSON-serialisable.
 * @param status - The HTTP status.
 * @param cache - The Cache-Control value.
 * @returns The response.
 */
export function jsonResponse(body: unknown, status: number, cache: string): Response {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return new Response(text, {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": cache,
    },
  });
}

/**
 * An error response, not cached unless `cache` says so.
 *
 * @param status - The HTTP status.
 * @param error - What went wrong, safe to show anyone.
 * @param cache - The Cache-Control value; `FAILURE_CACHE` for a failed read.
 * @returns The response.
 */
export function errorResponse(status: number, error: string, cache = "no-store"): Response {
  return jsonResponse({ error }, status, cache);
}

/**
 * Hold a request to its read function's one URL, before anything is read.
 *
 * Only `station` and the parameters in `checks` are allowed, each at most
 * once. `station` must name the configured station and is then dropped:
 * the configured one is the default.
 *
 * @param request - The incoming request.
 * @param sources - Holds the configured station.
 * @param checks - The function's own parameters, in the order its one URL
 *   carries them, each with its check.
 * @returns The checked parameters, by name; or a response to send instead:
 *   400 for an unknown, repeated or malformed parameter, 404 for another
 *   station (neither cached), 308 to the one URL for any other spelling.
 */
export function checkQuery(
  request: Request,
  sources: ReadSources,
  checks: Record<string, ParamCheck> = {},
): Record<string, string> | Response {
  const url = new URL(request.url);
  const seen = new Set<string>();
  for (const [name, value] of url.searchParams) {
    if (name !== "station" && !Object.hasOwn(checks, name)) {
      return errorResponse(400, "unknown query parameter");
    }
    if (seen.has(name)) return errorResponse(400, `${name} given more than once`);
    seen.add(name);
    if (name !== "station") continue;
    const mac = normaliseMac(value);
    if (mac === null) return errorResponse(400, "station must be a MAC address");
    if (mac !== sources.defaultMac) return errorResponse(404, "no such station");
  }

  const params: Record<string, string> = {};
  const one = new URLSearchParams();
  for (const [name, check] of Object.entries(checks)) {
    const value = url.searchParams.get(name);
    if (value === null) continue;
    const checked = check(value);
    if (checked instanceof Response) return checked;
    params[name] = checked;
    one.set(name, checked);
  }
  const query = one.toString();
  if (query !== url.search.slice(1)) {
    // Relative, so it resolves against the path the caller asked for.
    const path = url.pathname.split("/").pop() ?? "";
    const location = query ? `${path}?${query}` : path;
    return new Response(null, {
      status: 308,
      headers: { location, "cache-control": CANONICAL_CACHE },
    });
  }
  return params;
}

/**
 * GET /dashboard: the snapshot document.
 *
 * @param request - The incoming request; see `checkQuery`.
 * @param sources - Where to read.
 * @returns 200 with the document, 404 without one, 502 when Storj fails or
 *   the document breaks the contract.
 */
export async function handleDashboard(
  request: Request,
  sources: ReadSources,
): Promise<Response> {
  const query = checkQuery(request, sources);
  if (query instanceof Response) return query;
  const mac = sources.defaultMac;
  let text: string | null;
  try {
    text = await sources.readSnapshot(mac, "dashboard");
  } catch (error) {
    console.error("dashboard: Storj read failed", describeError(error));
    return errorResponse(502, "the snapshot could not be read", FAILURE_CACHE);
  }
  if (text === null) {
    return errorResponse(404, `no snapshot for station ${mac}`, FAILURE_CACHE);
  }
  try {
    parseDashboard(JSON.parse(text));
  } catch (error) {
    console.error("dashboard: snapshot breaks the contract", describeError(error));
    return errorResponse(502, "the snapshot does not match this site", FAILURE_CACHE);
  }
  return jsonResponse(text, 200, DASHBOARD_CACHE);
}

/**
 * GET /current: the live reading plus the context block.
 *
 * Cached for a minute when both halves were read; for 15 seconds when one
 * is missing, so a blip is not served on for long.
 *
 * @param request - The incoming request; see `checkQuery`.
 * @param sources - Where to read; needs `readLive`.
 * @returns 200 with whatever could be read, 502 when nothing could.
 */
export async function handleCurrent(
  request: Request,
  sources: ReadSources,
): Promise<Response> {
  const query = checkQuery(request, sources);
  if (query instanceof Response) return query;
  const mac = sources.defaultMac;
  const readLive = sources.readLive;
  if (!readLive) return errorResponse(500, "no live source configured");

  const [contextResult, liveResult] = await Promise.allSettled([
    sources.readSnapshot(mac, "context").then((text) =>
      text === null ? null : parseContext(JSON.parse(text)),
    ),
    readLive(mac),
  ]);
  if (contextResult.status === "rejected") {
    console.error("current: context read failed", describeError(contextResult.reason));
  }
  if (liveResult.status === "rejected") {
    console.error("current: Ambient read failed", describeError(liveResult.reason));
  }
  if (contextResult.status === "rejected" && liveResult.status === "rejected") {
    return errorResponse(
      502,
      "neither the live reading nor the snapshot could be read",
      FAILURE_CACHE,
    );
  }

  const context = contextResult.status === "fulfilled" ? contextResult.value : null;
  const live = liveResult.status === "fulfilled" ? liveResult.value : null;
  const lastData: LastData | null = live?.lastData ?? null;
  const station: Station = {
    mac,
    name: context?.station.name ?? live?.name ?? null,
  };
  const body: CurrentReading = {
    schema_version: SCHEMA_VERSION,
    kind: "current",
    station,
    observed_at_ms: newestObservation(lastData?.dateutc, context?.observed_at_ms),
    built_at_ms: context?.built_at_ms ?? null,
    last_data: lastData,
    context,
  };
  const whole = context !== null && lastData !== null;
  return jsonResponse(body, 200, whole ? CURRENT_CACHE : FAILURE_CACHE);
}
