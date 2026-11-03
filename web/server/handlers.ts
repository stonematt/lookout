/**
 * The read functions, free of Vercel and of `process.env` (`/event` is in
 * `event.ts`).
 *
 * - `/dashboard` returns the station's snapshot document as Storj holds it,
 *   cached by the CDN for 5 minutes (the refresh run's period).
 * - `/current` returns Ambient's live `lastData` with the context block, for
 *   the page header and any widget. Either half may be null when its source
 *   fails; only both failing is an error.
 * - `/status` returns the refresh run's status document (#268): how the last
 *   run ended, when, and the code version and git SHA it ran on. The page
 *   never reads it; it is for a post-release check.
 * - `/stations` returns the station index (#272): the stations the site can
 *   show, in inventory order, the first being the default station.
 *
 * The CDN keys its cache on the whole URL, so each read function reads its
 * sources only at one spelling of its URL: the bare path, plus its own
 * parameters (`station`, and `/event`'s `start`) in one spelling. An unknown
 * or repeated query parameter is a 400 and another spelling of the same
 * request is a 308 to the one URL, both answered before any read, so a
 * caller cannot get past the cache with a query string.
 *
 * `/dashboard`, `/current`, `/event` and `/status` serve one station: the
 * one `?station=<MAC>` names, or the default station (the station index's
 * first) without it. A malformed MAC is a 400; a MAC the station index does
 * not list is a 404, never cached, so a caller cannot fill the CDN with keys;
 * and the index is read at most once a minute per function instance
 * (`storjStationsReader`), so neither can they drive Storj reads with them.
 * The default station named outright is served at its own URL, not
 * redirected to the bare path: which station is the default is the index's
 * call, and can change, so it is never baked into a cached redirect.
 */
import type { CurrentReading, LastData, StationId } from "../src/snapshot.js";
import {
  SCHEMA_VERSION,
  newestObservation,
  parseContext,
  parseDashboard,
} from "../src/snapshot.js";
import { findStation } from "../src/stations.js";
import type { LiveReader } from "./ambient.js";
import { describeError } from "./errors.js";
import type { CurveReader, SnapshotReader, StationsReader } from "./storj.js";

/**
 * `/dashboard`: the CDN serves it; Storj is read about once per 5 minutes.
 * While a read fails, the CDN may serve the last good copy for 5 more.
 */
export const DASHBOARD_CACHE = "s-maxage=300, stale-while-revalidate=300, stale-if-error=300";

/** `/current`: short, so the live reading stays live and Ambient is spared. */
export const CURRENT_CACHE = "s-maxage=60, stale-while-revalidate=60, stale-if-error=300";

/** `/status`: short, so a check after a run sees that run within a minute. */
export const STATUS_CACHE = "s-maxage=60, stale-if-error=300";

/** `/stations`: the refresh run rewrites it every 5 minutes, as the dashboard. */
export const STATIONS_CACHE = DASHBOARD_CACHE;

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
  /** The station index: which stations are served, and the default one. */
  readStations: StationsReader;
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
 * `station`'s one spelling: the MAC address in upper case.
 *
 * @param value - As the caller wrote it.
 * @returns The upper-case MAC, or a 400 when it is not a MAC address.
 */
export function checkStation(value: string): string | Response {
  return normaliseMac(value) ?? errorResponse(400, "station must be a MAC address");
}

/**
 * Hold a request to its read function's one URL, before anything is read.
 *
 * Only the parameters in `checks` are allowed, each at most once.
 *
 * @param request - The incoming request.
 * @param checks - The function's own parameters, in the order its one URL
 *   carries them, each with its check.
 * @returns The checked parameters, by name; or a response to send instead:
 *   400 (not cached) for an unknown, repeated or malformed parameter, 308
 *   to the one URL for any other spelling.
 */
export function checkQuery(
  request: Request,
  checks: Record<string, ParamCheck> = {},
): Record<string, string> | Response {
  const url = new URL(request.url);
  const seen = new Set<string>();
  for (const [name] of url.searchParams) {
    if (!Object.hasOwn(checks, name)) return errorResponse(400, "unknown query parameter");
    if (seen.has(name)) return errorResponse(400, `${name} given more than once`);
    seen.add(name);
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
    // Relative, so it resolves against the path the caller asked for (a
    // rewrite may show this function a different pathname). Never empty:
    // the segment is "" after a trailing slash, and "" resolves to the URL
    // just asked for, which would loop.
    const path = url.pathname.split("/").pop() || "./";
    const location = query ? `${path}?${query}` : path;
    return new Response(null, {
      status: 308,
      headers: { location, "cache-control": CANONICAL_CACHE },
    });
  }
  return params;
}

/**
 * The station a read is for: the checked `station` parameter when the
 * station index lists it, else the index's first, the default station.
 *
 * @param station - The checked `station` parameter; undefined when absent.
 * @param sources - Holds the station index.
 * @param name - The read function, for the log line.
 * @returns Its MAC, as the index spells it; or a response to send instead:
 *   404 (not cached) for a station the index does not list, 404 (briefly
 *   cached) when there is no index yet, 502 (briefly cached) when it cannot
 *   be read.
 */
export async function resolveStation(
  station: string | undefined,
  sources: ReadSources,
  name: string,
): Promise<string | Response> {
  let index;
  try {
    index = await sources.readStations();
  } catch (error) {
    console.error(`${name}: station index read failed`, describeError(error));
    return errorResponse(502, "the station index could not be read", FAILURE_CACHE);
  }
  if (index === null) return errorResponse(404, "no station index yet", FAILURE_CACHE);
  if (station === undefined) return index.stations[0]!.mac;
  return findStation(index, station)?.mac ?? errorResponse(404, "no such station");
}

/**
 * Check a station read's query and resolve its station.
 *
 * @param request - The incoming request.
 * @param sources - Holds the station index.
 * @param name - The read function, for the log line.
 * @param checks - The function's own parameters besides `station`.
 * @returns The station's MAC and the checked parameters, or a response to
 *   send instead; see `checkQuery` and `resolveStation`.
 */
export async function stationQuery(
  request: Request,
  sources: ReadSources,
  name: string,
  checks: Record<string, ParamCheck> = {},
): Promise<{ mac: string; params: Record<string, string> } | Response> {
  const params = checkQuery(request, { ...checks, station: checkStation });
  if (params instanceof Response) return params;
  const mac = await resolveStation(params.station, sources, name);
  return mac instanceof Response ? mac : { mac, params };
}

/**
 * GET /dashboard: the snapshot document.
 *
 * @param request - The incoming request; see `checkQuery`.
 * @param sources - Where to read.
 * @returns 200 with the document, 404 without one, 502 when Storj fails or
 *   the document breaks the contract.
 */
export async function handleDashboard(request: Request, sources: ReadSources): Promise<Response> {
  const query = await stationQuery(request, sources, "dashboard");
  if (query instanceof Response) return query;
  const mac = query.mac;
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
 * GET /status: the refresh run's status document, as Storj holds it.
 *
 * It is the refresh run's own document, outside the snapshot contract, so
 * it is checked only for being a status document (`kind: "status"`).
 *
 * @param request - The incoming request; see `checkQuery`.
 * @param sources - Where to read.
 * @returns 200 with the document; 404 when no run has written one; 502 when
 *   Storj fails or holds something else.
 */
export async function handleStatus(request: Request, sources: ReadSources): Promise<Response> {
  const query = await stationQuery(request, sources, "status");
  if (query instanceof Response) return query;
  const mac = query.mac;
  let text: string | null;
  try {
    text = await sources.readSnapshot(mac, "status");
  } catch (error) {
    console.error("status: Storj read failed", describeError(error));
    return errorResponse(502, "the status could not be read", FAILURE_CACHE);
  }
  if (text === null) {
    return errorResponse(404, `no status for station ${mac}`, FAILURE_CACHE);
  }
  if (!isStatusDocument(text)) {
    console.error("status: the stored document is not a status document");
    return errorResponse(502, "the status is not a status document", FAILURE_CACHE);
  }
  return jsonResponse(text, 200, STATUS_CACHE);
}

/**
 * Whether a document's text is a status document.
 *
 * @param text - The stored text.
 * @returns True for a JSON object whose `kind` is "status".
 */
function isStatusDocument(text: string): boolean {
  try {
    const doc: unknown = JSON.parse(text);
    return typeof doc === "object" && doc !== null && (doc as { kind?: unknown }).kind === "status";
  } catch {
    return false;
  }
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
export async function handleCurrent(request: Request, sources: ReadSources): Promise<Response> {
  const readLive = sources.readLive;
  if (!readLive) return errorResponse(500, "no live source configured");
  const query = await stationQuery(request, sources, "current");
  if (query instanceof Response) return query;
  const mac = query.mac;

  const [contextResult, liveResult] = await Promise.allSettled([
    sources
      .readSnapshot(mac, "context")
      .then((text) => (text === null ? null : parseContext(JSON.parse(text)))),
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
  const station: StationId = {
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

/**
 * GET /stations: the station index, for the page (#272).
 *
 * @param request - The incoming request; it takes no parameters.
 * @param sources - Where to read.
 * @returns 200 with the index; 404, briefly cached, before the refresh run
 *   has published one; 502, briefly cached, when it cannot be read or
 *   breaks the contract.
 */
export async function handleStations(request: Request, sources: ReadSources): Promise<Response> {
  const query = checkQuery(request);
  if (query instanceof Response) return query;
  let index;
  try {
    index = await sources.readStations();
  } catch (error) {
    console.error("stations: station index read failed", describeError(error));
    return errorResponse(502, "the station index could not be read", FAILURE_CACHE);
  }
  if (index === null) return errorResponse(404, "no station index yet", FAILURE_CACHE);
  return jsonResponse(index, 200, STATIONS_CACHE);
}
