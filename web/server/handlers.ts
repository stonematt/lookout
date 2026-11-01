/**
 * The two read functions, free of Vercel and of `process.env`.
 *
 * - `/dashboard` returns the station's snapshot document as Storj holds it,
 *   cached by the CDN for 5 minutes (the refresh run's period).
 * - `/current` returns Ambient's live `lastData` with the context block, for
 *   the page header and any widget. Either half may be null when its source
 *   fails; only both failing is an error.
 *
 * Both take an optional `station` query parameter (a MAC address) that
 * defaults to the configured station. Only the configured station is served:
 * any other MAC is a 404, answered before any read and never cached, so a
 * caller cannot read another station's snapshot or fill the CDN with keys.
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

/** `/dashboard`: the CDN serves it; Storj is read about once per 5 minutes. */
export const DASHBOARD_CACHE = "s-maxage=300, stale-while-revalidate=300";

/** `/current`: short, so the live reading stays live and Ambient is spared. */
export const CURRENT_CACHE = "s-maxage=60, stale-while-revalidate=60";

/** A `/current` missing one half is not cached, so a blip is not served on. */
export const PARTIAL_CACHE = "no-store";

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
 * An error response, never cached.
 *
 * @param status - The HTTP status.
 * @param error - What went wrong, safe to show anyone.
 * @returns The response.
 */
export function errorResponse(status: number, error: string): Response {
  return jsonResponse({ error }, status, "no-store");
}

/**
 * The station a request asks for: the configured one, whether named by the
 * `station` parameter (in either case) or left to the default.
 *
 * @param request - The incoming request.
 * @param sources - Holds the configured station.
 * @returns The upper-case MAC; a 400 response for a malformed one; a 404,
 *   not cached, for any other station.
 */
export function stationOf(request: Request, sources: ReadSources): string | Response {
  const asked = new URL(request.url).searchParams.get("station");
  if (asked === null) return sources.defaultMac;
  const mac = normaliseMac(asked);
  if (mac === null) return errorResponse(400, "station must be a MAC address");
  if (mac !== sources.defaultMac) return errorResponse(404, "no such station");
  return mac;
}

/**
 * GET /dashboard: the snapshot document.
 *
 * @param request - The incoming request; `?station=<MAC>` is optional.
 * @param sources - Where to read.
 * @returns 200 with the document, 404 without one, 502 when Storj fails or
 *   the document breaks the contract.
 */
export async function handleDashboard(
  request: Request,
  sources: ReadSources,
): Promise<Response> {
  const mac = stationOf(request, sources);
  if (mac instanceof Response) return mac;
  let text: string | null;
  try {
    text = await sources.readSnapshot(mac, "dashboard");
  } catch (error) {
    console.error("dashboard: Storj read failed", describeError(error));
    return errorResponse(502, "the snapshot could not be read");
  }
  if (text === null) return errorResponse(404, `no snapshot for station ${mac}`);
  try {
    parseDashboard(JSON.parse(text));
  } catch (error) {
    console.error("dashboard: snapshot breaks the contract", describeError(error));
    return errorResponse(502, "the snapshot does not match this site");
  }
  return jsonResponse(text, 200, DASHBOARD_CACHE);
}

/**
 * GET /current: the live reading plus the context block.
 *
 * Cached for a minute when both halves were read; not cached when one is
 * missing.
 *
 * @param request - The incoming request; `?station=<MAC>` is optional.
 * @param sources - Where to read; needs `readLive`.
 * @returns 200 with whatever could be read, 502 when nothing could.
 */
export async function handleCurrent(
  request: Request,
  sources: ReadSources,
): Promise<Response> {
  const mac = stationOf(request, sources);
  if (mac instanceof Response) return mac;
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
    return errorResponse(502, "neither the live reading nor the snapshot could be read");
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
  return jsonResponse(body, 200, whole ? CURRENT_CACHE : PARTIAL_CACHE);
}
