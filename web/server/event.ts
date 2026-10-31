/**
 * The third read function, `/event?start=<ms>`: one completed rain event's
 * curve file, as the refresh run wrote it to Storj.
 *
 * A curve file is written once and never changes, so a found one is cached
 * by the CDN for a year. A missing one is a completed event the bounded
 * backfill has not reached yet (or an event with no archive data); it is
 * cached briefly so the next run's write shows up soon.
 */
import { parseEventCurve } from "../src/snapshot.js";
import type { ReadSources } from "./handlers.js";
import { errorResponse, jsonResponse, stationOf } from "./handlers.js";

/** A found curve file: immutable. */
export const CURVE_CACHE = "public, max-age=86400, s-maxage=31536000, immutable";

/** No curve file yet: the next refresh run may write it. */
export const MISSING_CURVE_CACHE = "s-maxage=300";

const START_PATTERN = /^\d{12,14}$/;

/**
 * GET /event: one event's curve file.
 *
 * @param request - The incoming request; `?start=<ms>` names the event,
 *   `?station=<MAC>` is optional.
 * @param sources - Where to read; needs `readCurve`.
 * @returns 200 with the curve, 400 for a malformed start, 404 when it is
 *   not written, 502 when Storj fails or the file breaks the contract.
 */
export async function handleEvent(
  request: Request,
  sources: ReadSources,
): Promise<Response> {
  const mac = stationOf(request, sources);
  if (mac instanceof Response) return mac;
  const start = new URL(request.url).searchParams.get("start") ?? "";
  if (!START_PATTERN.test(start)) {
    return errorResponse(400, "start must be an event start in epoch milliseconds");
  }
  const readCurve = sources.readCurve;
  if (!readCurve) return errorResponse(500, "no curve source configured");

  let text: string | null;
  try {
    text = await readCurve(mac, Number(start));
  } catch (error) {
    console.error("event: Storj read failed", error);
    return errorResponse(502, "the event curve could not be read");
  }
  if (text === null) {
    return jsonResponse(
      { error: `no curve for the event starting ${start}` },
      404,
      MISSING_CURVE_CACHE,
    );
  }
  try {
    parseEventCurve(JSON.parse(text));
  } catch (error) {
    console.error("event: curve breaks the contract", error);
    return errorResponse(502, "the event curve does not match this site");
  }
  return jsonResponse(text, 200, CURVE_CACHE);
}
