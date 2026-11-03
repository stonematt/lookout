import { serveFromEnv } from "../server/env.js";
import { handleStations } from "../server/handlers.js";

/**
 * GET /stations (rewritten to /api/stations): the station index (#272).
 *
 * @param request - The incoming request.
 * @returns The response; see `handleStations`.
 */
export function GET(request: Request): Promise<Response> {
  return serveFromEnv(request, process.env, false, handleStations);
}
