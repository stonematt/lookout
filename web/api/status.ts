import { serveFromEnv } from "../server/env.js";
import { handleStatus } from "../server/handlers.js";

/**
 * GET /status (rewritten to /api/status): the refresh run's status document.
 *
 * @param request - The incoming request.
 * @returns The response; see `handleStatus`.
 */
export function GET(request: Request): Promise<Response> {
  return serveFromEnv(request, process.env, false, handleStatus);
}
