/**
 * Vercel function behind `/last-data` (rewritten in `vercel.json`).
 * The logic lives in `server/handlers.ts`; this file only wires the env.
 */
import { serveFromEnv } from "../server/env.js";
import { handleLastData } from "../server/handlers.js";

export function GET(request: Request): Promise<Response> {
  return serveFromEnv(request, process.env, true, (req, sources) => handleLastData(req, sources));
}
