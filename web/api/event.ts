/**
 * Vercel function behind `/event` (rewritten in `vercel.json`).
 * The logic lives in `server/event.ts`; this file only wires the env.
 */
import { serveFromEnv } from "../server/env.js";
import { handleEvent } from "../server/event.js";

export function GET(request: Request): Promise<Response> {
  return serveFromEnv(request, process.env, false, handleEvent);
}
