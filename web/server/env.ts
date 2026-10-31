/**
 * Build the read functions' sources from environment variables.
 *
 * Vercel sets these for Production and Preview (see `web/README.md`):
 * `STORJ_ACCESS_KEY_ID`, `STORJ_SECRET_ACCESS_KEY`, `STORJ_ENDPOINT_URL`,
 * `STORJ_BUCKET`, `STATION_MAC`, `AMBIENT_API_KEY`, `AMBIENT_APPLICATION_KEY`,
 * and optionally `AMBIENT_ENDPOINT`.
 */
import { DEFAULT_AMBIENT_ENDPOINT, ambientReader } from "./ambient.js";
import type { ReadSources } from "./handlers.js";
import { errorResponse, normaliseMac } from "./handlers.js";
import { storjReader } from "./storj.js";

/** What every read function needs. */
export const STORJ_NAMES = [
  "STORJ_ACCESS_KEY_ID",
  "STORJ_SECRET_ACCESS_KEY",
  "STORJ_ENDPOINT_URL",
  "STORJ_BUCKET",
  "STATION_MAC",
] as const;

/** What `/current` needs on top. */
export const AMBIENT_NAMES = ["AMBIENT_API_KEY", "AMBIENT_APPLICATION_KEY"] as const;

/** Environment variables, as `process.env` holds them. */
export type Env = Record<string, string | undefined>;

/** Some required environment variables are not set. */
export class MissingEnvError extends Error {
  constructor(readonly names: string[]) {
    super(`missing environment variables: ${names.join(", ")}`);
    this.name = "MissingEnvError";
  }
}

/**
 * The read sources the environment configures.
 *
 * @param env - The environment, usually `process.env`.
 * @param live - Whether `/current`'s Ambient source is needed.
 * @param fetchImpl - The fetch both sources call.
 * @returns The sources.
 * @throws MissingEnvError naming every unset variable.
 */
export function sourcesFromEnv(
  env: Env,
  live: boolean,
  fetchImpl: typeof fetch = fetch,
): ReadSources {
  const needed: readonly string[] = live ? [...STORJ_NAMES, ...AMBIENT_NAMES] : STORJ_NAMES;
  const missing = needed.filter((name) => !env[name]?.trim());
  if (missing.length) throw new MissingEnvError(missing);
  const value = (name: string) => env[name]!.trim();

  const defaultMac = normaliseMac(value("STATION_MAC"));
  if (!defaultMac) throw new MissingEnvError(["STATION_MAC (not a MAC address)"]);

  const sources: ReadSources = {
    defaultMac,
    readSnapshot: storjReader(
      {
        accessKeyId: value("STORJ_ACCESS_KEY_ID"),
        secretAccessKey: value("STORJ_SECRET_ACCESS_KEY"),
        endpointUrl: value("STORJ_ENDPOINT_URL"),
        bucket: value("STORJ_BUCKET"),
      },
      fetchImpl,
    ),
  };
  if (live) {
    sources.readLive = ambientReader(
      {
        apiKey: value("AMBIENT_API_KEY"),
        applicationKey: value("AMBIENT_APPLICATION_KEY"),
        endpoint: env.AMBIENT_ENDPOINT?.trim() || DEFAULT_AMBIENT_ENDPOINT,
      },
      fetchImpl,
    );
  }
  return sources;
}

/**
 * Run a handler with sources from the environment, answering 500 (naming
 * the unset variables, never their values) when the environment is short.
 *
 * @param request - The incoming request.
 * @param env - The environment.
 * @param live - Whether the handler needs Ambient.
 * @param handler - The handler to run.
 * @returns The handler's response, or the 500.
 */
export function serveFromEnv(
  request: Request,
  env: Env,
  live: boolean,
  handler: (request: Request, sources: ReadSources) => Promise<Response>,
): Promise<Response> {
  let sources: ReadSources;
  try {
    // Read `fetch` per request so a test's stub (or Vercel's) is the one used.
    sources = sourcesFromEnv(env, live, (input, init) => fetch(input, init));
  } catch (error) {
    if (error instanceof MissingEnvError) {
      console.error(error.message);
      return Promise.resolve(errorResponse(500, error.message));
    }
    throw error;
  }
  return handler(request, sources);
}
