/**
 * The site's release: the repo's release version (the root `VERSION`
 * file, `YYYY.MM.N`, ADR-0007) and the short git SHA Vercel builds from
 * (`VERCEL_GIT_COMMIT_SHA`, "local" without it), and `/api/v1/version`.
 *
 * The page gets both as build-time constants: `vite.config.ts` calls
 * `releaseDefine`, which reads `VERSION` from the repo root (`../VERSION`
 * from `web/`; Vercel's build can read files outside the Root Directory
 * unless the project turns that off) and throws when it cannot, so a build
 * never ships an unknown version. `src/release.ts` reads the constants.
 *
 * The function `/api/v1/version` reads the same file at run time: Vercel's
 * Node builder traces the static `new URL("../../VERSION", import.meta.url)`
 * read and ships the file with the function, and `VERCEL_GIT_COMMIT_SHA` is
 * set at run time too, so both answer from the deployed commit.
 *
 * Offers `Release`, `ReleaseError`, `VERSION_FILE`, `VERSION_CACHE`,
 * `parseVersion`, `shortSha`, `readRelease`, `releaseDefine` and
 * `handleVersion`.
 */
import { readFileSync } from "node:fs";

import { describeError } from "./errors.js";
import { checkQuery, errorResponse, jsonResponse } from "./handlers.js";
import type { Env } from "./env.js";

/** The release a build or a function runs: version and short SHA. */
export interface Release {
  /** The release version, `YYYY.MM.N`. */
  version: string;
  /** The commit's short SHA (7 hex), or "local" outside a Vercel build. */
  sha: string;
}

/** The release version could not be read, or is not `YYYY.MM.N`. */
export class ReleaseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ReleaseError";
  }
}

/** The repo's `VERSION` file, from `web/server/`. */
export const VERSION_FILE = new URL("../../VERSION", import.meta.url);

/** `/api/v1/version`'s cache: a minute, since a deploy changes it. */
export const VERSION_CACHE = "s-maxage=60";

/** `YYYY.MM.N`: the month without a leading zero, N from 1 (ADR-0007). */
const VERSION_PATTERN = /^\d{4}\.(?:[1-9]|1[0-2])\.[1-9]\d*$/;

/** A git SHA, short or full. */
const SHA_PATTERN = /^[0-9a-f]{7,40}$/i;

/**
 * The release version a `VERSION` file holds.
 *
 * @param text - The file's text.
 * @returns The version, trimmed.
 * @throws ReleaseError when it is not `YYYY.MM.N`.
 */
export function parseVersion(text: string): string {
  const version = text.trim();
  if (!VERSION_PATTERN.test(version)) {
    throw new ReleaseError(`VERSION holds ${JSON.stringify(version)}, not YYYY.MM.N`);
  }
  return version;
}

/**
 * The short SHA of the commit Vercel builds or runs.
 *
 * @param env - The environment; reads `VERCEL_GIT_COMMIT_SHA`.
 * @returns Its first 7 characters, lower case; "local" when it is unset
 *   or not a SHA.
 */
export function shortSha(env: Env): string {
  const sha = env.VERCEL_GIT_COMMIT_SHA?.trim() ?? "";
  return SHA_PATTERN.test(sha) ? sha.slice(0, 7).toLowerCase() : "local";
}

/**
 * The release: `VERSION`'s value and the short SHA.
 *
 * @param env - The environment, for the SHA.
 * @param file - The `VERSION` file; the repo's by default.
 * @returns The release.
 * @throws ReleaseError when the file cannot be read or is malformed.
 */
export function readRelease(env: Env, file: URL | string = VERSION_FILE): Release {
  let text: string;
  try {
    // Kept as the literal `new URL(...)` by default, so the function's file
    // trace finds the file and ships it.
    text = readFileSync(file, "utf8");
  } catch (error) {
    throw new ReleaseError(`VERSION could not be read: ${describeError(error)}`);
  }
  return { version: parseVersion(text), sha: shortSha(env) };
}

/**
 * Vite's `define` for the page's release constants, read at build time.
 *
 * @param env - The build's environment, for the SHA.
 * @param file - The `VERSION` file; the repo's by default.
 * @returns `__LOOKOUT_VERSION__` and `__LOOKOUT_SHA__` as JSON literals.
 * @throws ReleaseError, failing the build, when `VERSION` cannot be read.
 */
export function releaseDefine(env: Env, file: URL | string = VERSION_FILE): Record<string, string> {
  const release = readRelease(env, file);
  return {
    __LOOKOUT_VERSION__: JSON.stringify(release.version),
    __LOOKOUT_SHA__: JSON.stringify(release.sha),
  };
}

/**
 * GET /api/v1/version: the deployed release, `{version, sha}`, the first
 * versioned client API (ADR-0007). It takes no parameters.
 *
 * @param request - The incoming request; see `checkQuery`.
 * @param read - Reads the release; `readRelease` over the environment.
 * @returns 200 with the release, cached for a minute; 400 for any query
 *   parameter; 500 when the release cannot be read.
 */
export async function handleVersion(request: Request, read: () => Release): Promise<Response> {
  const query = checkQuery(request);
  if (query instanceof Response) return query;
  let release: Release;
  try {
    release = read();
  } catch (error) {
    console.error("version: the release could not be read", describeError(error));
    return errorResponse(500, "the version could not be read");
  }
  return jsonResponse({ version: release.version, sha: release.sha }, 200, VERSION_CACHE);
}
