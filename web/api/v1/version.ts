import { handleVersion, readRelease } from "../../server/release.js";

/**
 * GET /api/v1/version: the deployed release, `{version, sha}` (#536).
 *
 * @param request - The incoming request.
 * @returns The response; see `handleVersion`.
 */
export function GET(request: Request): Promise<Response> {
  return handleVersion(request, () => readRelease(process.env));
}
