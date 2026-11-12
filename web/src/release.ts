/**
 * The release this build of the site is: the repo's release version
 * (`VERSION`, ADR-0007) and the short git SHA it was built from ("local"
 * outside a Vercel build). Both are baked in at build time by
 * `vite.config.ts` (`releaseDefine` in `server/release.ts`), which fails the
 * build when `VERSION` cannot be read; nothing is fetched at run time.
 *
 * Offers `LOCAL_SHA`, `SiteRelease`, `SITE_RELEASE` and `releaseLine`.
 */

declare const __LOOKOUT_VERSION__: string;
declare const __LOOKOUT_SHA__: string;

/**
 * The SHA of a build made outside Vercel, which names no commit. `shortSha`
 * in `server/release.ts` sets it and spells it too: the page cannot import
 * that module (it reads `node:fs`), nor it this one (its constants exist
 * only in a Vite build).
 */
export const LOCAL_SHA = "local";

/** A release: version and short SHA. */
export interface SiteRelease {
  readonly version: string;
  readonly sha: string;
}

/** This build's release. */
export const SITE_RELEASE: SiteRelease = Object.freeze({
  version: __LOOKOUT_VERSION__,
  sha: __LOOKOUT_SHA__,
});

/**
 * The release as one line, for a footer.
 *
 * @param release - The release; this build's by default.
 * @returns e.g. "Lookout 2026.11.12 · 93e95b8".
 */
export function releaseLine(release: SiteRelease = SITE_RELEASE): string {
  return `Lookout ${release.version} · ${release.sha}`;
}
