/**
 * The site's release (#536): the build-time constants the page reads, the
 * `VERSION` read behind them, and `/api/v1/version` called the way Vercel
 * calls it, `GET(request)` with `process.env` set.
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GET } from "../api/v1/version.js";
import { devResponse } from "../server/dev-fixtures.js";
import {
  ReleaseError,
  VERSION_CACHE,
  VERSION_FILE,
  handleVersion,
  parseVersion,
  readRelease,
  releaseDefine,
  shortSha,
} from "../server/release.js";
import { LOCAL_SHA, SITE_RELEASE, releaseLine } from "../src/release.js";

const REPO_VERSION = readFileSync(VERSION_FILE, "utf8").trim();
const FULL_SHA = "93e95b8a0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f";

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "lookout-release-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

/** A VERSION file holding `text`, in the test's own directory. */
function versionFile(text: string): string {
  const path = join(dir, "VERSION");
  writeFileSync(path, text);
  return path;
}

describe("the page's release", () => {
  it("is the repo's VERSION, baked in at build", () => {
    expect(SITE_RELEASE.version).toBe(REPO_VERSION);
  });

  it("is built locally here, so its SHA is local", () => {
    expect(SITE_RELEASE.sha).toBe("local");
  });

  it("reads as one footer line", () => {
    expect(releaseLine({ version: "2026.11.12", sha: "93e95b8" })).toBe(
      "Lookout 2026.11.12 · 93e95b8",
    );
    expect(releaseLine()).toBe(`Lookout ${REPO_VERSION} · local`);
  });
});

describe("releaseDefine", () => {
  it("follows a VERSION bump with no other edit", () => {
    const file = versionFile("2026.11.12\n");
    expect(releaseDefine({}, file).__LOOKOUT_VERSION__).toBe('"2026.11.12"');
    writeFileSync(file, "2026.12.1\n");
    expect(releaseDefine({}, file)).toEqual({
      __LOOKOUT_VERSION__: '"2026.12.1"',
      __LOOKOUT_SHA__: '"local"',
    });
  });

  it("bakes in the short SHA Vercel builds from", () => {
    const file = versionFile("2026.11.12");
    expect(releaseDefine({ VERCEL_GIT_COMMIT_SHA: FULL_SHA }, file).__LOOKOUT_SHA__).toBe(
      '"93e95b8"',
    );
  });

  it("fails the build when VERSION is missing", () => {
    expect(() => releaseDefine({}, join(dir, "VERSION"))).toThrow(ReleaseError);
    expect(() => releaseDefine({}, join(dir, "VERSION"))).toThrow(/VERSION could not be read/);
  });

  it("fails the build when VERSION is not a release version", () => {
    expect(() => releaseDefine({}, versionFile(""))).toThrow(/not YYYY.MM.N/);
    expect(() => releaseDefine({}, versionFile("unknown"))).toThrow(ReleaseError);
  });
});

describe("parseVersion", () => {
  it("takes YYYY.MM.N with the month unpadded", () => {
    expect(parseVersion(" 2026.11.12\n")).toBe("2026.11.12");
    expect(parseVersion("2027.1.1")).toBe("2027.1.1");
  });

  it.each(["2026.011.1", "2026.13.1", "2026.11.0", "v2026.11.1", "2026.11"])(
    "refuses %s",
    (text) => {
      expect(() => parseVersion(text)).toThrow(ReleaseError);
    },
  );
});

describe("shortSha", () => {
  it("shortens Vercel's SHA to 7", () => {
    expect(shortSha({ VERCEL_GIT_COMMIT_SHA: FULL_SHA })).toBe("93e95b8");
  });

  it("is local without one, or with something that is not a SHA", () => {
    expect(shortSha({})).toBe(LOCAL_SHA);
    expect(shortSha({ VERCEL_GIT_COMMIT_SHA: "" })).toBe("local");
    expect(shortSha({ VERCEL_GIT_COMMIT_SHA: "not-a-sha" })).toBe("local");
  });
});

describe("GET /api/v1/version", () => {
  it("answers the deployed version and short SHA as JSON, cached briefly", async () => {
    vi.stubEnv("VERCEL_GIT_COMMIT_SHA", FULL_SHA);
    const response = await GET(new Request("https://lookout.test/api/v1/version"));
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toBe("application/json; charset=utf-8");
    expect(response.headers.get("cache-control")).toBe(VERSION_CACHE);
    expect(await response.json()).toEqual({ version: REPO_VERSION, sha: "93e95b8" });
  });

  it("answers 400, not cached, for an unknown query parameter", async () => {
    const response = await GET(new Request("https://lookout.test/api/v1/version?station=x"));
    expect(response.status).toBe(400);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(await response.json()).toEqual({ error: "unknown query parameter" });
  });

  it("answers 500 with no detail when VERSION cannot be read, logging it safely", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const response = await handleVersion(new Request("https://lookout.test/api/v1/version"), () =>
      readRelease({}, join(dir, "VERSION")),
    );
    expect(response.status).toBe(500);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(await response.json()).toEqual({ error: "the version could not be read" });
    expect(log).toHaveBeenCalledWith(
      "version: the release could not be read",
      expect.stringMatching(/^ReleaseError: VERSION could not be read/),
    );
  });
});

describe("the lane server's /api/v1/version", () => {
  it("serves the checkout's VERSION with the SHA local", async () => {
    const response = await devResponse("/api/v1/version", {}, Date.now());
    expect(response!.status).toBe(200);
    expect(await response!.json()).toEqual({ version: REPO_VERSION, sha: "local" });
  });

  it("refuses the lane server's own parameters, as the function does", async () => {
    const response = await devResponse("/api/v1/version?fixture=dry", {}, Date.now());
    expect(response!.status).toBe(400);
  });
});
