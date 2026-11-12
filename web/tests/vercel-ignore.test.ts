/**
 * Which branches Vercel deploys. `git.deploymentEnabled` in `vercel.json`
 * stops a deployment being created for branches nobody previews. The
 * `ignoreCommand` script then decides the rest: main and preview/* build,
 * goal/** builds only when web/ changed, anything else skips (exit 0 skips,
 * exit 1 builds).
 */
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

const script = new URL("../scripts/vercel-ignore.sh", import.meta.url).pathname;
const config = JSON.parse(readFileSync(new URL("../vercel.json", import.meta.url), "utf8")) as {
  ignoreCommand?: string;
  git?: { deploymentEnabled?: Record<string, boolean> };
};

/** The script's exit status for a branch and environment, run from `cwd`. */
function exitFor(branch: string, env: Record<string, string> = {}, cwd?: string): number | null {
  return spawnSync("sh", [script], {
    cwd,
    env: { PATH: process.env.PATH ?? "", VERCEL_GIT_COMMIT_REF: branch, ...env },
  }).status;
}

describe("vercel.json", () => {
  it("runs the script as the ignoreCommand", () => {
    expect(config.ignoreCommand).toBe("sh scripts/vercel-ignore.sh");
  });

  it("deploys only main, goal/** and preview/**", () => {
    expect(config.git?.deploymentEnabled).toEqual({
      main: true,
      "goal/**": true,
      "preview/**": true,
      dev: false,
      "feat/**": false,
      "fix/**": false,
      "docs/**": false,
      "chore/**": false,
      "lane/**": false,
      "shots/**": false,
      "prototype/**": false,
    });
  });
});

describe("vercel ignoreCommand", () => {
  it("builds main", () => {
    expect(exitFor("main")).toBe(1);
  });

  it.each(["preview/new-header", "preview/a/b"])("builds %s", (branch) => {
    expect(exitFor(branch)).toBe(1);
  });

  it.each([
    "dev",
    "docs/x",
    "lane/x",
    "shots/x",
    "fix/x",
    "feat/x",
    "chore/x",
    "prototype/x",
    "refactor/x",
    "mainline",
  ])("skips %s", (branch) => {
    expect(exitFor(branch)).toBe(0);
  });

  it("skips fix/x even with [preview] in the commit message", () => {
    expect(exitFor("fix/x", { VERCEL_GIT_COMMIT_MESSAGE: "fix: thing [preview]" })).toBe(0);
  });

  describe("goal branches", () => {
    let repo: string;
    let web: string;
    let base: string;
    let webChange: string;
    let pythonChange: string;

    /** Run git in the temp repo and return its trimmed output. */
    const git = (...args: string[]): string =>
      execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@t", ...args], { cwd: repo })
        .toString()
        .trim();

    /** Commit one file and return the new HEAD SHA. */
    const commit = (path: string, body: string): string => {
      writeFileSync(join(repo, path), body);
      git("add", "-A");
      git("commit", "-q", "-m", `change ${path}`);
      return git("rev-parse", "HEAD");
    };

    beforeAll(() => {
      repo = mkdtempSync(join(tmpdir(), "vercel-ignore-"));
      web = join(repo, "web");
      mkdirSync(web);
      git("init", "-q");
      writeFileSync(join(repo, "app.py"), "a = 1\n");
      base = commit("web/index.html", "<p>one</p>\n");
      webChange = commit("web/index.html", "<p>two</p>\n");
      pythonChange = commit("app.py", "a = 2\n");
    });

    afterAll(() => {
      rmSync(repo, { recursive: true, force: true });
    });

    it("builds when the push changed web/", () => {
      git("checkout", "-q", webChange);
      expect(exitFor("goal/x", { VERCEL_GIT_PREVIOUS_SHA: base }, web)).toBe(1);
    });

    it("skips when the push changed nothing under web/", () => {
      git("checkout", "-q", pythonChange);
      expect(exitFor("goal/x", { VERCEL_GIT_PREVIOUS_SHA: webChange }, web)).toBe(0);
    });

    it("builds when there is no previous SHA", () => {
      git("checkout", "-q", pythonChange);
      expect(exitFor("goal/x", {}, web)).toBe(1);
      expect(exitFor("goal/x", { VERCEL_GIT_PREVIOUS_SHA: "" }, web)).toBe(1);
    });

    it("builds when the previous SHA is not in the clone", () => {
      git("checkout", "-q", pythonChange);
      const missing = "0123456789abcdef0123456789abcdef01234567";
      expect(exitFor("goal/x", { VERCEL_GIT_PREVIOUS_SHA: missing }, web)).toBe(1);
    });
  });
});
