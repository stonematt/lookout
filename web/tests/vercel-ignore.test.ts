/**
 * Which branches Vercel deploys. `git.deploymentEnabled` in `vercel.json` is
 * an allowlist: a catch-all false, and true for main, dev and uat/**. Vercel
 * deploys a branch when at least one matching rule is true. The
 * `ignoreCommand` script repeats the allowlist as a backstop (exit 0 skips,
 * exit 1 builds).
 */
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";

import { minimatch } from "minimatch";
import { describe, expect, it } from "vitest";

const script = new URL("../scripts/vercel-ignore.sh", import.meta.url).pathname;
const config = JSON.parse(readFileSync(new URL("../vercel.json", import.meta.url), "utf8")) as {
  ignoreCommand?: string;
  git?: { deploymentEnabled?: Record<string, boolean> };
};

const deployed = ["main", "dev", "uat/mosaic-r1", "uat/a/b"];
const skipped = [
  "goal/mosaic-r1",
  "goal/a/b",
  "preview/x",
  "feat/x",
  "fix/x",
  "docs/x",
  "chore/x",
  "lane/x",
  "shots/x",
  "prototype/mosaic-overview",
  "refactor/x",
  "mainline",
  "uat",
];

/**
 * Vercel's documented rule: a branch deploys when at least one matching
 * `deploymentEnabled` pattern is true, and a branch no pattern matches deploys.
 */
function vercelDeploys(branch: string): boolean {
  const rules = Object.entries(config.git?.deploymentEnabled ?? {}).filter(([pattern]) =>
    minimatch(branch, pattern),
  );
  return rules.length === 0 || rules.some(([, enabled]) => enabled);
}

/** The script's exit status for a branch name. */
function exitFor(branch: string): number | null {
  return spawnSync("sh", [script], {
    env: { PATH: process.env.PATH ?? "", VERCEL_GIT_COMMIT_REF: branch },
  }).status;
}

describe("vercel.json", () => {
  it("runs the script as the ignoreCommand", () => {
    expect(config.ignoreCommand).toBe("sh scripts/vercel-ignore.sh");
  });

  it("allowlists main, dev and uat/** behind a catch-all false", () => {
    expect(config.git?.deploymentEnabled).toEqual({
      "**": false,
      main: true,
      dev: true,
      "uat/**": true,
    });
  });

  it.each(deployed)("deploys %s", (branch) => {
    expect(vercelDeploys(branch)).toBe(true);
  });

  it.each(skipped)("creates no deployment for %s", (branch) => {
    expect(vercelDeploys(branch)).toBe(false);
  });
});

describe("vercel ignoreCommand", () => {
  it.each(deployed)("builds %s", (branch) => {
    expect(exitFor(branch)).toBe(1);
  });

  it.each(skipped)("skips %s", (branch) => {
    expect(exitFor(branch)).toBe(0);
  });

  it("skips a branch with no name", () => {
    expect(exitFor("")).toBe(0);
  });
});
