/**
 * The `ignoreCommand` in `vercel.json`: shots/* and lane/* branches build no
 * preview (exit 0 skips), every other branch builds (exit 1).
 */
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const script = new URL("../scripts/vercel-ignore.sh", import.meta.url).pathname;
const config = JSON.parse(readFileSync(new URL("../vercel.json", import.meta.url), "utf8")) as {
  ignoreCommand?: string;
};

/** The script's exit status for a branch name. */
function exitFor(branch: string): number | null {
  return spawnSync("sh", [script], { env: { VERCEL_GIT_COMMIT_REF: branch } }).status;
}

describe("vercel ignoreCommand", () => {
  it("runs the script from vercel.json", () => {
    expect(config.ignoreCommand).toBe("sh scripts/vercel-ignore.sh");
  });

  it.each(["shots/goal-tiles-that-talk", "lane/fix-thing", "shots/a/b"])("skips %s", (branch) => {
    expect(exitFor(branch)).toBe(0);
  });

  it.each(["main", "dev", "goal/tiles-that-talk", "feat/x", "fix/shots/y", "docs/lane", "shotsy"])(
    "builds %s",
    (branch) => {
      expect(exitFor(branch)).toBe(1);
    },
  );
});
