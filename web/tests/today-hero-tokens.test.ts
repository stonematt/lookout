/**
 * The usual band hero's marks (#468): the band, the median, today's line, the
 * now dot and the rain bars each draw in a token DESIGN.md's Contrast table
 * covers, so the token contrast test speaks for every one of them; and the
 * median draws in `ink-on-now` in both themes (README decision 12: `median`
 * all but vanishes on the dark now surface).
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { contrastPairs } from "./contrast.js";

const CSS = readFileSync(
  resolve(import.meta.dirname, "../src/overview/first-screen.css"),
  "utf8",
).replace(/\/\*[\s\S]*?\*\//g, "");

/** The colour tokens a rule's property draws in. */
function tokensOf(selector: string, property: string): string[] {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const rule = new RegExp(`(?:^|\\n|\\})\\s*${escaped}\\s*\\{([^}]*)\\}`).exec(CSS);
  expect(rule, selector).not.toBeNull();
  const value = new RegExp(`(?:^|;|\\s)${property}:([^;]*);`).exec(rule![1]!);
  expect(value, `${selector} ${property}`).not.toBeNull();
  return [...value![1]!.matchAll(/var\(--([a-z0-9-]+)\)/g)].map((m) => m[1]!);
}

describe("the usual band hero's marks", () => {
  const pairs = contrastPairs();
  const covered = new Set(pairs.flatMap((p) => [...p.fg, p.bg]));

  it.each([
    [".today-band", "fill"],
    [".today-median", "stroke"],
    [".today-line,\n.today-join", "stroke"],
    [".today-dot", "fill"],
    [".today-rain rect[data-rain]", "fill"],
  ])("draw %s in a token the Contrast table covers", (selector, property) => {
    const names = tokensOf(selector, property);
    expect(names).toHaveLength(1);
    expect(covered).toContain(names[0]);
  });

  it("draw the median in ink-on-now, not median, in both themes", () => {
    expect(tokensOf(".today-median", "stroke")).toEqual(["ink-on-now"]);
    expect(CSS).not.toMatch(/var\(--median\)/);
  });

  it("hold the median against the band and the now surface in the Contrast table", () => {
    const has = (fg: string, bg: string) =>
      pairs.some((p) => p.fg.includes(fg) && p.bg === bg && p.themes.length === 2);
    expect(has("ink-on-now", "temp-band")).toBe(true);
    expect(has("ink-on-now", "now-surface")).toBe(true);
    expect(has("ink", "temp-band")).toBe(true);
  });
});
