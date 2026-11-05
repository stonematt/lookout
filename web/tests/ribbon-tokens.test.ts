/**
 * The 48-hour ribbon's marks (#336): night, day, the rain bars, the
 * temperature line and its halo, the outline, the rain track's floor and the
 * now dot each draw in a token DESIGN.md's Contrast table covers, so the
 * token contrast test speaks for every one of them.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { contrastPairs } from "./contrast.js";

const CSS = readFileSync(
  resolve(import.meta.dirname, "../src/overview/first-screen.css"),
  "utf8",
).replace(/\/\*[\s\S]*?\*\//g, "");

/** The one colour token a rule's property draws in. */
function tokenOf(selector: string, property: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const rule = new RegExp(`(?:^|\\n|\\})\\s*${escaped}\\s*\\{([^}]*)\\}`).exec(CSS);
  expect(rule, selector).not.toBeNull();
  const value = new RegExp(`(?:^|;|\\s)${property}:([^;]*);`).exec(rule![1]!);
  expect(value, `${selector} ${property}`).not.toBeNull();
  const names = [...value![1]!.matchAll(/var\(--([a-z0-9-]+)\)/g)].map((m) => m[1]!);
  expect(names, `${selector} ${property}`).toHaveLength(1);
  // the now tile's surface is `now-surface`
  return names[0] === "surface" ? "now-surface" : names[0]!;
}

describe("the ribbon's marks", () => {
  const covered = new Set(contrastPairs().flatMap((p) => [...p.fg, p.bg]));

  it.each([
    ['.ribbon-band rect[data-light="night"]', "fill"],
    ['.ribbon-band rect[data-light="day"]', "fill"],
    [".ribbon-rain rect", "fill"],
    [".ribbon-line", "stroke"],
    [".ribbon-halo", "stroke"],
    [".ribbon-band::after", "border"],
    [".ribbon-rain", "border-bottom"],
    [".ribbon-dot", "background"],
  ])("draw %s in a token the Contrast table covers", (selector, property) => {
    expect(covered).toContain(tokenOf(selector, property));
  });

  it("draw the rain bars in the ribbon's rain colour, not the rain ramp", () => {
    expect(tokenOf(".ribbon-rain rect", "fill")).toBe("ribbon-rain");
    expect(CSS).not.toMatch(/\.ribbon[^{]*\{[^}]*var\(--rain-\d\)/);
  });
});
