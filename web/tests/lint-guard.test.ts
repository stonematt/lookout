/**
 * The lint guard on HTML built from snapshot values (#148, #260): an
 * unescaped value assigned to innerHTML fails `npm run lint`, and one passed
 * through `esc` or `escapeHtml` does not.
 */

import { ESLint } from "eslint";
import { describe, expect, it } from "vitest";

/** Lint one snippet as if it were a file under `src/`, with the repo config. */
async function lint(code: string): Promise<string[]> {
  const eslint = new ESLint({ cwd: new URL("..", import.meta.url).pathname });
  const [result] = await eslint.lintText(code, { filePath: "src/lint-guard-probe.ts" });
  return result!.messages.map((m) => m.ruleId ?? m.message);
}

describe("the innerHTML lint guard", () => {
  it("fails an unescaped value assigned to innerHTML", async () => {
    const rules = await lint(
      "export function f(el: HTMLElement, name: string): void {\n  el.innerHTML = `<b>${name}</b>`;\n}\n",
    );
    expect(rules).toContain("no-unsanitized/property");
  });

  it("fails an unescaped value passed to insertAdjacentHTML", async () => {
    const rules = await lint(
      'export function f(el: HTMLElement, name: string): void {\n  el.insertAdjacentHTML("beforeend", name);\n}\n',
    );
    expect(rules).toContain("no-unsanitized/method");
  });

  it("passes a value escaped with esc or escapeHtml", async () => {
    const rules = await lint(
      [
        "declare function esc(v: unknown): string;",
        "declare function escapeHtml(v: unknown): string;",
        "export function f(el: HTMLElement, name: string): void {",
        "  el.innerHTML = `<b>${esc(name)}</b><i>${escapeHtml(name)}</i>`;",
        "}",
        "",
      ].join("\n"),
    );
    expect(rules).toEqual([]);
  });
});
