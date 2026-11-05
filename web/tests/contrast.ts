/**
 * The token contrast checker (#333), ported from the prototype's
 * `contrast-check.py`: WCAG contrast between colour tokens, the Contrast
 * table read from DESIGN.md's frontmatter, and the tokens read from
 * `web/src/styles/tokens.css` in each theme.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const REPO = resolve(import.meta.dirname, "../..");

export type Theme = "light" | "dark";

/** One row of DESIGN.md's Contrast table. */
export interface ContrastPair {
  /** The foreground token, or tokens meaning "the better of these". */
  fg: string[];
  bg: string;
  /** 4.5 for text, 3 for graphics. */
  min: number;
  themes: Theme[];
}

/** A theme's tokens: name (no `--`) to value, `var()` references resolved. */
export type Tokens = Record<string, string>;

/**
 * WCAG relative luminance of a `#rrggbb` colour.
 *
 * @param hex - The colour.
 * @returns Its luminance, 0 to 1.
 */
export function luminance(hex: string): number {
  if (!/^#[0-9a-f]{6}$/i.test(hex)) throw new Error(`not a #rrggbb colour: ${hex}`);
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
}

/**
 * The WCAG contrast ratio between two colours.
 *
 * @param a - One colour.
 * @param b - The other.
 * @returns The ratio, 1 to 21.
 */
export function ratio(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

/** DESIGN.md's YAML frontmatter, as text. */
function frontmatter(): string {
  const text = readFileSync(`${REPO}/DESIGN.md`, "utf8");
  const match = /^---\n([\s\S]*?)\n---\n/.exec(text);
  if (!match) throw new Error("DESIGN.md has no frontmatter");
  return match[1]!;
}

/** The lines of one top-level frontmatter key's block. */
function block(key: string): string[] {
  const lines = frontmatter().split("\n");
  const start = lines.indexOf(`${key}:`);
  if (start < 0) throw new Error(`DESIGN.md frontmatter has no ${key}:`);
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((l) => /^\S/.test(l) && !l.startsWith("#"));
  return (end < 0 ? rest : rest.slice(0, end)).filter((l) => !/^\s*(#.*)?$/.test(l));
}

/**
 * DESIGN.md's colour tokens in one theme: the base keys in light, the
 * `-dark` keys (named without the suffix) in dark.
 *
 * @param theme - Which theme.
 * @returns The tokens.
 */
export function designColours(theme: Theme): Tokens {
  const out: Tokens = {};
  for (const line of block("colors")) {
    const m = /^\s+([a-z0-9-]+):\s*"(#[0-9a-f]{6})"\s*$/.exec(line);
    if (!m) throw new Error(`unread colors line: ${line}`);
    const [, key, value] = m;
    const dark = key!.endsWith("-dark");
    if (dark === (theme === "dark")) out[key!.replace(/-dark$/, "")] = value!;
  }
  return out;
}

/**
 * The Contrast table, from the frontmatter's `contrast:` list. A line the
 * reader does not understand throws, so the list cannot quietly shrink.
 *
 * @returns Every pair, in order.
 */
export function contrastPairs(): ContrastPair[] {
  const row =
    /^\s+- \{fg: (\[[^\]]+\]|[a-z0-9-]+), bg: ([a-z0-9-]+), min: ([0-9.]+), themes: \[([^\]]+)\]\}\s*$/;
  const list = (s: string) => s.split(",").map((x) => x.trim());
  return block("contrast").map((line) => {
    const m = row.exec(line);
    if (!m) throw new Error(`unread contrast line: ${line}`);
    const [, fg, bg, min, themes] = m;
    return {
      fg: fg!.startsWith("[") ? list(fg!.slice(1, -1)) : [fg!],
      bg: bg!,
      min: Number(min),
      themes: list(themes!) as Theme[],
    };
  });
}

/**
 * A stylesheet's rules, top level and inside at-rules, as [selector, body].
 * A plain brace count: enough for tokens.css, which has no braces in strings.
 */
function rules(css: string, context = ""): [string, string][] {
  const out: [string, string][] = [];
  const text = css.replace(/\/\*[\s\S]*?\*\//g, "");
  let i = 0;
  while (i < text.length) {
    const open = text.indexOf("{", i);
    if (open < 0) break;
    const selector = text.slice(i, open).trim();
    let depth = 1;
    let j = open + 1;
    for (; j < text.length && depth > 0; j += 1) {
      if (text[j] === "{") depth += 1;
      if (text[j] === "}") depth -= 1;
    }
    const body = text.slice(open + 1, j - 1);
    if (selector.startsWith("@")) out.push(...rules(body, `${context}${selector} `));
    else out.push([`${context}${selector}`, body]);
    i = j;
  }
  return out;
}

/** Every `--name: value` in a rule body. */
function declarations(body: string): Tokens {
  const out: Tokens = {};
  for (const m of body.matchAll(/--([a-z0-9-]+)\s*:\s*([^;]+);/g)) out[m[1]!] = m[2]!.trim();
  return out;
}

/** The tokens stylesheet's text. */
export function tokensCss(): string {
  return readFileSync(`${REPO}/web/src/styles/tokens.css`, "utf8");
}

/** The selectors that set the dark theme: by the OS, and by `?theme=dark`. */
export const DARK_BY_OS = '@media (prefers-color-scheme: dark) :root:not([data-theme="light"])';
export const DARK_BY_CHOICE = ':root[data-theme="dark"]';

/**
 * The custom properties one selector of the tokens stylesheet sets.
 *
 * @param selector - e.g. `:root` or {@link DARK_BY_OS}.
 * @param css - The stylesheet (default: tokens.css).
 * @returns The declarations, unresolved, merged across that selector's rules.
 */
export function declared(selector: string, css = tokensCss()): Tokens {
  return Object.assign(
    {},
    ...rules(css)
      .filter(([s]) => s === selector)
      .map(([, body]) => declarations(body)),
  ) as Tokens;
}

/**
 * The tokens as a browser would compute them in one theme: the light rules,
 * with the dark rule over them in dark, every `var()` resolved. Dark reads
 * the `?theme=dark` rule; tokens.test.ts holds the OS dark rule equal to it.
 *
 * @param theme - Which theme.
 * @param css - The stylesheet (default: tokens.css).
 * @returns The resolved tokens.
 */
export function themeTokens(theme: Theme, css = tokensCss()): Tokens {
  const raw = {
    ...declared(":root", css),
    ...(theme === "dark" ? declared(DARK_BY_CHOICE, css) : {}),
  };
  const resolveVar = (value: string, seen: string[]): string =>
    value.replace(/var\(--([a-z0-9-]+)\)/g, (_, name: string) => {
      if (seen.includes(name)) throw new Error(`token cycle: ${[...seen, name].join(" -> ")}`);
      const next = raw[name];
      if (next === undefined) throw new Error(`undefined token --${name}`);
      return resolveVar(next, [...seen, name]);
    });
  return Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, resolveVar(v, [k])]));
}

/** One pair that falls under its rule. */
export interface Failure {
  theme: Theme;
  pair: ContrastPair;
  ratio: number;
}

/**
 * Check every pair against a theme's tokens.
 *
 * @param pairs - The Contrast table.
 * @param tokens - Each theme's tokens.
 * @returns Every pair, per theme, whose best foreground falls under its minimum.
 */
export function failures(pairs: ContrastPair[], tokens: Record<Theme, Tokens>): Failure[] {
  const out: Failure[] = [];
  for (const pair of pairs) {
    for (const theme of pair.themes) {
      const t = tokens[theme];
      const colour = (name: string) => {
        const v = t[name];
        if (v === undefined) throw new Error(`${theme} has no token --${name}`);
        return v;
      };
      const best = Math.max(...pair.fg.map((fg) => ratio(colour(fg), colour(pair.bg))));
      if (best < pair.min) out.push({ theme, pair, ratio: best });
    }
  }
  return out;
}
