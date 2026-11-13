/**
 * The Tiles That Talk tokens (#333): tokens.css carries DESIGN.md's colours
 * in both themes, every pair in DESIGN.md's Contrast table passes, and the
 * site's stylesheets draw only from the tokens, in Lexend, at 12px or more.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
  contrastPairs,
  DARK_BY_CHOICE,
  DARK_BY_OS,
  declared,
  designColours,
  failures,
  ratio,
  themeTokens,
  type Theme,
} from "./contrast.js";

const WEB = resolve(import.meta.dirname, "..");
const THEMES: Theme[] = ["light", "dark"];
const tokens = { light: themeTokens("light"), dark: themeTokens("dark") };

/** Every file under `web/src` with one of the extensions, as [path, text]. */
function sources(...exts: string[]): [string, string][] {
  const out: [string, string][] = [];
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (exts.some((e) => entry.name.endsWith(e)))
        out.push([path, readFileSync(path, "utf8")]);
    }
  };
  walk(join(WEB, "src"));
  return out;
}

const uncommented = (text: string) => text.replace(/\/\*[\s\S]*?\*\//g, "");

describe("the tokens stylesheet", () => {
  it.each(THEMES)("carries every DESIGN.md colour in %s, at its value", (theme) => {
    const design = designColours(theme);
    expect(Object.keys(design).length).toBeGreaterThan(40);
    for (const [name, value] of Object.entries(design)) {
      expect(tokens[theme][name], `--${name}`).toBe(value);
    }
  });

  it("sets dark the same way by the OS and by ?theme=dark", () => {
    expect(declared(DARK_BY_OS)).toEqual(declared(DARK_BY_CHOICE));
  });

  it("keeps no Paper-world token", () => {
    for (const old of [
      "note",
      "sun",
      "sun-soft",
      "night",
      "gap-a",
      "gap-b",
      "temp-hot",
      "temp-cold",
    ]) {
      expect(tokens.light[old], `--${old}`).toBeUndefined();
    }
    for (const [path, text] of sources(".css", ".ts")) {
      expect(text, path).not.toMatch(/claret/i);
    }
  });
});

describe("the Contrast table", () => {
  const pairs = contrastPairs();

  it("is read from the DESIGN.md frontmatter, both themes on every row", () => {
    expect(pairs.length).toBeGreaterThan(50);
    expect(pairs.every((p) => p.themes.includes("light") && p.themes.includes("dark"))).toBe(true);
    // text and graphics, the band edges in light, and text in every ramp cell
    expect(pairs.some((p) => p.min === 4.5)).toBe(true);
    expect(pairs.some((p) => p.min === 3)).toBe(true);
    for (const edge of ["rain-band-edge", "temp-band-edge", "sun-band-edge"]) {
      expect(pairs.some((p) => p.fg.includes(edge) && p.bg === "tile")).toBe(true);
    }
    for (const ramp of ["rain", "sun"]) {
      for (let step = 0; step < 5; step += 1) {
        expect(pairs.some((p) => p.bg === `${ramp}-${step}` && p.min === 4.5)).toBe(true);
      }
    }
  });

  it("passes every pair in both themes", () => {
    const found = failures(pairs, tokens).map(
      (f) =>
        `${f.theme}: ${f.pair.fg.join("|")} on ${f.pair.bg} ${f.ratio.toFixed(2)} < ${f.pair.min}`,
    );
    expect(found).toEqual([]);
  });

  it("fails a pair that drops below its rule", () => {
    const light = { ...tokens.light, "ink-3": "#9aa4ae" };
    const found = failures(pairs, { light, dark: tokens.dark });
    expect(found.length).toBeGreaterThan(0);
    expect(found.every((f) => f.theme === "light" && f.pair.fg.includes("ink-3"))).toBe(true);
    // the light band edge carries the band: a pale edge fails it
    const pale = { ...tokens.light, "rain-band-edge": tokens.light["rain-band"]! };
    expect(failures(pairs, { light: pale, dark: tokens.dark }).map((f) => f.pair.fg[0])).toContain(
      "rain-band-edge",
    );
  });

  it("measures as WCAG does", () => {
    expect(ratio("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(ratio("#777777", "#777777")).toBe(1);
  });
});

describe("text on a ramp cell", () => {
  it.each(THEMES)("takes a cell ink that passes 4.5:1 on every %s step", (theme) => {
    const t = tokens[theme];
    for (const ramp of ["rain", "sun"]) {
      for (let step = 0; step < 5; step += 1) {
        const ink = t[`on-${ramp}-${step}`]!;
        expect([t["on-ramp-pale"], t["on-ramp-deep"]]).toContain(ink);
        expect(ratio(ink, t[`${ramp}-${step}`]!), `${ramp}-${step}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("is the ink the solar grid's cells take, step by step", () => {
    const css = readFileSync(join(WEB, "src/solar/solar.css"), "utf8");
    // the grid's steps 1 to 5 are the sun ramp's 0 to 4
    for (let step = 1; step <= 5; step += 1) {
      const fill = new RegExp(`\\.solar-step-${step} \\{\\s*fill: var\\(--sun-${step - 1}\\);`);
      const text = new RegExp(
        `\\.solar-cell-text\\[data-step="${step}"\\] \\{\\s*fill: var\\(--on-sun-${step - 1}\\);`,
      );
      expect(css).toMatch(fill);
      expect(css).toMatch(text);
    }
  });
});

describe("the dry spell's dry squares (#606)", () => {
  it.each(THEMES)("meet the graphics minimum on the tile in %s", (theme) => {
    const css = readFileSync(join(WEB, "src/overview/mosaic.css"), "utf8");
    const token = /\.mosaic-dry \{\s*fill: var\(--([a-z0-9-]+)\);/.exec(css)?.[1];
    expect(token).toBeDefined();
    const t = tokens[theme];
    expect(ratio(t[token!]!, t.tile!)).toBeGreaterThanOrEqual(3);
    // and the token is the one the Contrast table holds to it
    const pairs = contrastPairs();
    expect(pairs.some((p) => p.fg.includes(token!) && p.bg === "tile" && p.min === 3)).toBe(true);
  });
});

describe("the site's stylesheets", () => {
  const css = sources(".css");

  it("set no colour outside the tokens stylesheet", () => {
    const literal = /#[0-9a-f]{3,8}\b|\brgba?\(|\bhsla?\(/i;
    for (const [path, text] of css) {
      if (path.endsWith("styles/tokens.css")) continue;
      const lines = uncommented(text)
        .split("\n")
        .filter((l) => literal.test(l));
      expect(lines, path).toEqual([]);
    }
  });

  it("reference only custom properties something declares", () => {
    const all = [...css, ...sources(".ts")];
    const declaredNames = new Set(
      all.flatMap(([, text]) => [...text.matchAll(/--([a-z0-9-]+)\s*:/g)].map((m) => m[1]!)),
    );
    for (const [path, text] of all) {
      for (const m of uncommented(text).matchAll(/var\(--([a-z0-9-]+)/g)) {
        expect(declaredNames.has(m[1]!), `${path}: --${m[1]}`).toBe(true);
      }
    }
  });

  it("set text left-aligned, numbers right, and centre only a column head", () => {
    const centred: string[] = [];
    for (const [, text] of css) {
      for (const m of uncommented(text).matchAll(
        /([^{}]+)\{[^}]*text-align:\s*(center|justify)/g,
      )) {
        centred.push(m[1]!.trim());
      }
    }
    // the almanac's month names sit over their column of cells
    expect(centred).toEqual([".alm-month"]);
  });

  it("set no all-caps, no italic, and no text under 12px", () => {
    for (const [path, text] of css) {
      const body = uncommented(text);
      expect(body, path).not.toMatch(
        /text-transform:\s*uppercase|font-variant(-caps)?:\s*[^;]*caps/,
      );
      expect(body, path).not.toMatch(/font-style:\s*(italic|oblique)/);
      for (const m of body.matchAll(/font-size:\s*([0-9.]+)(px|rem|em)\b/g)) {
        const [, n, unit] = m;
        // em sizes scale their parent; only the superscript degree uses one
        if (unit === "em") continue;
        const px = unit === "px" ? Number(n) : Number(n) * 16;
        expect(px, `${path}: ${m[0]}`).toBeGreaterThanOrEqual(12);
      }
    }
    for (const [name, value] of Object.entries(declared(":root"))) {
      const m = /^([0-9.]+)(px|rem)$/.exec(value);
      if (name.startsWith("type-") && m) {
        const px = m[2] === "px" ? Number(m[1]) : Number(m[1]) * 16;
        expect(px, `--${name}`).toBeGreaterThanOrEqual(12);
      }
    }
  });
});

describe("the type", () => {
  it("is Lexend, self-hosted, and Atkinson is gone", () => {
    const pkg = JSON.parse(readFileSync(join(WEB, "package.json"), "utf8")) as {
      dependencies: Record<string, string>;
    };
    expect(pkg.dependencies["@fontsource-variable/lexend"]).toBeDefined();
    expect(Object.keys(pkg.dependencies).some((d) => /atkinson/i.test(d))).toBe(false);
    const main = readFileSync(join(WEB, "src/main.ts"), "utf8");
    expect(main).toContain('import "@fontsource-variable/lexend/wght.css";');
    expect(declared(":root")["sans"]).toMatch(/^"Lexend Variable", Lexend,/);
    for (const [path, text] of sources(".css", ".ts", ".html")) {
      expect(text, path).not.toMatch(/atkinson/i);
    }
  });
});

describe("the sun ramp", () => {
  it.each(THEMES)("brightens with the sun, step by step, in %s", (theme) => {
    const t = tokens[theme];
    const lum = (step: number) => ratio(t[`sun-${step}`]!, "#000000");
    for (let step = 1; step < 5; step += 1) {
      expect(lum(step), `sun-${step} brighter than sun-${step - 1}`).toBeGreaterThan(lum(step - 1));
    }
  });
});
