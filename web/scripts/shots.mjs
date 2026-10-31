#!/usr/bin/env node
/**
 * Screenshot a running lane server at phone and desktop width, light and
 * dark, with the headless Chromium from the `playwright` dev dependency.
 *
 *   npm run shots -- <base-url> <out-dir> [query ...]
 *
 * e.g. npm run shots -- http://localhost:5177 /tmp/shots "fixture=raining&age=3"
 * Each query (default: none) is shot at 390px and 1280px in both themes.
 * Install the browser once with `npx playwright install chromium`.
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { chromium } from "playwright";

const [base, outDir, ...queries] = process.argv.slice(2);
if (!base || !outDir) {
  console.error("usage: npm run shots -- <base-url> <out-dir> [query ...]");
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

const WIDTHS = [390, 1280];
const SCHEMES = ["light", "dark"];
const browser = await chromium.launch();
try {
  for (const query of queries.length ? queries : [""]) {
    for (const width of WIDTHS) {
      for (const colorScheme of SCHEMES) {
        const page = await browser.newPage({
          viewport: { width, height: width < 600 ? 844 : 800 },
          colorScheme,
          deviceScaleFactor: 2,
        });
        const url = `${base}/${query ? `?${query}` : ""}`;
        await page.goto(url, { waitUntil: "load" });
        await page.waitForSelector(".masthead-age:not(:empty)", { timeout: 5000 });
        const slug = (query || "default").replace(/[^a-z0-9]+/gi, "-");
        const file = join(outDir, `${slug}-${width}-${colorScheme}.png`);
        await page.screenshot({ path: file, fullPage: true });
        console.log(file);
        await page.close();
      }
    }
  }
} finally {
  await browser.close();
}
