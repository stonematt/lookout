#!/usr/bin/env node
/**
 * Check a running lane server for sideways scroll: load every tab at 320,
 * 390, 760 and 1180 px, in light and dark, and compare the page's
 * `scrollWidth` with its viewport (`clientWidth`), with the headless
 * Chromium from the `playwright` dev dependency.
 *
 *   npm run scroll-check -- <base-url> [query ...]
 *
 * e.g. npm run scroll-check -- http://localhost:5174 "fixture=dry&age=3" "fixture=raining&age=3"
 * Each query (default: none) is checked on #overview, #rain, #events and
 * #solar. Prints one line per case and exits 1 when any page scrolls
 * sideways. Install the browser once with `npx playwright install chromium`.
 */
import { chromium } from "playwright";

const [base, ...queries] = process.argv.slice(2);
if (!base) {
  console.error("usage: npm run scroll-check -- <base-url> [query ...]");
  process.exit(2);
}

const WIDTHS = [320, 390, 760, 1180];
const SCHEMES = ["light", "dark"];
const TABS = ["overview", "rain", "events", "solar"];

const failures = [];
const browser = await chromium.launch();
try {
  for (const query of queries.length ? queries : [""]) {
    for (const width of WIDTHS) {
      for (const colorScheme of SCHEMES) {
        const page = await browser.newPage({ viewport: { width, height: 844 }, colorScheme });
        for (const tab of TABS) {
          const url = `${base}/${query ? `?${query}` : ""}#${tab}`;
          await page.goto(url, { waitUntil: "load" });
          await page.waitForSelector(".masthead-age:not(:empty)", { timeout: 5000 });
          // Let charts that draw on a resize settle.
          await page.waitForTimeout(150);
          const { scroll, client } = await page.evaluate(() => ({
            scroll: document.documentElement.scrollWidth,
            client: document.documentElement.clientWidth,
          }));
          const ok = scroll <= client;
          console.log(
            `${ok ? "ok  " : "FAIL"} ${width}px ${colorScheme} ${url} ${scroll}/${client}`,
          );
          if (!ok) failures.push(url);
        }
        await page.close();
      }
    }
  }
} finally {
  await browser.close();
}
if (failures.length) {
  console.error(`${failures.length} page(s) scroll sideways`);
  process.exit(1);
}
