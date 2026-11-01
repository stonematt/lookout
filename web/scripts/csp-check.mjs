#!/usr/bin/env node
/**
 * Load a served build with the security headers `vercel.json` sets, and
 * report every Content-Security-Policy violation the page raises.
 *
 *   npm run build && npx vite preview --port <port> --strictPort
 *   node scripts/csp-check.mjs http://localhost:<port> [out-dir]
 *
 * Neither `vite` nor `vite preview` applies `vercel.json`, so each response
 * is intercepted and given the catch-all rule's headers. Every tab is opened
 * at 390px and 1280px, light and dark, over both goldens; on Rain Events the
 * live event's charts and a completed event's charts (fetched from /event)
 * are drawn with Plotly. With an out-dir, each view is screenshotted.
 * Exits 1 on any violation or page error. Uses the `playwright` dev
 * dependency's Chromium (`npx playwright install chromium` once).
 */
import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { chromium } from "playwright";

const [base, outDir] = process.argv.slice(2);
if (!base) {
  console.error("usage: node scripts/csp-check.mjs <base-url> [out-dir]");
  process.exit(2);
}
if (outDir) mkdirSync(outDir, { recursive: true });

const config = JSON.parse(readFileSync(new URL("../vercel.json", import.meta.url), "utf8"));
const rule = config.headers.find((r) => r.source === "/(.*)");
// LOOKOUT_NO_HEADERS=1 runs the same views without the headers, for a
// baseline to compare the screenshots against.
const headers = process.env.LOOKOUT_NO_HEADERS
  ? {}
  : Object.fromEntries(rule.headers.map((h) => [h.key.toLowerCase(), h.value]));
console.log(`CSP: ${headers["content-security-policy"] ?? "(none)"}`);

const QUERIES = ["fixture=raining&age=3", "fixture=dry&age=3"];
const TABS = ["overview", "rain", "events", "solar"];
const problems = [];

const browser = await chromium.launch();
try {
  for (const query of QUERIES) {
    for (const width of [390, 1280]) {
      for (const colorScheme of ["light", "dark"]) {
        const page = await browser.newPage({
          viewport: { width, height: width < 600 ? 844 : 800 },
          colorScheme,
        });
        await page.route("**/*", async (route) => {
          const response = await route.fetch();
          await route.fulfill({ response, headers: { ...response.headers(), ...headers } });
        });
        // Init scripts run outside the page's CSP, so the listener is not
        // itself a violation.
        await page.addInitScript(() => {
          window.__violations = [];
          document.addEventListener("securitypolicyviolation", (e) => {
            window.__violations.push(
              `${e.effectiveDirective} blocked ${e.blockedURI || "(inline)"} ${e.sourceFile}:${e.lineNumber} ${e.sample}`,
            );
          });
        });
        const where = `${query} ${width}px ${colorScheme}`;
        page.on("pageerror", (error) => problems.push(`${where}: page error ${error.message}`));
        page.on("console", (message) => {
          if (/content security policy/i.test(message.text())) {
            problems.push(`${where}: console ${message.text()}`);
          }
        });

        const shot = async (name) => {
          if (!outDir) return;
          const slug = `${query.replace(/[^a-z0-9]+/gi, "-")}-${name}-${width}-${colorScheme}`;
          await page.screenshot({ path: join(outDir, `${slug}.png`), fullPage: true });
        };

        await page.goto(`${base}/?${query}#overview`, { waitUntil: "load" });
        await page.waitForSelector(".masthead-age:not(:empty)", { timeout: 5000 });
        for (const tab of TABS) {
          await page.evaluate((id) => (location.hash = id), tab);
          await page.waitForTimeout(300);
          if (tab === "events" && (await page.$(".ev-chart"))) {
            await page.waitForSelector(".ev-chart.js-plotly-plot", { timeout: 10000 });
          }
          await shot(tab);
        }
        // A completed event: its curve comes from /event.
        await page.evaluate(() => (location.hash = "events"));
        await page.waitForTimeout(300);
        const links = await page.$$(".ev-open");
        const closed = await page.$$eval(".ev-table tbody tr:not(.pin) .ev-open", (as) =>
          as.map((a) => a.dataset.start),
        );
        if (links.length && closed.length) {
          // A DOM click: at 390px Plotly's svg can overlap the list.
          await page.$eval(`.ev-open[data-start="${closed[0]}"]`, (a) => a.click());
          await page.waitForFunction(
            () => document.querySelectorAll(".ev-chart.js-plotly-plot").length === 2,
            undefined,
            { timeout: 10000 },
          );
          await page.waitForTimeout(300);
          await shot("event-opened");
        }

        const violations = await page.evaluate(() => window.__violations);
        for (const v of violations) problems.push(`${where}: ${v}`);
        console.log(`${where}: ${violations.length} violation(s)`);
        await page.close();
      }
    }
  }
} finally {
  await browser.close();
}

if (problems.length) {
  console.error([...new Set(problems)].join("\n"));
  process.exit(1);
}
console.log("no CSP violations");
