#!/usr/bin/env node
/**
 * Layout check (#402): open every tab of the built site in its own headless
 * Chromium and fail, naming the view, on four layout facts.
 *
 *   npm run layout-check            # builds with VITE_LANE_SERVER=1, then checks
 *   node scripts/layout-check.mjs <base-url>  # check a running build instead
 *   npm run layout-check -- --prove # also proves the #395 shape would fail
 *
 *  1. Sideways scroll: the page's `scrollWidth` is wider than its
 *     `clientWidth`.
 *  2. A chart narrower than its tile's content box: an `svg[role=img]` or a
 *     Plotly chart whose width falls short of the element that holds it.
 *  3. Stat values that touch: in a `dl`, the text of two different items
 *     (a value, its label) whose boxes overlap.
 *
 * Every tab is visited at 320, 390, 760 and 1440 px, in light and dark, on
 * three cases: the `dry` and `raining` goldens, and `live`, the raining
 * golden with every Rain and Solar window valid and holding the wide values
 * live data produces (5 digits, as in #395), which the goldens do not. On a
 * phone each folded finding row is opened in turn, so a chart inside a
 * closed row is measured too. Then every registered zoom (#474) is opened
 * at its route, `#overview/<slug>`, at each width, scheme and case, and
 * measured the same way: the zooms are found from the Overview's tile
 * links (`a[data-zoom]`, set by `src/overview/zoom.ts` for each slug in
 * `src/overview/zooms/index.ts`), so a new zoom needs no change here.
 *  4. At 320px, the Overview's closed sources footer is not shorter than a
 *     320px phone's screen, 568px (#604): checked on every case above and on
 *     the `seasoned` golden, in light and dark.
 *
 * Layout facts only; taste stays with the design-compliance pass.
 *
 * It serves `dist/` with `vite preview` itself (the lane server's read
 * functions, so no secrets and no network), on a free port. `--prove` adds
 * the pre-fix Solar window CSS (a0c7d57^) to the `live` case and exits 1
 * unless the check then reports touching stats, so the check is shown to
 * catch #395. Uses the `playwright` dev dependency's Chromium
 * (`npx playwright install chromium` once).
 */
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";
import { preview } from "vite";

const WIDTHS = [320, 390, 760, 1440];
const SCHEMES = ["light", "dark"];
const TABS = ["overview", "rain", "events", "solar"];
const CASES = [
  { name: "dry", query: "fixture=dry&age=3" },
  { name: "raining", query: "fixture=raining&age=3" },
  { name: "live", query: "fixture=raining&age=3", wide: true },
];

/** The pre-fix Solar window strip (before a0c7d57): nowrap values, 140px columns. */
const PRE_FIX_CSS = `
.solar-stats { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)) !important; }
.solar-stats dd { white-space: nowrap !important; overflow-wrap: normal !important; }
`;

const prove = process.argv.includes("--prove");

/** Give the windows the wide values live data holds, all valid. */
function widen(dashboard) {
  const wide = [4.12, 5.87, 38.5, 284.4, 99999.9, 1188.2, 54321.1];
  for (const [tab, field] of [
    ["solar_tab", "energy_kwh_m2"],
    ["rain_tab", "total_in"],
  ]) {
    dashboard[tab]?.windows.forEach((window, i) => {
      window[field] = wide[i % wide.length];
      window.status = { ...window.status, status: "valid", reason: null, missing_days: 0 };
    });
  }
  return dashboard;
}

/** Runs in the page: the layout facts that are wrong in the open tab. */
function measure() {
  const found = [];
  const panel = document.querySelector("[role=tabpanel]:not([hidden])") ?? document.body;
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && el.checkVisibility();
  };
  const label = (el) =>
    `${el.tagName.toLowerCase()}${[...el.classList]
      .slice(0, 2)
      .map((c) => `.${c}`)
      .join("")}`;
  const tile = (el) => {
    const zoom = el.closest(".zoom");
    if (zoom) return `zoom ${zoom.dataset.zoom}`;
    return el.closest("[data-section]")?.dataset.section ?? "page";
  };

  // 2. a chart narrower than the box that holds it
  const charts = [...panel.querySelectorAll("svg[role=img], .js-plotly-plot")].filter(
    (el) => visible(el) && !el.closest("button, summary"),
  );
  for (const chart of charts) {
    const host = chart.parentElement;
    const style = getComputedStyle(host);
    const content =
      host.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const width = chart.getBoundingClientRect().width;
    if (width < content - 1.5) {
      // (1.5px of sub-pixel rounding is not a defect)
      found.push(
        `chart ${label(chart)} in ${tile(chart)} is ${width.toFixed(0)}px, its ${label(host)} holds ${content.toFixed(0)}px`,
      );
    }
  }

  // 3. stat values that touch: text boxes of different items of one dl
  for (const dl of panel.querySelectorAll("dl")) {
    if (!visible(dl)) continue;
    const boxes = [];
    for (const item of dl.children) {
      const walker = document.createTreeWalker(item, NodeFilter.SHOW_TEXT);
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        if (!node.textContent.trim()) continue;
        // Screen-reader-only text (`.gap-sr`, `.ov-sr`) is clipped, not drawn.
        if (node.parentElement.closest(".gap-sr, .ov-sr")) continue;
        const range = document.createRange();
        range.selectNodeContents(node);
        for (const r of range.getClientRects()) {
          if (r.width > 0 && r.height > 0) boxes.push({ item, r, text: node.textContent.trim() });
        }
      }
    }
    for (let a = 0; a < boxes.length; a++) {
      for (let b = a + 1; b < boxes.length; b++) {
        if (boxes[a].item === boxes[b].item) continue;
        const x =
          Math.min(boxes[a].r.right, boxes[b].r.right) - Math.max(boxes[a].r.left, boxes[b].r.left);
        const y =
          Math.min(boxes[a].r.bottom, boxes[b].r.bottom) - Math.max(boxes[a].r.top, boxes[b].r.top);
        if (x > 1 && y > 1) {
          found.push(
            `stats touch in ${tile(dl)} ${label(dl)}: "${boxes[a].text}" and "${boxes[b].text}"`,
          );
        }
      }
    }
  }
  const stats = [...panel.querySelectorAll("dl")].filter(visible).length;
  return { found, charts: charts.length, stats };
}

/** Page-level sideways scroll, in the open tab. */
async function scrolls(page) {
  return page.evaluate(() => {
    const { scrollWidth, clientWidth } = document.documentElement;
    return scrollWidth > clientWidth
      ? `page scrolls sideways (${scrollWidth}/${clientWidth})`
      : null;
  });
}

/** How many charts and stat lists were measured, so a silent run shows. */
const seen = { charts: 0, stats: 0 };

/** Facts for the open tab; on a phone, each folded row opened in turn. */
async function inspectTab(page) {
  const problems = [];
  const take = async () => {
    const sideways = await scrolls(page);
    if (sideways) problems.push(sideways);
    const result = await page.evaluate(measure);
    problems.push(...result.found);
    seen.charts += result.charts;
    seen.stats += result.stats;
  };
  await take();
  const rows = await page.$$("[role=tabpanel]:not([hidden]) details.tile-row");
  for (const row of rows) {
    await row.evaluate((d) => {
      d.open = true;
    });
    await page.waitForTimeout(150);
    await take();
  }
  return problems;
}

/**
 * The Overview's sources footer, closed, is shorter than one screen (#604).
 * The 844px check viewport is a tall phone, so the bar is the shortest common
 * one: 568px, a 320px-wide phone's height.
 */
const FOOTER_MAX_PX = 568;

/** The closed sources footer's height against `FOOTER_MAX_PX`, as a problem. */
async function footerFits(page) {
  const footer = await page.evaluate(() => {
    const node = document.querySelector("[role=tabpanel]:not([hidden]) footer.ov-sources");
    return node
      ? {
          height: node.getBoundingClientRect().height,
          open: Boolean(node.querySelector("details")?.open),
        }
      : null;
  });
  if (!footer) return ["no sources footer"];
  if (footer.open) return ["sources footer disclosure open on load"];
  return footer.height < FOOTER_MAX_PX
    ? []
    : [`closed sources footer ${Math.round(footer.height)}px, not under ${FOOTER_MAX_PX}px`];
}

const root = fileURLToPath(new URL("..", import.meta.url));
// An optional base URL checks a server already running (as `shots.mjs` does);
// without one the script serves `dist/` itself.
const given = process.argv.slice(2).find((arg) => !arg.startsWith("--"));
const server = given
  ? null
  : await preview({ root, preview: { port: 0, host: "127.0.0.1", strictPort: false } });
const base = (given ?? server.resolvedUrls.local[0]).replace(/\/$/, "");
const browser = await chromium.launch();
const failures = [];
const started = Date.now();
let proved = false;
let zoomViews = 0;

/**
 * The hero's Rain column (#619) stands on the ribbon chart's x-axis line:
 * its floor and the chart's rain-track floor at one height on the page.
 */
async function heroRainOnBaseline(page) {
  return page.evaluate(() => {
    const column = document.querySelector("[data-part=rain-column] .hero-rain-floor");
    const chart = document.querySelector(".today-chart .today-rain-floor");
    if (!column || !column.getBoundingClientRect().width) return ["hero Rain column not shown"];
    const a = column.getBoundingClientRect().top;
    const b = chart.getBoundingClientRect().top;
    return Math.abs(a - b) > 0.5
      ? [`hero Rain column floor at ${a.toFixed(1)}px, the ribbon's x-axis at ${b.toFixed(1)}px`]
      : [];
  });
}

/** Print a view's result and keep its problems as failures. */
function report(view, found, kase) {
  const problems = [...new Set(found)];
  console.log(`${problems.length ? "FAIL" : "ok  "} ${view}`);
  for (const p of problems) {
    console.log(`       ${p}`);
    // Under --prove only the wide case's touching stats are the
    // expected failure; anything else still fails the run.
    if (prove && kase.wide && p.startsWith("stats touch")) proved = true;
    else failures.push(`${view}: ${p}`);
  }
}
try {
  for (const kase of CASES) {
    for (const width of WIDTHS) {
      for (const colorScheme of SCHEMES) {
        const page = await browser.newPage({ viewport: { width, height: 844 }, colorScheme });
        if (kase.wide) {
          await page.route("**/dashboard*", async (route) => {
            const response = await route.fetch();
            await route.fulfill({ response, json: widen(await response.json()) });
          });
        }
        page.on("pageerror", (e) =>
          failures.push(`${kase.name} ${width}px ${colorScheme}: page error ${e.message}`),
        );
        await page.goto(`${base}/?${kase.query}#overview`, { waitUntil: "load" });
        await page.waitForSelector(".masthead-age:not(:empty)", { timeout: 5000 });
        if (kase.wide && prove) await page.addStyleTag({ content: PRE_FIX_CSS });
        for (const tab of TABS) {
          await page.evaluate((id) => (location.hash = id), tab);
          if (tab === "events" && (await page.$(".ev-chart"))) {
            await page.waitForSelector(".ev-chart.js-plotly-plot", { timeout: 10000 });
          }
          await page.waitForTimeout(300);
          const found = await inspectTab(page);
          if (tab === "overview" && width === 320) found.push(...(await footerFits(page)));
          report(`${kase.name} #${tab} ${width}px ${colorScheme}`, found, kase);
        }
        // Every zoom the Overview registers (#474): its tiles' links carry
        // `data-zoom`, so a zoom a lane registers is measured with no change here.
        await page.evaluate(() => (location.hash = "overview"));
        await page.waitForTimeout(150);
        const zooms = await page.$$eval("a[data-zoom]", (links) => [
          ...new Set(links.map((a) => a.dataset.zoom)),
        ]);
        for (const slug of zooms) {
          await page.evaluate((s) => (location.hash = `overview/${s}`), slug);
          await page.waitForSelector(`.zoom[data-zoom="${slug}"]`, { timeout: 5000 });
          await page.waitForTimeout(300);
          const view = `${kase.name} #overview/${slug} ${width}px ${colorScheme}`;
          report(view, await inspectTab(page), kase);
          zoomViews += 1;
        }
        await page.close();
      }
    }
  }
  // The sources footer on the third golden too (#604: every fixture).
  for (const colorScheme of SCHEMES) {
    const page = await browser.newPage({ viewport: { width: 320, height: 844 }, colorScheme });
    await page.goto(`${base}/?fixture=seasoned&age=3#overview`, { waitUntil: "load" });
    await page.waitForSelector(".masthead-age:not(:empty)", { timeout: 5000 });
    await page.waitForTimeout(300);
    report(`seasoned #overview footer 320px ${colorScheme}`, await footerFits(page), {});
    await page.close();
  }
  // The rain gauge on the golden with earlier years (#619), at phone widths.
  for (const width of [320, 390]) {
    for (const colorScheme of SCHEMES) {
      const page = await browser.newPage({ viewport: { width, height: 844 }, colorScheme });
      await page.goto(`${base}/?fixture=seasoned&age=3#overview/rain`, { waitUntil: "load" });
      await page.waitForSelector("[data-panel=rain-windows] svg", { timeout: 5000 });
      await page.waitForTimeout(300);
      report(`seasoned #overview/rain ${width}px ${colorScheme}`, await inspectTab(page), {});
      zoomViews += 1;
      await page.close();
    }
  }
  // The hero's Rain column on the ribbon's baseline (#619), on the grids.
  for (const width of [900, 1280]) {
    for (const colorScheme of SCHEMES) {
      const page = await browser.newPage({ viewport: { width, height: 844 }, colorScheme });
      await page.goto(`${base}/?fixture=raining&age=3#overview`, { waitUntil: "load" });
      await page.waitForSelector(".masthead-age:not(:empty)", { timeout: 5000 });
      await page.waitForTimeout(300);
      report(`raining hero rain ${width}px ${colorScheme}`, await heroRainOnBaseline(page), {});
      await page.close();
    }
  }
} finally {
  await browser.close();
  await server?.close();
}
const seconds = ((Date.now() - started) / 1000).toFixed(0);
console.log(
  `${CASES.length * WIDTHS.length * SCHEMES.length * TABS.length + zoomViews} views (${zoomViews} zooms) in ${seconds}s, ${seen.charts} charts and ${seen.stats} stat lists measured`,
);

if (failures.length) {
  console.error(`${failures.length} layout problem(s):\n${failures.join("\n")}`);
  process.exit(1);
}
if (prove) {
  if (!proved) {
    console.error("--prove: the #395 shape did not fail the check");
    process.exit(1);
  }
  console.log("--prove: the #395 shape fails the check, as it should");
  process.exit(0);
}
console.log("no layout problems");
