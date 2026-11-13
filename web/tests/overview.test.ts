// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from "vitest";

import { rainStep } from "../src/charts/rain-step.js";
import { setLayout } from "../src/layout.js";
import { renderOverview, SECTIONS } from "../src/overview/index.js";
import { releaseLine, SITE_RELEASE } from "../src/release.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const raining = parseDashboard(golden("raining", "dashboard"));
const dry = parseDashboard(golden("dry", "dashboard"));

let panel: HTMLElement;

beforeEach(() => {
  // No layout set: the phone rules apply, as on a page before its switch runs.
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const text = (selector: string) =>
  panel.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim();
const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const all = (selector: string) => [...panel.querySelectorAll<HTMLElement>(selector)];

describe("the Overview", () => {
  it("opens with the now tile, in the Phone First-Screen order", () => {
    renderOverview(panel, raining);

    const first = panel.querySelector(".tile-grid > .tile")!;
    expect(first.getAttribute("data-section")).toBe("first-screen");
    expect(first.getAttribute("data-surface")).toBe("now");
    const order = [...first.querySelectorAll("[data-part]")].map((el) =>
      el.getAttribute("data-part"),
    );
    // The raining golden had rain today: the hero's rain line and column (#619).
    expect(order).toEqual([
      "kicker",
      "finding",
      "temperature",
      "delta",
      "rain-line",
      "today-chart",
      "rain-column",
    ]);
  });

  it("renders each section into its own tile, in order, after its kicker", () => {
    // The grids draw every section as a tile; a phone draws rows (tiles.test).
    setLayout("four");
    const seen: string[] = [];
    renderOverview(panel, dry, [
      {
        id: "a",
        label: "A",
        kicker: "Kicker A",
        render: (host) => void seen.push(`${host.dataset.section} ${host.textContent}`),
      },
      {
        id: "b",
        label: "B",
        kicker: (s) => `Kicker for ${s.station.name}`,
        span: 3,
        render: (host) => void seen.push(`${host.dataset.section} ${host.textContent}`),
      },
    ]);

    expect(seen).toEqual(["a Kicker A", `b Kicker for ${dry.station.name}`]);
    // No tile is a named region: its content names it (#472).
    expect(all(".tile").map((el) => el.getAttribute("aria-label"))).toEqual([null, null]);
    expect(all(".tile").map((el) => el.dataset.span)).toEqual(["1", "3"]);
  });

  it("starts its section list with the now tile", () => {
    expect(SECTIONS[0]?.id).toBe("first-screen");
  });
});

describe("the current temperature", () => {
  it("shows the live temperature large", () => {
    renderOverview(panel, raining);

    expect(text("[data-part=temperature]")).toBe("59.3°F");
  });

  it("shows n/a when the station reports no temperature", () => {
    renderOverview(panel, { ...dry, current: { ...dry.current, temperature_f: null } });

    expect(text("[data-part=temperature]")).toBe("n/a");
  });
});

describe("the Overview's sources footer (prototype decision 22, #604)", () => {
  /** Each short line: the tile's name, then its source and comparison. */
  const lines = () =>
    all(".ov-sources-list > div").map((item) =>
      [...item.children].map((child) => child.textContent),
    );
  /** Each method paragraph in the disclosure: the tile's name, then its method. */
  const methods = () =>
    all(".ov-sources-how [data-source]").map((item) =>
      [...item.children].map((child) => child.textContent),
    );
  const NAMES = [
    "Right now",
    "The feature",
    "Temperature",
    "Rain since Oct 1",
    "Last 10 days",
    "Dry spell",
    "Wind",
    "Pressure",
    "Humidity",
    "Sun today",
    "Daylight",
    "Inside",
  ];
  const wordCount = (s: string) => s.trim().split(/\s+/).length;

  it("folds no source on any Overview tile, and lists every drawn tile once at the foot", () => {
    renderOverview(panel, raining);

    expect(panel.querySelector(".tile-grid details, .tile-grid .ov-source")).toBeNull();
    const footer = panel.querySelector<HTMLElement>("footer.ov-sources")!;
    expect(panel.lastElementChild).toBe(footer);
    expect(words(footer.querySelector("h2"))).toBe("Sources");
    expect(lines().map(([name]) => name)).toEqual(NAMES);
  });

  it.each(["dry", "raining", "seasoned"] as const)(
    "gives each tile one short line, at most 12 words, on the %s golden",
    (scenario) => {
      renderOverview(panel, parseDashboard(golden(scenario, "dashboard")));

      for (const item of lines()) {
        expect(item).toHaveLength(2);
        const line = item[1]!;
        expect(wordCount(line), line).toBeLessThanOrEqual(12);
        expect(line.length).toBeGreaterThan(5);
      }
    },
  );

  it("keeps a left-out tile's reason to the same cap, with no method", () => {
    renderOverview(panel, { ...raining, sun_moon: null });

    const daylight = lines().find(([name]) => name === "Daylight")!;
    expect(wordCount(daylight[1]!)).toBeLessThanOrEqual(12);
    expect(methods().map(([name]) => name)).not.toContain("Daylight");
  });

  it("keeps the method paragraphs behind one closed disclosure, after the lines", () => {
    renderOverview(panel, raining);
    const footer = panel.querySelector<HTMLElement>("footer.ov-sources")!;

    const disclosures = footer.querySelectorAll("details");
    expect(disclosures).toHaveLength(1);
    const how = disclosures[0]!;
    expect(how.open).toBe(false);
    expect(how.classList.contains("ov-sources-how")).toBe(true);
    expect(words(how.querySelector("summary"))).toBe("How these are worked out");
    expect(footer.querySelector(".ov-sources-list")!.compareDocumentPosition(how)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
    expect(methods().map(([name]) => name)).toEqual(NAMES);
    for (const [, method] of methods()) expect(method!.length).toBeGreaterThan(40);
    // No method paragraph outside the disclosure.
    expect(footer.querySelectorAll(".ov-sources-method")).toHaveLength(
      how.querySelectorAll(".ov-sources-method").length,
    );
  });

  it("ends with the release version and short SHA", () => {
    renderOverview(panel, raining);
    const footer = panel.querySelector<HTMLElement>("footer.ov-sources")!;

    expect(footer.lastElementChild).toBe(footer.querySelector(".ov-sources-version"));
    expect(words(footer.querySelector(".ov-sources-version"))).toBe(releaseLine());
    expect(words(footer.querySelector(".ov-sources-version"))).toContain(
      `${SITE_RELEASE.version} · ${SITE_RELEASE.sha}`,
    );
  });

  it("repeats no archive or reading time: the header says it once", () => {
    renderOverview(panel, raining);
    const footer = words(panel.querySelector("footer.ov-sources"))!;

    expect(footer).not.toMatch(/archived|\d:\d\d|[ap]m\b/);
  });

  it("words the now tile's band from its basis", () => {
    renderOverview(panel, raining);

    expect(methods()[0]![1]).toContain(
      "The band is the usual for each hour of the day: the median of your last 4 days",
    );
  });

  it("leaves out the line of a tile not drawn", () => {
    const d = structuredClone(golden("raining", "dashboard")) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
    delete d.feature;
    for (const t of d.record_depth.tiles) if (t.key === "rain") t.shown = false;
    renderOverview(panel, parseDashboard(d));

    const names = lines().map(([name]) => name);
    expect(names).not.toContain("The feature");
    expect(names).not.toContain("Rain since Oct 1");
    expect(methods().map(([name]) => name)).toEqual(names);
  });
});

describe("the now tile without a ribbon", () => {
  it("still draws the chart when there is no ribbon", () => {
    const none: DashboardSnapshot = { ...dry, condition_ribbon: null };
    renderOverview(panel, none);

    expect(panel.querySelector("[data-part=today-chart] svg")).not.toBeNull();
    expect(panel.querySelector("[data-part=temperature]")).not.toBeNull();
  });
});

describe("the rain steps", () => {
  it("tell four intensities apart by rate", () => {
    expect(rainStep(0.0125)).toBe(1); // 0.05 in/hr
    expect(rainStep(0.025)).toBe(2); // 0.10 in/hr
    expect(rainStep(0.075)).toBe(3); // 0.30 in/hr
    expect(rainStep(0.1)).toBe(4);
    expect(rainStep(0)).toBeNull();
  });
});
