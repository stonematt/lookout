// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from "vitest";

import { rainStep } from "../src/charts/rain-step.js";
import { setLayout } from "../src/layout.js";
import { renderOverview, SECTIONS } from "../src/overview/index.js";
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
    expect(order).toEqual([
      "kicker",
      "finding",
      "temperature",
      "delta",
      "today-chart",
      "source-note",
      // No lag chip (#458); the raining golden's today has no gap either.
      "source",
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

describe("the now tile's source note", () => {
  it("explains the hatched end of today in the source line, not in a headline", () => {
    renderOverview(panel, raining);

    // The lag itself is not named (#458): the header states freshness once.
    expect(text(".ov-source")).toBe(
      "The temperature is the station's reading at Jan 8, 4:25 pm. " +
        "Today's line is the archive in 15-minute slots, to Jan 8, 3:55 pm; " +
        "after it the live reading is joined by a dotted line. " +
        "The band is the usual for each hour of the day: the median of your last 4 days.",
    );
  });

  it("dates the reading itself, so it never borrows a fresher one", () => {
    renderOverview(panel, raining);
    expect(text(".ov-source")).toMatch(
      /^The temperature is the station's reading at Jan 8, 4:25 pm\./,
    );
  });

  it("says when there is no reading at all", () => {
    renderOverview(panel, { ...dry, observed_at_ms: null, condition_ribbon: null });
    expect(text(".ov-source")).toBe(
      "The temperature has no reading yet. " +
        "The band is the usual for each hour of the day: the median of your last day.",
    );
  });

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
