// @vitest-environment happy-dom
/**
 * Chart text over a hatch sits on a pill (#372, DESIGN.md Shapes: "Text
 * never sits on the stripes"). Through `startPage` on the goldens: every
 * reason over a hatched span and the now labels that can land on a hatch draw on a `pillLabel` pill of the
 * surface, 4px past the text; the periods chart draws its reasons after
 * the now rule, so the rule never crosses them; the ten-day list's day
 * names end clear of the hatch. The Rain events charts (Plotly) give each
 * label a box of the tile's surface with the same 4px pad.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const react = vi.fn(async () => {});
vi.mock("plotly.js-dist-min", () => ({ default: { react } }));

import { LABEL_CHAR_PX, PILL_PAD } from "../src/charts/svg.js";
import { drawEventChart } from "../src/events/charts.js";
import { reasonLabel } from "../src/gaps.js";
import { setLayout } from "../src/layout.js";
import { renderOverview } from "../src/overview/index.js";
import { temperaturePeriods, tenDays } from "../src/overview/temperature.js";
import type { Page } from "../src/page.js";
import { startPage } from "../src/page.js";
import type {
  CurrentReading,
  DashboardSnapshot,
  EventCurve,
  GapReason,
  RainEventRow,
} from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { SCENARIOS, golden } from "./golden.js";

let page: Page | null = null;

beforeEach(() => {
  react.mockClear();
  document.body.innerHTML =
    "<div class=page><div id=masthead></div><main id=sections></main></div>";
});

afterEach(() => {
  page?.stop();
  page = null;
});

/** Boot the page on a golden at its first tab, then draw the others. */
async function boot(scenario: Scenario, ...tabs: string[]): Promise<void> {
  const snapshot: DashboardSnapshot = parseDashboard(golden(scenario, "dashboard"));
  history.replaceState(null, "", `/#${tabs[0]}`);
  globalThis.fetch = (() => new Promise<Response>(() => {})) as typeof fetch;
  page = startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
    dashboard: () => Promise.resolve(snapshot),
    current: () => new Promise<CurrentReading>(() => {}),
  });
  await page.booted;
  for (const tab of tabs.slice(1)) {
    document.querySelector<HTMLElement>(`[aria-controls=panel-${tab}]`)!.click();
  }
}

/**
 * The temperature sections that left the Overview for its zoom (#472), drawn
 * as tiles after the page.
 */
function drawZoom(scenario: Scenario): void {
  setLayout("four");
  const host = document.createElement("div");
  document.body.append(host);
  renderOverview(host, parseDashboard(golden(scenario, "dashboard")), [
    temperaturePeriods,
    tenDays,
  ]);
}

/** Each reason, and the Overview's plain wording of it, without "n/a" (#470). */
const REASONS = new Set(
  (["gap", "low_coverage", "not_archived", "before_archive", "in_progress"] as GapReason[])
    .map(reasonLabel)
    .flatMap((label) => [label, label.replace(/^n\/a: /, "")]),
);

const texts = (root: ParentNode) => [...root.querySelectorAll<SVGTextElement>("svg text")];

/** The pill a label sits on, checked for its shape: a surface rect, then the text. */
function pillOf(text: SVGTextElement): SVGRectElement {
  const group = text.parentElement!;
  expect(group.classList.contains("chart-pill"), text.textContent!).toBe(true);
  expect(group.dataset.part).toBe("pill");
  const rect = group.firstElementChild as SVGRectElement;
  expect(rect.tagName.toLowerCase()).toBe("rect");
  expect(rect.classList.contains("chart-pill-ground")).toBe(true);
  expect(group.lastElementChild).toBe(text);
  return rect;
}

/** A pill reaches at least 4px past its text on each side (by the estimate). */
function expectPadded(text: SVGTextElement, rect: SVGRectElement): void {
  const width = Number(rect.getAttribute("width"));
  const height = Number(rect.getAttribute("height"));
  expect(width).toBeGreaterThanOrEqual(text.textContent!.length * LABEL_CHAR_PX + 2 * PILL_PAD);
  expect(height).toBeGreaterThanOrEqual(12 + 2 * PILL_PAD);
}

describe.each(SCENARIOS)("the %s golden's hatch labels", (scenario) => {
  it("puts every reason over a hatch on a pill, on every SVG tab", async () => {
    await boot(scenario, "overview", "rain", "solar");
    drawZoom(scenario);
    const reasons = texts(document).filter((t) => REASONS.has(t.textContent ?? ""));

    // The dry golden lags its archive, so some window reads "not archived
    // yet"; the seasoned one has gaps. The raining golden draws only what its
    // record covers, all of it comparable (#470).
    expect(reasons.length > 0).toBe(scenario !== "raining");
    for (const reason of reasons) expectPadded(reason, pillOf(reason));
  });

  it("draws the periods chart's reasons after its now rule and dot", () => {
    drawZoom(scenario);
    const chart = document.querySelector("[data-part=periods-chart] svg")!;
    const marks = [...chart.querySelectorAll(".tc-now, .tc-now-dot")];
    const reasons = [...chart.querySelectorAll(".tc-reason")];

    expect(marks.length).toBeGreaterThan(0);
    // The raining golden's chart draws only the windows its record covers,
    // each comparable (#470): no hatch, so no reason to place.
    expect(reasons.length > 0).toBe(scenario !== "raining");
    for (const reason of reasons) {
      for (const mark of marks) {
        expect(
          mark.compareDocumentPosition(reason) & Node.DOCUMENT_POSITION_FOLLOWING,
        ).toBeTruthy();
      }
    }
  });

  it("ends each ten-day name clear of the hatch beside it", () => {
    drawZoom(scenario);
    const chart = document.querySelector("[data-part=days-chart] svg")!;
    const plot = chart.querySelector<SVGGElement>("g[transform]")!;
    const left = Number(/translate\(([\d.]+)/.exec(plot.getAttribute("transform")!)![1]);
    const names = [...chart.querySelectorAll<SVGTextElement>(".tc-row")];
    const shownDays = parseDashboard(
      golden(scenario, "dashboard"),
    ).temperature_context.ten_days.filter((d) => d.status.shown !== false);

    // The days the record covers (#470).
    expect(names.length).toBe(shownDays.length);
    for (const name of names) {
      const end = Number(name.getAttribute("x")) + name.textContent!.length * LABEL_CHAR_PX * 1.1;
      expect(left - end).toBeGreaterThanOrEqual(2 * PILL_PAD);
    }
  });

  it("puts the solar now label on a pill", async () => {
    await boot(scenario, "solar");
    const now = document.querySelector<SVGTextElement>("#panel-solar .solar-now-label")!;

    expect(now.textContent).toMatch(/^now /);
    expectPadded(now, pillOf(now));
  });
});

describe("the dry golden's solar daily bars", () => {
  // The dry golden hatches the selected day itself, with its reason.
  it("draws no selected-day outline over a solar reason", async () => {
    await boot("dry", "solar");
    const charts = [...document.querySelectorAll("#panel-solar svg")].filter((svg) =>
      svg.querySelector(".solar-selected"),
    );

    expect(charts.length).toBeGreaterThan(0);
    expect(charts.some((svg) => svg.querySelector(".chart-pill"))).toBe(true);
    for (const svg of charts) {
      const outline = svg.querySelector(".solar-selected")!;
      for (const pill of svg.querySelectorAll(".chart-pill")) {
        expect(
          outline.compareDocumentPosition(pill) & Node.DOCUMENT_POSITION_FOLLOWING,
        ).toBeTruthy();
      }
    }
  });
});

describe("the event charts (Plotly)", () => {
  const START = Date.UTC(2026, 0, 8, 22);

  function curve(): EventCurve {
    // Readings every 5 minutes, then the live reading 30 minutes on: the tail.
    const t_ms = Array.from({ length: 12 }, (_, i) => START + i * 300_000);
    t_ms.push(t_ms[11]! + 1_800_000);
    return {
      start_ms: START,
      t_ms,
      event_in: t_ms.map((_, i) => i * 0.01),
      rate_in_hr: t_ms.map(() => 0.12),
      gap_minutes: t_ms.map((_, i) => (i === 12 ? 30 : null)),
    } as unknown as EventCurve;
  }

  const row = {
    start_ms: START,
    end_ms: START + 3_600_000,
    total_in: 0.3,
    duration_minutes: 60,
    peak_rate_in_hr: 0.12,
    quality: "excellent",
    completeness_pct: 100,
    max_gap_minutes: 30,
    water_year: 2026,
    rank: 1,
    live: true,
    has_curve: true,
  } as RainEventRow;

  it("boxes every label in the tile's surface, 4px past its text", async () => {
    const host = document.createElement("div");
    host.style.setProperty("--surface", "rgb(1, 2, 3)");
    document.body.append(host);
    await drawEventChart(host, curve(), row, { chart: "total", medianIn: 0.36 });
    const [, data, layout] = react.mock.calls[0] as unknown as [
      unknown,
      Record<string, unknown>[],
      { annotations: Record<string, unknown>[] },
    ];

    expect(layout.annotations.map((a) => a.text)).toContain("only the live <br>total is known");
    for (const note of layout.annotations) {
      expect(note.bgcolor).toBe("rgb(1, 2, 3)");
      expect(note.borderpad).toBe(PILL_PAD);
    }
    // The hatch's ground is the same surface.
    const hatch = data[0]!.marker as { color: string; pattern: { bgcolor: string } };
    expect(hatch.color).toBe("rgb(1, 2, 3)");
    expect(hatch.pattern.bgcolor).toBe("rgb(1, 2, 3)");
  });
});
