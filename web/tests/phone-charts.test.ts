// @vitest-environment happy-dom
/**
 * The Rain Events and Rain charts at phone width (#149): the cumulative
 * line is a line, the event time axis reads plain times with the date said
 * once, the water-year chart keeps its labels inside and ticks every month
 * to Sep, the event table folds its quality into the row, and the list
 * leads with a finding.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const react = vi.fn(async () => {});
vi.mock("plotly.js-dist-min", () => ({ default: { react } }));

import { LABEL_CHAR_PX } from "../src/charts/svg.js";
import { drawEventCharts, timeTicks } from "../src/events/charts.js";
import { listFinding } from "../src/events/model.js";
import type { EventsDeps } from "../src/events/tab.js";
import { mountEvents } from "../src/events/tab.js";
import { drawYearCurves } from "../src/rain/charts.js";
import type { EventCurve, RainEventRow, RainTab, WaterYearCurve } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const HOUR = 3_600_000;
/** Jan 8, 2026, 2 pm Pacific. */
const START = Date.UTC(2026, 0, 8, 22);

/** A curve of readings every 5 minutes for `hours`, rain rising evenly. */
function curve(hours: number): EventCurve {
  const n = (hours * 60) / 5 + 1;
  const t_ms = Array.from({ length: n }, (_, i) => START + i * 300_000);
  return {
    start_ms: START,
    t_ms,
    event_in: t_ms.map((_, i) => i * 0.01),
    rate_in_hr: t_ms.map(() => 0.12),
    gap_minutes: t_ms.map(() => null),
  } as unknown as EventCurve;
}

function row(change: Partial<RainEventRow> = {}): RainEventRow {
  return {
    start_ms: START,
    end_ms: START + 20 * HOUR,
    total_in: 2.4,
    duration_minutes: 1200,
    peak_rate_in_hr: 0.12,
    quality: "excellent",
    completeness_pct: 100,
    max_gap_minutes: 0,
    water_year: 2026,
    rank: 1,
    live: false,
    has_curve: true,
    ...change,
  } as RainEventRow;
}

/** The Rain Events tab's sources, faked: no network, no Plotly. */
function deps(): EventsDeps {
  return { fetchCurve: async () => curve(1), draw: async () => {}, nowMs: () => Date.UTC(2026, 0, 9, 0, 30) };
}

beforeEach(() => {
  react.mockClear();
  document.body.innerHTML = "";
});

describe("the event charts", () => {
  async function draw(): Promise<{ data: Record<string, unknown>[]; layout: Record<string, unknown> }[]> {
    const accumulation = document.createElement("div");
    const rate = document.createElement("div");
    await drawEventCharts({ accumulation, rate }, curve(20), row());
    return react.mock.calls.map((call) => {
      const [, data, layout] = call as unknown as [unknown, Record<string, unknown>[], Record<string, unknown>];
      return { data, layout };
    });
  }

  it("draws rain so far as a line with nothing filled under it", async () => {
    const [acc] = await draw();
    const lines = acc!.data.filter((t) => t.type === "scatter");

    expect(lines.length).toBeGreaterThan(0);
    for (const line of lines) expect(line.fill ?? "none").toBe("none");
  });

  it("draws the whole end dot, though it sits on the plot's edge", async () => {
    const [acc] = await draw();
    const dot = acc!.data.find((t) => t.mode === "markers")!;

    expect(dot.cliponaxis).toBe(false);
  });

  it("gives both charts the same plain-time ticks", async () => {
    const [acc, rate] = await draw();
    const xa = acc!.layout.xaxis as Record<string, unknown>;
    const xr = rate!.layout.xaxis as Record<string, unknown>;

    expect(xr.tickmode).toBe("array");
    expect(xr.ticktext).toEqual(xa.ticktext);
    expect(xr.tickvals).toEqual(xa.tickvals);
  });
});

describe("the event time axis", () => {
  it("reads plain times, the date said once and again where the day turns", () => {
    // 2 pm Jan 8 to 10 am Jan 9, Pacific, on a phone's plot width.
    const ticks = timeTicks(START, START + 20 * HOUR, 294);

    expect(ticks.text).toEqual(["6 pm<br>Jan 8", "Jan 9", "6 am"]);
    expect(ticks.values).toEqual(["2026-01-08 18:00:00", "2026-01-09 00:00:00", "2026-01-09 06:00:00"]);
    expect(ticks.text.join(" ")).not.toMatch(/\d\d:\d\d/);
  });

  it("ticks every hour of a few hours' event", () => {
    const ticks = timeTicks(START, START + 3 * HOUR, 340);

    expect(ticks.text).toEqual(["2 pm<br>Jan 8", "3 pm", "4 pm", "5 pm"]);
  });

  it("ticks quarter hours for an event shorter than an hour", () => {
    const ticks = timeTicks(START + 10 * 60_000, START + 40 * 60_000, 294);

    expect(ticks.text).toEqual(["2:15 pm<br>Jan 8", "2:30 pm"]);
  });
});

/** A water year rising evenly to `total` over `days`, ≈ when prorated. */
function year(water_year: number, days: number, total: number, prorated = false): WaterYearCurve {
  const cumulative = Array.from({ length: days }, (_, i) => (total * (i + 1)) / days);
  return {
    water_year,
    first_day: `${water_year - 1}-10-01`,
    cumulative_in: cumulative,
    prorated: cumulative.map(() => prorated),
    status: {
      key: `water_year_${water_year}`,
      first_day: `${water_year - 1}-10-01`,
      last_day: `${water_year}-09-30`,
      days,
      missing_days: 0,
      status: prorated ? "prorated" : "valid",
      reason: prorated ? "gap" : null,
    },
    total_in: total,
    to_date_in: total,
  } as WaterYearCurve;
}

describe("the water-year chart at phone width", () => {
  const WIDTH = 358; // 390px less the 16px gutters
  const tab = {
    water_years: [year(2024, 365, 48.6, true), year(2025, 365, 39.2), year(2026, 30, 3.1)],
  } as unknown as RainTab;

  function chart(): SVGSVGElement {
    const host = document.createElement("div");
    drawYearCurves(host, tab, WIDTH, () => {});
    return host.querySelector("svg")!;
  }

  it("keeps every label inside the chart", () => {
    const labels = [...chart().querySelectorAll("text")];

    expect(labels.some((t) => t.textContent === "2024 ≈ 48.6 in")).toBe(true);
    for (const label of labels) {
      const x = Number(label.getAttribute("x"));
      const room = label.textContent!.length * LABEL_CHAR_PX;
      const anchor = label.getAttribute("text-anchor") ?? "start";
      const [from, to] = anchor === "end" ? [x - room, x] : anchor === "middle" ? [x - room / 2, x + room / 2] : [x, x + room];
      expect(from, label.textContent!).toBeGreaterThanOrEqual(0);
      expect(to, label.textContent!).toBeLessThanOrEqual(WIDTH);
    }
  });

  it("ticks and names every month from Oct to Sep", () => {
    const months = [...chart().querySelectorAll("text.rain-month")].map((t) => t.textContent);

    expect(months).toEqual(["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]);
  });
});

describe("the event table at phone width", () => {

  it("folds a low quality into the event's row and marks the column to drop", () => {
    history.replaceState(null, "", "/?fixture=raining#events");
    document.body.innerHTML = "<section id=panel></section>";
    const panel = document.getElementById("panel")!;
    const snapshot = parseDashboard(structuredClone(golden("raining", "dashboard")));
    snapshot.rain_events.events[2]!.quality = "poor";
    mountEvents(panel, snapshot, deps());
    const rows = [...panel.querySelectorAll(".ev-table tbody tr")];

    expect(panel.querySelector(".ev-table thead .ev-quality")!.textContent).toBe("Quality");
    expect(rows.every((r) => r.querySelector("td.ev-quality .ev-chip"))).toBe(true);
    const folded = [...panel.querySelectorAll(".ev-table tbody th .ev-chip-fold")];
    expect(folded.map((chip) => chip.textContent)).toEqual(["poor"]);
  });
});

describe("the event table's start column", () => {
  it("can break an event's start between its date and its time", () => {
    history.replaceState(null, "", "/?fixture=raining#events");
    document.body.innerHTML = "<section id=panel></section>";
    const panel = document.getElementById("panel")!;
    const snapshot = parseDashboard(structuredClone(golden("raining", "dashboard")));
    // A year later, so every start carries its year: "Jan 4, 2026, 10:00 am".
    mountEvents(panel, snapshot, { ...deps(), nowMs: () => Date.UTC(2027, 5, 1) });
    const start = panel.querySelector(".ev-table tbody tr:last-child .ev-open")!;

    const parts = [...start.querySelectorAll(".ev-when")].map((s) => s.textContent);
    expect(parts).toHaveLength(2);
    expect(parts[0]).toMatch(/^[A-Z][a-z]{2} \d{1,2}, 2026,$/);
    expect(parts[1]).toMatch(/^\d{1,2}:\d\d [ap]m$/);
  });
});

describe("the event list's headline", () => {
  const events = [row({ total_in: 0.48, start_ms: 2 }), row({ total_in: 0.3, start_ms: 3 }), row({ total_in: 0.36, start_ms: 1 })];
  const stats = { count: 3, total_in: 1.14, median_in: 0.36, mean_duration_minutes: 541 };

  it("leads with the wettest event against the median", () => {
    expect(listFinding(stats, "all", Date.UTC(2025, 9, 27, 18), events)).toBe(
      "The wettest of 3 events since Oct 27, 2025, brought 0.48 in, against 0.36 in for the median event",
    );
  });

  it("says the same for one water year", () => {
    expect(listFinding(stats, 2026, null, events)).toBe(
      "The wettest of 3 events in water year 2026 brought 0.48 in, against 0.36 in for the median event",
    );
  });

  it("names a lone event and an empty water year plainly", () => {
    const one = { count: 1, total_in: 0.36, median_in: 0.36, mean_duration_minutes: 530 };
    expect(listFinding(one, 2025, null, [events[2]!])).toBe("The one rain event in water year 2025 brought 0.36 in");
    const none = { count: 0, total_in: 0, median_in: null, mean_duration_minutes: null };
    expect(listFinding(none, 2024, null, [])).toBe("No rain events in water year 2024");
  });

  it("says so far when the wettest event is still raining", () => {
    const live = [row({ total_in: 0.9, live: true }), ...events];
    const four = { ...stats, count: 4 };
    expect(listFinding(four, 2026, null, live)).toBe(
      "The wettest of 4 events in water year 2026 brought 0.90 in so far, against 0.36 in for the median event",
    );
  });
});
