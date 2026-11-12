// @vitest-environment happy-dom
/**
 * The charts at phone width (#149, #338): the cumulative line is a line,
 * the event time axis reads plain times with the date said once, the
 * water-year chart keeps its labels inside and ticks every month to Sep,
 * the event table folds its quality into the row, and the list leads with
 * a finding. Under 520px of chart width (the Chart Width Rule) every chart
 * draws its phone layout: the daily bars and the rain-by-hour grid show
 * the last 30 days with a switch to 90, the period rows put each label
 * above its bar, the hero drops its closing midnight, a chart has three y
 * ticks and its now label above its dot, nothing is drawn scaled, and an
 * event chart redraws at its real width and in a new colour scheme.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const react = vi.fn(async () => {});
vi.mock("plotly.js-dist-min", () => ({ default: { react } }));

import { LABEL_CHAR_PX } from "../src/charts/svg.js";
import { drawEventChart, timeTicks } from "../src/events/charts.js";
import { renderOverview } from "../src/overview/index.js";
import { againstNormal, temperaturePeriods, tenDays } from "../src/overview/temperature.js";
import { renderRain } from "../src/rain/index.js";
import { renderSolar } from "../src/solar/index.js";
import { listFinding } from "../src/events/model.js";
import type { EventsDeps } from "../src/events/tab.js";
import { mountEvents } from "../src/events/tab.js";
import { drawYearCurves } from "../src/rain/charts.js";
import type {
  DashboardSnapshot,
  EventCurve,
  RainEventRow,
  RainTab,
  WaterYearCurve,
} from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { SCENARIOS, golden } from "./golden.js";
import { viewport } from "./viewport.js";

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
  return {
    fetchCurve: async () => curve(1),
    draw: async () => {},
    nowMs: () => Date.UTC(2026, 0, 9, 0, 30),
  };
}

beforeEach(() => {
  react.mockClear();
  document.body.innerHTML = "";
});

describe("the event charts", () => {
  async function draw(): Promise<
    { data: Record<string, unknown>[]; layout: Record<string, unknown> }[]
  > {
    const host = document.createElement("div");
    await drawEventChart(host, curve(20), row(), { chart: "total", medianIn: 0.36 });
    await drawEventChart(host, curve(20), row(), { chart: "rate", medianIn: 0.36 });
    return react.mock.calls.map((call) => {
      const [, data, layout] = call as unknown as [
        unknown,
        Record<string, unknown>[],
        Record<string, unknown>,
      ];
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

  it("draws Total against the median event's total, dashed and labelled", async () => {
    const [acc] = await draw();
    const median = acc!.data.find(
      (t) => (t.line as Record<string, unknown> | undefined)?.dash === "dash",
    )!;
    const notes = acc!.layout.annotations as Record<string, unknown>[];

    expect(median.y).toEqual([0.36, 0.36]);
    expect(notes.map((n) => n.text)).toContain("median 0.36 in");
  });

  it("draws Total and Rate at one height, so the switch keeps the row still", async () => {
    const [acc, rate] = await draw();

    expect(acc!.layout.height).toBe(rate!.layout.height);
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
    expect(ticks.values).toEqual([
      "2026-01-08 18:00:00",
      "2026-01-09 00:00:00",
      "2026-01-09 06:00:00",
    ]);
    expect(ticks.text.join(" ")).not.toMatch(/\d\d:\d\d/);
  });

  it("ticks every hour of a few hours' event", () => {
    const ticks = timeTicks(START, START + 3 * HOUR, 340);

    expect(ticks.text).toEqual(["2 pm<br>Jan 8", "3 pm", "4 pm", "5 pm"]);
  });

  it("ticks quarter hours for an event shorter than an hour", () => {
    const ticks = timeTicks(START + 10 * 60_000, START + 40 * 60_000, 294);

    expect(ticks.text).toEqual(["2:15 pm<br>Jan 8", "2:30 pm"]);
    const noon = timeTicks(START - 2 * HOUR + 10 * 60_000, START - 2 * HOUR + 40 * 60_000, 294);
    expect(noon.text).toEqual(["12:15 pm<br>Jan 8", "12:30 pm"]);
  });

  it("thins the daily ticks of an event many days long to fit", () => {
    const ticks = timeTicks(START, START + 30 * 24 * HOUR, 294);

    expect(ticks.text.length).toBeLessThanOrEqual(5);
    expect(ticks.text[0]).toBe("Jan 9");
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
      const [from, to] =
        anchor === "end"
          ? [x - room, x]
          : anchor === "middle"
            ? [x - room / 2, x + room / 2]
            : [x, x + room];
      expect(from, label.textContent!).toBeGreaterThanOrEqual(0);
      expect(to, label.textContent!).toBeLessThanOrEqual(WIDTH);
    }
  });

  it("ticks and names every month from Oct to Sep", () => {
    const months = [...chart().querySelectorAll("text.rain-month")].map((t) => t.textContent);

    expect(months).toEqual(["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]);
  });
});

describe("the event list at phone width", () => {
  it("carries every row's quality chip in its meta line, a poor one dashed", () => {
    history.replaceState(null, "", "/?fixture=raining#events");
    document.body.innerHTML = "<section id=panel></section>";
    const panel = document.getElementById("panel")!;
    const snapshot = parseDashboard(structuredClone(golden("raining", "dashboard")));
    snapshot.rain_events.events[2]!.quality = "poor";
    mountEvents(panel, snapshot, deps());
    const rows = [...panel.querySelectorAll(".ev-list .ev-row")];

    expect(rows.every((r) => r.querySelector(".ev-meta .ev-chip"))).toBe(true);
    const low = [...panel.querySelectorAll(".ev-meta .ev-chip.low")];
    expect(low.map((chip) => chip.textContent)).toEqual(["poor"]);
  });
});

describe("an event row's start", () => {
  it("can break an event's start between its date and its time", () => {
    history.replaceState(null, "", "/?fixture=raining#events");
    document.body.innerHTML = "<section id=panel></section>";
    const panel = document.getElementById("panel")!;
    const snapshot = parseDashboard(structuredClone(golden("raining", "dashboard")));
    // A year later, so every start carries its year: "Jan 4, 2026, 10:00 am".
    mountEvents(panel, snapshot, { ...deps(), nowMs: () => Date.UTC(2027, 5, 1) });
    const start = panel.querySelector(".ev-list .ev-row:last-child .ev-open")!;

    const parts = [...start.querySelectorAll(".ev-when")].map((s) => s.textContent);
    expect(parts).toHaveLength(2);
    expect(parts[0]).toMatch(/^[A-Z][a-z]{2} \d{1,2}, 2026,$/);
    expect(parts[1]).toMatch(/^\d{1,2}:\d\d [ap]m$/);
  });
});

describe("the event list's headline", () => {
  const events = [
    row({ total_in: 0.48, start_ms: 2 }),
    row({ total_in: 0.3, start_ms: 3 }),
    row({ total_in: 0.36, start_ms: 1 }),
  ];
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
    expect(listFinding(one, 2025, null, [events[2]!])).toBe(
      "The one rain event in water year 2025 brought 0.36 in",
    );
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

// ---------- #338: the phone layouts, by chart width ----------

/** 390px less the 16px gutters and the tile's 16px padding. */
const PHONE = 326;
const DESKTOP = 1100;

/** Lay every element out at `width` px: happy-dom lays nothing out. */
function layOut(width: number): void {
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: () => width,
  });
}

afterEach(() => {
  delete (HTMLElement.prototype as { clientWidth?: number }).clientWidth;
  vi.unstubAllGlobals();
});

const dashboard = (scenario: Scenario): DashboardSnapshot =>
  parseDashboard(structuredClone(golden(scenario, "dashboard")));

/** A fresh panel in the page. */
function panel(): HTMLElement {
  document.body.innerHTML = "<section id=panel></section>";
  return document.getElementById("panel")!;
}

/** A section's tile, or on a phone its finding row. */
const tile = (host: HTMLElement, id: string) =>
  host.querySelector<HTMLElement>(`[data-section=${id}]`)!;

/** The days a chart of days names in its label: "Rain each day, Oct 5 to today". */
const firstShown = (chart: Element) =>
  chart
    .querySelector("svg")!
    .getAttribute("aria-label")!
    .match(/, (\w+ \d+) to today/)![1];

/** Press a segment of a tile's 30 / 90 switch. */
function press(host: HTMLElement, days: 30 | 90): void {
  host.querySelector<HTMLButtonElement>(`.days-seg [data-days="${days}"]`)!.click();
}

const pressed = (host: HTMLElement) =>
  host.querySelector(".days-seg [aria-pressed=true]")?.textContent;

describe.each([
  ["daily", ".rain-chart"],
  ["hours", ".rain-chart"],
] as const)("the Rain tab's %s chart", (section, chartSel) => {
  // The raining golden's 90 days run Oct 11, 2025 to Jan 8, 2026.
  it("shows the last 30 days on a phone, and the switch shows 90 and back", () => {
    layOut(PHONE);
    const host = panel();
    renderRain(host, dashboard("raining"));
    const t = tile(host, section);
    const chart = t.querySelector(chartSel)!;

    expect(t.querySelector<HTMLElement>(".days-seg")!.hidden).toBe(false);
    expect(pressed(t)).toBe("30 days");
    expect(firstShown(chart)).toBe("Dec 10");
    expect(t.querySelector("[data-part=standfirst]")!.textContent).toMatch(/of the last 30\b/);

    press(t, 90);
    expect(pressed(t)).toBe("90 days");
    expect(firstShown(chart)).toBe("Oct 11");
    expect(t.querySelector("[data-part=standfirst]")!.textContent).toMatch(/of the last 90\b/);

    press(t, 30);
    expect(firstShown(chart)).toBe("Dec 10");
  });

  it("shows 90 days with no switch on desktop", () => {
    layOut(DESKTOP);
    const host = panel();
    renderRain(host, dashboard("raining"));
    const t = tile(host, section);

    expect(t.querySelector<HTMLElement>(".days-seg")!.hidden).toBe(true);
    expect(firstShown(t.querySelector(chartSel)!)).toBe("Oct 11");
  });
});

describe("the Rain tab's daily bars in 30 and 90 days", () => {
  /** The raining golden with a gap 10 days ago and another 60 days ago. */
  function gapped(): DashboardSnapshot {
    const snapshot = dashboard("raining");
    const recent = snapshot.rain_tab.recent;
    for (const back of [10, 60]) {
      const i = recent.rain_in.length - back;
      recent.rain_in[i] = null;
      recent.reason[i] = "gap";
    }
    return snapshot;
  }
  const hatched = (t: HTMLElement) =>
    [...t.querySelectorAll(".rain-chart svg > rect")].filter((r) =>
      r.getAttribute("fill")?.startsWith("url("),
    ).length;
  const today = (t: HTMLElement) =>
    t.querySelector(".rain-chart .rain-now-label")?.textContent ?? null;

  it("hatches the hour grid's missing days in the days shown", () => {
    layOut(PHONE);
    const host = panel();
    renderRain(host, gapped());
    const t = tile(host, "hours");
    const columns = () =>
      [...t.querySelectorAll(".rain-chart svg > rect:not(.rain-cell)")].filter((r) =>
        r.getAttribute("fill")?.startsWith("url("),
      ).length;

    expect(columns()).toBe(1);
    press(t, 90);
    expect(columns()).toBe(2);
  });

  it("hatches the days missing in the days shown and marks today in both", () => {
    layOut(PHONE);
    const host = panel();
    renderRain(host, gapped());
    const t = tile(host, "daily");

    expect(hatched(t)).toBe(1);
    expect(today(t)).toBe("today");
    press(t, 90);
    expect(hatched(t)).toBe(2);
    expect(today(t)).toBe("today");
  });

  it("keeps today's label off the record-begins label in 90 days", () => {
    layOut(PHONE);
    const host = panel();
    renderRain(host, dashboard("raining"));
    const t = tile(host, "daily");
    press(t, 90);
    const y = (sel: string) => Number(t.querySelector(sel)?.getAttribute("y"));
    const record = [...t.querySelectorAll(".rain-chart text.rain-label")].find((l) =>
      l.textContent?.startsWith("record begins"),
    )!;

    // A pill apart: the record begins a few days before today.
    expect(
      Number(record.getAttribute("y")) - y(".rain-chart text.rain-now-label"),
    ).toBeGreaterThanOrEqual(20);
  });

  it("ticks a month of days by date, today's at the right", () => {
    layOut(PHONE);
    const host = panel();
    renderRain(host, dashboard("raining"));
    const ticks = [...tile(host, "daily").querySelectorAll(".rain-chart text.rain-tick")]
      .map((t) => t.textContent)
      .filter((t) => /^[A-Z]/.test(t!));

    expect(ticks).toEqual(["Jan 8", "Dec 25", "Dec 11"]);
  });
});

describe("the Solar tab's daily bars", () => {
  /** The raining golden with 90 days of sun, ending Jan 8, 2026. */
  function sunny(): DashboardSnapshot {
    const snapshot = dashboard("raining");
    snapshot.solar_tab!.days = {
      first_day: "2025-10-11",
      energy_kwh_m2: Array.from({ length: 90 }, (_, i) => 1 + (i % 5) / 2),
      reason: Array.from({ length: 90 }, () => null),
    };
    return snapshot;
  }

  it("shows the last 30 days on a phone, and the switch shows 90 and back", () => {
    layOut(PHONE);
    history.replaceState(null, "", "/#solar");
    const host = panel();
    renderSolar(host, sunny());
    const t = tile(host, "daily");
    const chart = t.querySelector(".solar-chart")!;

    expect(pressed(t)).toBe("30 days");
    expect(firstShown(chart)).toBe("Dec 10");
    press(t, 90);
    expect(firstShown(chart)).toBe("Oct 11");
    press(t, 30);
    expect(firstShown(chart)).toBe("Dec 10");
  });

  it("has no switch for a record of a month or less", () => {
    layOut(PHONE);
    history.replaceState(null, "", "/#solar");
    const host = panel();
    // The raining golden's solar record is 5 days long.
    renderSolar(host, dashboard("raining"));

    expect(tile(host, "daily").querySelector<HTMLElement>(".days-seg")!.hidden).toBe(true);
  });

  it("shows 90 days with no switch on desktop", () => {
    layOut(DESKTOP);
    history.replaceState(null, "", "/#solar");
    const host = panel();
    renderSolar(host, sunny());
    const t = tile(host, "daily");

    expect(t.querySelector<HTMLElement>(".days-seg")!.hidden).toBe(true);
    expect(firstShown(t.querySelector(".solar-chart")!)).toBe("Oct 11");
  });
});

describe("every hand-drawn chart at phone width", () => {
  /** Each tab's tiles rendered at phone width, both goldens. */
  function rendered(): HTMLElement[] {
    layOut(PHONE);
    const hosts: HTMLElement[] = [];
    for (const scenario of SCENARIOS) {
      // The temperature sections left the Overview for the zoom (#472).
      const zoom = (host: HTMLElement, s: DashboardSnapshot) =>
        renderOverview(host, s, [temperaturePeriods, againstNormal, tenDays]);
      for (const render of [renderOverview, zoom, renderRain, renderSolar]) {
        const host = document.createElement("section");
        document.body.append(host);
        render(host, dashboard(scenario));
        hosts.push(host);
      }
    }
    return hosts;
  }

  it("is drawn at its width, never scaled, so its text keeps its 12px and up", () => {
    const svgs = rendered().flatMap((h) => [...h.querySelectorAll<SVGSVGElement>("svg[viewBox]")]);
    const charts = svgs.filter((svg) => svg.querySelector("text"));

    expect(charts.length).toBeGreaterThan(8);
    for (const svg of charts) {
      const [, , w] = svg.getAttribute("viewBox")!.split(" ").map(Number);
      expect(Number(svg.getAttribute("width")), svg.getAttribute("class")!).toBe(w);
      for (const text of svg.querySelectorAll("text")) {
        // Sizes come only from the stylesheets, where tokens.test holds 12px.
        expect(text.getAttribute("font-size")).toBeNull();
        expect(text.getAttribute("style") ?? "").not.toMatch(/font/);
      }
    }
  });

  it("draws three y ticks on the daily bars, the sun trace and the usual band hero", () => {
    const hosts = rendered();
    // The y ticks end at the axis and read as numbers ("0.4 in", "40°").
    const yTicks = (svg: Element, cls: string) =>
      [...svg.querySelectorAll(`text.${cls}[text-anchor=end]`)].filter((t) =>
        /^-?\d/.test(t.textContent!),
      ).length;
    const charts: [string, string][] = [
      ["[data-section=daily] .rain-chart svg", "rain-tick"],
      ["[data-section=daily] .solar-chart svg", "solar-tick"],
      ["[data-section=now] .solar-chart svg", "solar-tick"],
      ["[data-section=first-screen] .today-svg", "today-tick"],
    ];
    for (const [sel, cls] of charts) {
      // Charts with a y axis.
      const found = hosts
        .flatMap((h) => [...h.querySelectorAll(sel)])
        .filter((svg) => svg.querySelector("line[class$=-grid]"));
      expect(found.length, sel).toBeGreaterThan(0);
      for (const svg of found) expect(yTicks(svg, cls), sel).toBe(3);
    }
  });
});

describe("the period rows at phone width", () => {
  it("put each window's name above its bar, the bar the whole width", () => {
    layOut(PHONE);
    const host = panel();
    renderOverview(host, dashboard("raining"), [temperaturePeriods]);
    const chart = host.querySelector("[data-part=periods-chart] svg")!;
    const names = [...chart.querySelectorAll("text.tc-row")];
    const bars = [...chart.querySelectorAll("rect.tc-iqr, rect.tc-hatch")];

    // Today, yesterday and Full history: the rows the record covers (#470).
    expect(names).toHaveLength(3);
    expect(bars).toHaveLength(3);
    const shift = (el: Element) =>
      Number(el.parentElement!.getAttribute("transform")!.match(/,([\d.]+)\)/)![1]);
    names.forEach((name, i) => {
      const bar = bars[i]!;
      expect(Number(name.getAttribute("x"))).toBe(0);
      // Its baseline above the bar's top.
      expect(Number(name.getAttribute("y")) + shift(name)).toBeLessThan(
        Number(bar.getAttribute("y")) + shift(bar),
      );
    });
  });

  it("keep the label column beside the bars on desktop", () => {
    layOut(DESKTOP);
    const host = panel();
    renderOverview(host, dashboard("raining"), [temperaturePeriods]);
    const chart = host.querySelector("[data-part=periods-chart] svg")!;
    const plot = chart.querySelector("g[transform]")!.getAttribute("transform")!;

    expect(Number(plot.match(/translate\(([\d.]+),/)![1])).toBeGreaterThanOrEqual(118);
  });
});

describe("the usual band hero at phone width", () => {
  const times = (host: HTMLElement) =>
    [...host.querySelectorAll("[data-section=first-screen] .today-x")].map((t) => t.textContent);

  it("names midnight, 6 am, noon and 6 pm under 520px, and the closing midnight from 520px", () => {
    layOut(PHONE);
    const host = panel();
    renderOverview(host, dashboard("seasoned"));
    expect(times(host)).toEqual(["midnight", "6 am", "noon", "6 pm"]);

    layOut(DESKTOP);
    renderOverview(host, dashboard("seasoned"));
    expect(times(host)).toEqual(["midnight", "6 am", "noon", "6 pm", "midnight"]);
  });

  it("keeps its now label inside the chart", () => {
    layOut(PHONE);
    const host = panel();
    renderOverview(host, dashboard("seasoned"));
    const label = host.querySelector("[data-section=first-screen] .today-now")!;
    expect(Number(label.getAttribute("x"))).toBeLessThanOrEqual(PHONE);
    expect(Number(label.getAttribute("y"))).toBeGreaterThan(0);
  });
});

describe("an event chart's redraws", () => {
  /** A fake ResizeObserver and colour-scheme query the test can fire. */
  function fakes(): { resize: () => void; scheme: () => void } {
    const observers: (() => void)[] = [];
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(private readonly callback: () => void) {}
        observe() {
          observers.push(this.callback);
        }
        disconnect() {}
      },
    );
    const view = viewport(390);
    vi.stubGlobal("matchMedia", view.match);
    return {
      resize: () => observers.forEach((fn) => fn()),
      scheme: view.scheme,
    };
  }

  const yTicks = () =>
    react.mock.calls.map((call) => {
      const layout = (call as unknown[])[2] as { yaxis: { nticks?: number } };
      return layout.yaxis.nticks;
    });

  it("redraws with phone ticks once a chart drawn hidden gets its width", async () => {
    const fire = fakes();
    layOut(0);
    const host = document.createElement("div");
    document.body.append(host);
    await drawEventChart(host, curve(20), row(), { chart: "total", medianIn: 0.36 });
    // Drawn hidden: the fallback's desktop width, Plotly's own ticks.
    expect(yTicks()).toEqual([0]);

    layOut(PHONE);
    fire.resize();
    await vi.waitFor(() => expect(react.mock.calls.length).toBe(2));
    expect(yTicks()).toEqual([0, 3]);
  });

  it("draws every word at 12px or more", async () => {
    fakes();
    layOut(PHONE);
    for (const chart of ["total", "rate"] as const) {
      await drawEventChart(document.createElement("div"), curve(20), row(), {
        chart,
        medianIn: 0.36,
      });
    }
    const sizes: number[] = [];
    const walk = (value: unknown, key = ""): void => {
      if (value && typeof value === "object") {
        for (const [k, v] of Object.entries(value)) walk(v, k);
      } else if (key === "size" && typeof value === "number") sizes.push(value);
    };
    for (const call of react.mock.calls) {
      const layout = (call as unknown[])[2] as Record<string, unknown>;
      for (const part of ["font", "xaxis", "yaxis", "annotations", "hoverlabel"])
        walk(layout[part]);
    }

    expect(sizes.length).toBeGreaterThan(4);
    expect(Math.min(...sizes)).toBeGreaterThanOrEqual(12);
  });

  it("redraws every chart in the page when the colour scheme changes", async () => {
    const fire = fakes();
    layOut(PHONE);
    const total = document.createElement("div");
    const rate = document.createElement("div");
    document.body.append(total, rate);
    await drawEventChart(total, curve(20), row(), { chart: "total", medianIn: 0.36 });
    await drawEventChart(rate, curve(20), row(), { chart: "rate", medianIn: 0.36 });
    react.mockClear();

    fire.scheme();

    await vi.waitFor(() => expect(react.mock.calls.length).toBe(2));
  });
});
