// @vitest-environment happy-dom
/**
 * The Rain tab's head, Rain tab C (#664): now, then the season, then the
 * water year against usual with its window strip, from the golden
 * snapshots and worked examples; and a version 4 document without the new
 * fields drawing today's head (ADR-0008).
 */
import { beforeEach, describe, expect, it } from "vitest";

import { LABEL_CHAR_PX } from "../src/charts/svg.js";
import { drawYearCurves } from "../src/rain/charts.js";
import { bandLabels } from "../src/rain/head.js";
import { agoWords } from "../src/now-story.js";
import { renderRain } from "../src/rain/index.js";
import {
  nowLine,
  seasonLead,
  stripUsualWords,
  verdictHeadline,
  yearsHeadline,
} from "../src/rain/text.js";
import type {
  DashboardSnapshot,
  RainNow,
  RainTab,
  StripWindow,
  WaterYearCurve,
} from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const raining = parseDashboard(golden("raining", "dashboard"));
const seasoned = parseDashboard(golden("seasoned", "dashboard"));

let panel: HTMLElement;

beforeEach(() => {
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const head = () => panel.querySelector("[data-section=years]")!;
const all = (selector: string) => [...head().querySelectorAll(selector)].map((el) => text(el));

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;

/** A copy of the seasoned snapshot with its now story replaced. */
function withNow(rain: Partial<RainNow> | null): DashboardSnapshot {
  const copy = structuredClone(seasoned);
  if (rain === null) delete copy.rain_now;
  else copy.rain_now = { ...copy.rain_now!, ...rain };
  return copy;
}

/** A copy of a snapshot written before R1 of #647: no now, season or strip. */
function beforeHead(snapshot: DashboardSnapshot): DashboardSnapshot {
  const copy = structuredClone(snapshot);
  delete copy.rain_now;
  delete copy.wet_hours;
  delete copy.rain_tab.wet_hours_to_date;
  delete copy.rain_tab.window_strip;
  return copy;
}

describe("the head's now part", () => {
  const built = seasoned.built_at_ms;

  it("reads last rain alone when rain is not leading", () => {
    expect(nowLine(withNow({ last_rain_ms: built - 9 * HOUR - 20 * MINUTE }))).toBe(
      "Last rain 9 h ago",
    );
    expect(nowLine(seasoned)).toBe("Last rain 3 d ago");
  });

  it("tells the now story while it rains: the rate, then the event since its start", () => {
    // The raining golden: the event started at 2 pm the same day.
    expect(nowLine(raining)).toBe("Raining · 0.12 in/hr · 0.30 in since 2:00 pm");
  });

  it("says rain stopped while the story is still told, and the rate alone with no event", () => {
    const stopped = withNow({
      last_rain_ms: built - 40 * MINUTE,
      raining: false,
      rate_in_hr: 0.02,
      event_start_ms: null,
      event_total_in: null,
    });
    expect(nowLine(stopped)).toBe("Rain stopped 40 min ago");
    expect(nowLine(withNow({ raining: true, rate_in_hr: 0.08 }))).toBe("Raining · 0.08 in/hr");
  });

  it("draws no now part for a snapshot without rain_now (N−1)", () => {
    expect(nowLine(withNow(null))).toBeNull();
    expect(nowLine(withNow({ last_rain_ms: null }))).toBe("No rain in the record yet");
  });

  it("words how long ago in minutes, hours, then days", () => {
    expect([40 * MINUTE, 9 * HOUR, 47 * HOUR, 73 * HOUR].map(agoWords)).toEqual([
      "40 min",
      "9 h",
      "47 h",
      "3 d",
    ]);
  });
});

describe("the head on the golden snapshots", () => {
  it("runs now, then the season, then usual", () => {
    renderRain(panel, seasoned);

    expect(text(head().querySelector("[data-part=kicker]"))).toBe("Now");
    expect(text(head().querySelector("[data-part=finding]"))).toBe("Last rain 3 d ago");
    expect(all("[data-part=label]")).toEqual([
      "Wet hours",
      "The water year against usual",
      "Each window against its usual",
    ]);
    expect(text(head().querySelector("[data-part=season]"))).toBe(
      "2 wet hours in the last 7 days, 6 in the last 30",
    );
    expect(text(head().querySelector("[data-part=verdict]"))).toBe(
      verdictHeadline(seasoned.rain_tab),
    );
    // In reading order: the season's bars, then the curves, then the strip.
    const order = ["[data-part=wet-bars]", ".rain-chart", "[data-part=windows]"].map((s) =>
      head().querySelector(s)!,
    );
    expect(order[0]!.compareDocumentPosition(order[1]!) & Node.DOCUMENT_POSITION_FOLLOWING).toBe(4);
    expect(order[1]!.compareDocumentPosition(order[2]!) & Node.DOCUMENT_POSITION_FOLLOWING).toBe(4);
  });

  it("marks the now line while it rains", () => {
    renderRain(panel, raining);
    const finding = head().querySelector("[data-part=finding]")!;

    expect(text(finding)).toBe(nowLine(raining));
    expect(finding.querySelector(".rain-now-mark")).not.toBeNull();
    renderRain(panel, seasoned);
    expect(head().querySelector(".rain-now-mark")).toBeNull();
  });

  it("draws today's head for a version 4 document without the new fields", () => {
    const old = beforeHead(seasoned);
    renderRain(panel, old);

    expect(text(head().querySelector("[data-part=kicker]"))).toBe("The water year");
    expect(text(head().querySelector("[data-part=finding]"))).toBe(yearsHeadline(old.rain_tab));
    expect(head().querySelector("[data-part=label], [data-part=wet-bars]")).toBeNull();
    expect(head().querySelector(".rain-strip")).toBeNull();
    expect(head().querySelector(".rain-windows")).not.toBeNull();
  });
});

describe("the season part", () => {
  it("counts one wet hour, and the water year's alone without the meter's windows", () => {
    const one = structuredClone(seasoned);
    one.wet_hours!.last_7_days.wet_hours = 1;
    expect(seasonLead(one)).toBe("1 wet hour in the last 7 days, 6 in the last 30");
    delete one.wet_hours;
    expect(seasonLead(one)).toBe("6 wet hours since Oct 1");
    delete one.rain_tab.wet_hours_to_date;
    expect(seasonLead(one)).toBeNull();
  });

  it("draws a bar for 7 days, 30 days and since Oct 1, wet hours of countable hours", () => {
    renderRain(panel, raining);
    const rows = [...head().querySelectorAll(".rain-wet-row")];

    expect(rows.map((r) => r.getAttribute("data-key"))).toEqual(["7d", "30d", "water_year"]);
    expect(rows.map((r) => text(r.querySelector("[data-part=count]")))).toEqual([
      "12 h of 111",
      "12 h of 111",
      "12 h of 111",
    ]);
    // Missing hours are hatched at the bar's end and said, never dry.
    expect(rows[0]!.querySelector(".rain-wet-missing.gap-hatch")).not.toBeNull();
    expect(text(rows[0]!.querySelector("[data-part=missing]"))).toBe(
      "57 h missing, not dry · 3 d missing",
    );
  });

  it("draws usual wet hours to date as a band on the since-Oct-1 bar", () => {
    const three = structuredClone(seasoned);
    const wet = three.rain_tab.wet_hours_to_date!;
    const earlier = (water_year: number, usual_hours: number) => ({
      water_year,
      wet_hours: usual_hours,
      countable_hours: 5000,
      usual_hours,
    });
    wet.usual = {
      years: 3,
      earlier: [earlier(2023, 100), earlier(2024, 140), earlier(2025, 200)],
      low_hours: 120,
      high_hours: 170,
      band: "range",
    };
    renderRain(panel, three);
    const band = head().querySelector<HTMLElement>("[data-part=wet-to-date] [data-part=usual]")!;

    expect(band.style.getPropertyValue("--from")).toBe(`${(100 * 120) / wet.hours}%`);
    expect(band.style.getPropertyValue("--to")).toBe(`${(100 * 170) / wet.hours}%`);
    expect(text(head().querySelector("[data-part=usual-words]"))).toBe(
      "usual to date 120–170 h, range of 3 earlier years · 1 h missing, not dry",
    );
  });
});

describe("usual wet hours below 3 earlier years", () => {
  it("ticks each earlier year's plain value, never a band the snapshot did not send", () => {
    const two = structuredClone(seasoned);
    const wet = two.rain_tab.wet_hours_to_date!;
    const earlier = (water_year: number, usual_hours: number) => ({
      water_year,
      wet_hours: usual_hours,
      countable_hours: 5000,
      usual_hours,
    });
    wet.usual = {
      years: 2,
      earlier: [earlier(2024, 150), earlier(2025, 170)],
      low_hours: null,
      high_hours: null,
      band: null,
    };
    renderRain(panel, two);
    const marks = [
      ...head().querySelectorAll<HTMLElement>("[data-part=wet-to-date] [data-part=usual]"),
    ].map((m) => [m.style.getPropertyValue("--from"), m.style.getPropertyValue("--to")]);

    expect(marks).toEqual([
      [`${(100 * 150) / wet.hours}%`, `${(100 * 150) / wet.hours}%`],
      [`${(100 * 170) / wet.hours}%`, `${(100 * 170) / wet.hours}%`],
    ]);
  });
});

describe("the window strip", () => {
  const rows = () => [...head().querySelectorAll(".rain-strip-row")];

  it("draws each window the record covers, in the snapshot's order, with its mark in words", () => {
    renderRain(panel, seasoned);

    expect(rows().map((r) => r.getAttribute("data-key"))).toEqual([
      "today",
      "yesterday",
      "7d",
      "30d",
      "90d",
      "365d",
      "water_year",
    ]);
    const week = rows()[2]!;
    expect(text(week.querySelector(".rain-strip-name"))).toBe("7 days");
    expect(text(week.querySelector(".rain-strip-value b"))).toBe("0.24 in");
    expect(text(week.querySelector(".rain-strip-mark"))).toBe("above usual range");
    // Today sits in its usual range: no mark.
    expect(rows()[0]!.querySelector(".rain-strip-mark")).toBeNull();
    // The band's end values are labelled under the track: the water year's single value.
    expect([...rows()[6]!.querySelectorAll("svg text")].map((t) => t.textContent)).toEqual([
      "0.30 in",
    ]);
  });

  it("draws no key when every window shown is hatched", () => {
    // Dry: today and yesterday are covered, neither archived yet.
    renderRain(panel, parseDashboard(golden("dry", "dashboard")));

    expect(rows().length).toBe(2);
    expect(head().querySelector(".rain-strip-track, .rain-strip-key")).toBeNull();
    renderRain(panel, seasoned);
    expect(head().querySelector(".rain-strip-key")).not.toBeNull();
  });

  it("hatches a window it covers but cannot compare, with no track", () => {
    renderRain(panel, seasoned);
    const year = rows()[5]!;

    expect(year.querySelector(".rain-strip-value.gap-hatch")).not.toBeNull();
    expect(year.querySelector("svg")).toBeNull();
  });

  it("draws a window with no earlier year as its dot, with no range, and says so", () => {
    // Raining: only today and yesterday are covered, and no earlier year has them.
    renderRain(panel, raining);

    expect(rows().map((r) => r.getAttribute("data-key"))).toEqual(["today", "yesterday"]);
    for (const row of rows()) {
      expect(row.querySelector("circle.rain-strip-dot")).not.toBeNull();
      expect(row.querySelector(".rain-strip-band, .rain-strip-tick")).toBeNull();
      expect([...row.querySelectorAll("svg text")].map((t) => t.textContent)).toEqual([
        "no earlier year",
      ]);
    }
  });

  it("prints the snapshot's mark, never its own", () => {
    // #690 changes which windows Python marks; the site draws what it is sent.
    const sent = structuredClone(seasoned);
    sent.rain_tab.window_strip!.find((w) => w.key === "7d")!.mark = null;
    sent.rain_tab.window_strip!.find((w) => w.key === "yesterday")!.mark = "below usual range";
    renderRain(panel, sent);

    expect(rows()[2]!.querySelector(".rain-strip-mark")).toBeNull();
    expect(text(rows()[1]!.querySelector(".rain-strip-mark"))).toBe("below usual range");
  });

  it("names the usual each band rests on", () => {
    const window = (change: Partial<StripWindow>): StripWindow => ({
      key: "30d",
      rain_in: 3.97,
      years: 0,
      earlier: [],
      low_in: null,
      high_in: null,
      band: null,
      mark: null,
      ...change,
    });
    const one = { last_day: "2024-11-22", rain_in: 3.1 };

    expect(stripUsualWords(window({}))).toBe("no earlier year");
    expect(stripUsualWords(window({ years: 1, earlier: [one], low_in: 3.1, high_in: 3.1 }))).toBe(
      "usual 3.10 in, from 2024",
    );
    expect(
      stripUsualWords(
        window({ key: "water_year", years: 1, earlier: [one], low_in: 3.1, high_in: 3.1 }),
      ),
    ).toBe("usual 3.10 in, from 2025");
    expect(
      stripUsualWords(window({ years: 2, earlier: [one, one], low_in: 2, high_in: 3.1 })),
    ).toBe("usual 2.00–3.10 in, 2 earlier years");
    expect(stripUsualWords(window({ years: 3, low_in: 2, high_in: 3.1, band: "range" }))).toBe(
      "usual 2.00–3.10 in, range of 3 earlier years",
    );
    expect(stripUsualWords(window({ years: 4, low_in: 2, high_in: 3.1, band: "percentile" }))).toBe(
      "usual 2.00–3.10 in, 25th–75th percentile of 4 earlier years",
    );
  });

  it("labels a band's ends either side of it, and as one label at the track's edge", () => {
    const band = (low: number, high: number): StripWindow => ({
      key: "30d",
      rain_in: 1,
      years: 4,
      earlier: [],
      low_in: low,
      high_in: high,
      band: "percentile",
      mark: null,
    });
    const x = (width: number) => (v: number) => 7 + ((width - 14) * v) / 10;

    const apart = bandLabels(band(3, 7), x(326), 326);
    expect(apart.map((l) => [l.text, l.anchor])).toEqual([
      ["3.00", "end"],
      ["7.00 in", "start"],
    ]);
    // Outside the band, a narrow band's labels never meet.
    const near = bandLabels(band(4.8, 5.2), x(326), 326);
    expect(near.map((l) => l.text)).toEqual(["4.80", "5.20 in"]);
    // At either edge, one label, inside the track.
    for (const labels of [
      bandLabels(band(0, 0.3), x(256), 256),
      bandLabels(band(9.6, 10), x(256), 256),
    ]) {
      expect(labels).toHaveLength(1);
      const [label] = labels;
      const half = (label!.text.length * LABEL_CHAR_PX) / 2;
      expect(label!.x - half).toBeGreaterThanOrEqual(0);
      expect(label!.x + half).toBeLessThanOrEqual(256);
    }
  });
});

describe("the water-year curves at 390 px", () => {
  /** A water year rising to `top` by `place`. */
  function curve(water_year: number, top: number, place: number): WaterYearCurve {
    const cumulative_in = Array.from({ length: place + 1 }, (_, i) => (top * i) / place);
    return {
      water_year,
      first_day: `${water_year - 1}-10-01`,
      cumulative_in,
      prorated: cumulative_in.map(() => true),
      status: seasoned.rain_tab.water_years[0]!.status,
      total_in: top,
      to_date_in: top,
    };
  }

  it.each([40, 200, 363])("keeps every label inside a 326 px tile, this year at day %i", (day) => {
    const tab: RainTab = {
      ...seasoned.rain_tab,
      day_index: day,
      water_years: [curve(2024, 48.62, 364), curve(2025, 51.3, 364), curve(2026, 47.88, day)],
    };
    const host = document.createElement("figure");
    drawYearCurves(host, tab, 326, () => {});
    const labels = [...host.querySelectorAll("text.rain-label, text.rain-now-label")];

    expect(labels.length).toBeGreaterThan(0);
    for (const label of labels) {
      const x = Number(label.getAttribute("x"));
      const w = label.textContent!.length * LABEL_CHAR_PX;
      const anchor = label.getAttribute("text-anchor") ?? "start";
      const right = anchor === "end" ? x : anchor === "middle" ? x + w / 2 : x + w;
      expect(right, label.textContent!).toBeLessThanOrEqual(326);
    }
  });
});
