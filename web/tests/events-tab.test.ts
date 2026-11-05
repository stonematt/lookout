// @vitest-environment happy-dom
/**
 * The Rain Events tab rendered from the golden snapshots, with the curve
 * source and the chart drawing faked (no network, no Plotly): the list as
 * one exclusive accordion, each row opening its own curve (#339).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EventsDeps } from "../src/events/tab.js";
import { mountEvents } from "../src/events/tab.js";
import { ReadError } from "../src/load.js";
import type { DashboardSnapshot, EventCurve, GapStatus } from "../src/snapshot.js";
import { parseDashboard, parseEventCurve } from "../src/snapshot.js";
import { goldenCurve, golden } from "./golden.js";

// The raining golden: event 3 live, events 2 and 1 completed (Jan 2026).
const EVENT_1 = Date.UTC(2026, 0, 4, 18);
const EVENT_2 = Date.UTC(2026, 0, 7, 4);
const EVENT_3 = Date.UTC(2026, 0, 8, 22);
const NOW = Date.UTC(2026, 0, 9, 0, 30);

let panel: HTMLElement;
let deps: EventsDeps & {
  fetchCurve: ReturnType<typeof vi.fn>;
  draw: ReturnType<typeof vi.fn>;
};

function raining(): DashboardSnapshot {
  return parseDashboard(structuredClone(golden("raining", "dashboard")));
}

function curveOf(start: number): EventCurve {
  return parseEventCurve(goldenCurve("raining", start));
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));
const rows = () => [...panel.querySelectorAll<HTMLElement>(".ev-list .ev-row")];
const starts = () => rows().map((r) => Number(r.dataset.start));
const openRows = () => [...panel.querySelectorAll<HTMLElement>(".ev-row[data-open]")];
const openStart = () => Number(openRows()[0]?.dataset.start ?? NaN);
const finding = () => panel.querySelector(".ev-row[data-open] .ev-finding")?.textContent;
const charts = () => panel.querySelector(".ev-row[data-open] .ev-charts")!;
const click = (selector: string) =>
  panel
    .querySelector<HTMLElement>(selector)!
    .dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
const row = (start: number) => `.ev-open[data-start='${start}']`;
/** Pick a water year in the select, as a visitor does. */
function pick(value: string): void {
  const select = panel.querySelector<HTMLSelectElement>(".ev-select")!;
  select.value = value;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}
const drawnLast = () => deps.draw.mock.calls.at(-1)!;

beforeEach(() => {
  history.replaceState(null, "", "/?fixture=raining#events");
  document.body.innerHTML = "<section id=panel></section>";
  panel = document.getElementById("panel")!;
  deps = {
    fetchCurve: vi.fn(async (start: number) => curveOf(start)),
    draw: vi.fn(async () => {}),
    nowMs: () => NOW,
  };
});

describe("the event list", () => {
  it("lists every catalog event newest first, the live one marked", () => {
    mountEvents(panel, raining(), deps);

    expect(starts()).toEqual([EVENT_3, EVENT_2, EVENT_1]);
    expect(rows()[0]!.hasAttribute("data-live")).toBe(true);
    expect(rows()[0]!.querySelector(".ev-live")!.textContent).toBe("Live");
    expect(rows()[1]!.querySelector(".ev-live")).toBeNull();
    expect(rows()[1]!.querySelector(".ev-total")!.textContent).toBe("0.48 in");
    expect(rows()[1]!.querySelector(".ev-meta")!.textContent).toBe(
      "15 h 50 m · peak 0.12 in/hr excellent",
    );
  });

  it("heads the list with the summary the snapshot computed", () => {
    mountEvents(panel, raining(), deps);

    expect(panel.querySelector(".ev-title")!.textContent).toBe(
      "The wettest of 3 events since Jan 4, 2026, brought 0.48 in, against 0.36 in for the median event",
    );
    const stats = panel.querySelector(".ev-stats")!.textContent!;
    expect(stats).toContain("3");
    expect(stats).toContain("0.36 in"); // the median event
    expect(panel.querySelector(".ev-list-finding")!.textContent).toBe("3 events, newest first");
  });

  it("sorts by total with the live event still pinned on top", () => {
    mountEvents(panel, raining(), deps);

    click("[data-sort=wettest]");

    expect(starts()).toEqual([EVENT_3, EVENT_2, EVENT_1]);
    expect(panel.querySelector("[data-sort=wettest]")!.getAttribute("aria-pressed")).toBe("true");
  });

  it("orders Newest by start, Wettest by total and Longest by duration", () => {
    const snapshot = raining();
    snapshot.rain_events.events[0]!.live = false;
    snapshot.rain_events.events[2]!.duration_minutes = 2000;
    mountEvents(panel, snapshot, deps);
    const order = (sort: string) => {
      click(`[data-sort=${sort}]`);
      return starts();
    };

    expect(order("longest")).toEqual([EVENT_1, EVENT_2, EVENT_3]);
    expect(order("wettest")).toEqual([EVENT_2, EVENT_1, EVENT_3]);
    expect(order("newest")).toEqual([EVENT_3, EVENT_2, EVENT_1]);
    expect(
      [...panel.querySelectorAll<HTMLElement>(".ev-sort button")].map((b) => b.textContent),
    ).toEqual(["Newest", "Wettest", "Longest"]);
  });

  it("picks one water year's events and its summary in the select", () => {
    const snapshot = raining();
    const section = snapshot.rain_events;
    section.events[2]!.water_year = 2025;
    section.seasons = [
      {
        ...section.seasons[0]!,
        stats: { count: 2, total_in: 0.78, median_in: 0.39, mean_duration_minutes: 547.5 },
      },
      {
        ...section.seasons[0]!,
        water_year: 2025,
        stats: { count: 1, total_in: 0.36, median_in: 0.36, mean_duration_minutes: 530 },
      },
    ];
    mountEvents(panel, snapshot, deps);
    const options = [...panel.querySelectorAll<HTMLOptionElement>(".ev-select option")];
    expect(options.map((o) => o.value)).toEqual(["all", "2026", "2025"]);
    expect(options[0]!.textContent).toBe("All years");

    pick("2025");

    expect(starts()).toEqual([EVENT_1]);
    expect(panel.querySelector(".ev-title")!.textContent).toBe(
      "The one rain event in water year 2025 brought 0.36 in",
    );
  });

  it("draws each size bar as its total over the catalog's largest, hidden from assistive tech", () => {
    mountEvents(panel, raining(), deps);

    const bars = rows().map((r) => r.querySelector<HTMLElement>(".ev-size")!);
    for (const bar of bars) expect(bar.getAttribute("aria-hidden")).toBe("true");
    const widths = bars.map((b) => (b.firstElementChild as HTMLElement).style.width);
    // 0.30, 0.48 and 0.36 against the wettest, 0.48.
    expect(widths).toEqual(["62.5%", "100.0%", "75.0%"]);
    expect(panel.querySelector(".ev-key")!.textContent).toContain("the wettest on record, 0.48 in");
  });

  it("ends a long list with Show N more", () => {
    const snapshot = raining();
    const oldest = snapshot.rain_events.events.at(-1)!;
    for (let i = 1; i <= 22; i += 1) {
      const start = EVENT_1 - i * 86_400_000;
      snapshot.rain_events.events.push({ ...oldest, start_ms: start, end_ms: start + 3_600_000 });
    }
    mountEvents(panel, snapshot, deps);

    expect(rows()).toHaveLength(20);
    expect(panel.querySelector(".ev-more")!.textContent).toBe("Show 5 more");
    click(".ev-more");
    expect(rows()).toHaveLength(25);
    expect(panel.querySelector(".ev-more")).toBeNull();
  });
});

describe("the open row", () => {
  it("opens the live event with its curve when the URL names none, without writing it", async () => {
    const snapshot = raining();
    mountEvents(panel, snapshot, deps);
    await flush();

    expect(openRows()).toHaveLength(1);
    expect(openStart()).toBe(EVENT_3);
    expect(panel.querySelector(row(EVENT_3))!.getAttribute("aria-expanded")).toBe("true");
    expect(finding()).toBe("0.30 in so far over 2 h 25 m, and still raining");
    expect(deps.fetchCurve).not.toHaveBeenCalled();
    expect(deps.draw).toHaveBeenCalledOnce();
    expect(drawnLast()[1]).toBe(snapshot.rain_events.open_curve);
    expect(location.search).toBe("?fixture=raining");
  });

  it("opens the newest event, with the latest chip, when none is live", async () => {
    const snapshot = raining();
    snapshot.rain_events.events[0]!.live = false;
    mountEvents(panel, snapshot, deps);
    await flush();

    expect(openStart()).toBe(EVENT_3);
    expect(rows()[0]!.querySelector(".ev-latest")!.textContent).toBe("latest");
    expect(panel.querySelectorAll(".ev-latest")).toHaveLength(1);
    expect(panel.querySelector(".ev-live")).toBeNull();
  });

  it("charts an event still open but no longer live from the open curve, unmarked", async () => {
    // The catalog holds event 3 open; the live counter has ended it.
    const snapshot = raining();
    snapshot.rain_events.events[0]!.live = false;
    mountEvents(panel, snapshot, deps);
    await flush();

    expect(deps.fetchCurve).not.toHaveBeenCalled();
    expect(drawnLast()[1]).toBe(snapshot.rain_events.open_curve);
    expect(rows()[0]!.hasAttribute("data-live")).toBe(false);
  });

  it("draws Total against the catalog's median event", async () => {
    mountEvents(panel, raining(), deps);
    await flush();

    expect(drawnLast()[3]).toEqual({ chart: "total", medianIn: 0.36 });
    expect(drawnLast()[0]).toBe(charts().querySelector(".ev-chart"));
  });

  it("answers a click once after a newer snapshot is mounted over it", async () => {
    // The page mounts the tab again whenever the snapshot changes.
    mountEvents(panel, raining(), deps);
    mountEvents(panel, raining(), deps);

    click(row(EVENT_2));
    await flush();

    expect(deps.fetchCurve).toHaveBeenCalledTimes(1);
  });

  it("opening another row closes the open one, reads its curve and writes the URL", async () => {
    mountEvents(panel, raining(), deps);

    click(row(EVENT_2));
    await flush();

    expect(openRows().map((r) => Number(r.dataset.start))).toEqual([EVENT_2]);
    expect(panel.querySelector(row(EVENT_3))!.getAttribute("aria-expanded")).toBe("false");
    expect(location.search).toBe(`?fixture=raining&event=${EVENT_2}`);
    expect(location.hash).toBe("#events");
    expect(deps.fetchCurve).toHaveBeenCalledWith(EVENT_2);
    expect(drawnLast()[1]).toEqual(curveOf(EVENT_2));
    expect(finding()).toBe("0.48 in over 15 h 50 m, the wettest of 3 events");
    // The curve sits under the event it belongs to.
    expect(drawnLast()[0].closest(".ev-row").dataset.start).toBe(String(EVENT_2));
  });

  it("closes the open row when its own header is clicked, and clears the URL", async () => {
    mountEvents(panel, raining(), deps);
    click(row(EVENT_2));
    await flush();

    click(row(EVENT_2));

    expect(openRows()).toHaveLength(0);
    expect(location.search).toBe("?fixture=raining");
  });

  it("opens the event a shared link names", async () => {
    history.replaceState(null, "", `/?event=${EVENT_1}#events`);

    mountEvents(panel, raining(), deps);
    await flush();

    expect(openStart()).toBe(EVENT_1);
    expect(deps.fetchCurve).toHaveBeenCalledWith(EVENT_1);
    expect(finding()).toBe("0.36 in over 8 h 50 m, the 2nd wettest of 3 events");
  });

  it("names each row's header for its event and the body it controls", async () => {
    mountEvents(panel, raining(), deps);

    const head = panel.querySelector<HTMLButtonElement>(row(EVENT_3))!;
    expect(head.tagName).toBe("BUTTON");
    expect(head.closest("h3")).not.toBeNull();
    const body = document.getElementById(head.getAttribute("aria-controls")!)!;
    expect(body.closest(".ev-row")).toBe(head.closest(".ev-row"));
  });

  it("says so when a linked event is not in the catalog", async () => {
    history.replaceState(null, "", "/?event=1600000000000#events");

    mountEvents(panel, raining(), deps);
    await flush();

    expect(openRows()[0]!.textContent).toContain("The linked event is not in the catalog");
    expect(finding()).toContain("still raining");
  });

  it("shows totals only, with the gap note and no switch, for an event with no archive data", async () => {
    const snapshot = raining();
    const event = snapshot.rain_events.events[2]!;
    Object.assign(event, { has_curve: false, quality: "invalid" });
    history.replaceState(null, "", `/?event=${EVENT_1}#events`);

    mountEvents(panel, snapshot, deps);
    await flush();

    expect(deps.fetchCurve).not.toHaveBeenCalled();
    expect(deps.draw).not.toHaveBeenCalled();
    const open = openRows()[0]!;
    expect(open.querySelector(".ev-gap.gap-hatch")!.textContent).toContain("n/a: gap in record");
    expect(open.querySelector(".ev-facts")!.textContent).toContain("0.36 in");
    expect(open.querySelector("[data-chart]")).toBeNull();
    expect(open.textContent).not.toMatch(/error/i);
    expect(rows()[2]!.querySelector(".ev-chip.low")!.textContent).toBe("no data");
  });

  it("says a curve not yet written is coming, not an error", async () => {
    deps.fetchCurve.mockRejectedValue(new ReadError("/event?start=1", 404));
    history.replaceState(null, "", `/?event=${EVENT_2}#events`);

    mountEvents(panel, raining(), deps);
    await flush();

    expect(charts().textContent).toContain("not published yet");
  });

  it("shows the loading state, keeping the rest of the row, until the curve arrives", async () => {
    let arrive: (curve: EventCurve) => void = () => {};
    deps.fetchCurve.mockReturnValueOnce(new Promise((resolve) => (arrive = resolve)));
    mountEvents(panel, raining(), deps);

    click(row(EVENT_2));

    expect(charts().textContent).toBe("Loading the curve…");
    expect(finding()).toBe("0.48 in over 15 h 50 m, the wettest of 3 events");
    arrive(curveOf(EVENT_2));
    await flush();
    expect(charts().textContent).not.toContain("Loading");
    expect(charts().querySelector(".ev-chart")).not.toBeNull();
  });

  it("shows the failed read with a Try again pill that reads again", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    deps.fetchCurve.mockRejectedValueOnce(new ReadError("/event?start=1", 502));
    history.replaceState(null, "", `/?event=${EVENT_2}#events`);
    mountEvents(panel, raining(), deps);
    await flush();

    expect(charts().textContent).toContain("Couldn't load this event's curve.");
    expect(panel.querySelector(".ev-retry")!.textContent).toBe("Try again");
    expect(finding()).toBe("0.48 in over 15 h 50 m, the wettest of 3 events");
    click(".ev-retry");
    await flush();

    expect(deps.fetchCurve).toHaveBeenCalledTimes(2);
    expect(deps.draw).toHaveBeenCalledOnce();
  });

  it("switches the open row's chart between Total and Rate in place", async () => {
    mountEvents(panel, raining(), deps);
    click(row(EVENT_2));
    await flush();
    const head = panel.querySelector(row(EVENT_2));

    click("[data-chart=rate]");
    await flush();

    expect(drawnLast()[3]).toEqual({ chart: "rate", medianIn: 0.36 });
    expect(drawnLast()[1]).toEqual(curveOf(EVENT_2));
    expect(deps.fetchCurve).toHaveBeenCalledTimes(1);
    expect(panel.querySelector(row(EVENT_2))).toBe(head);
    expect(panel.querySelector("[data-chart=rate]")!.getAttribute("aria-pressed")).toBe("true");
    expect(charts().querySelector<HTMLElement>(".ev-chart")!.dataset.chart).toBe("rate");

    click("[data-chart=total]");
    expect(drawnLast()[3].chart).toBe("total");
  });

  it("keeps the rest of the open row as it is when the chart switches", async () => {
    history.replaceState(null, "", "/?event=1600000000000#events");
    mountEvents(panel, raining(), deps);
    await flush();
    const open = openRows()[0]!;
    const words = open.querySelector(".ev-finding");
    const fold = open.querySelector<HTMLDetailsElement>("details[data-part=source-note]")!;
    fold.open = true;
    const rate = open.querySelector<HTMLButtonElement>("[data-chart=rate]")!;

    click("[data-chart=rate]");

    expect(open.querySelector(".ev-finding")).toBe(words);
    expect(open.querySelector("[data-chart=rate]")).toBe(rate);
    expect(fold.isConnected && fold.open).toBe(true);
    expect(open.textContent).toContain("The linked event is not in the catalog");
  });

  it("draws the curve against the median event of the water year shown", async () => {
    const snapshot = raining();
    snapshot.rain_events.seasons[0]!.stats = {
      ...snapshot.rain_events.seasons[0]!.stats,
      median_in: 0.42,
    };
    mountEvents(panel, snapshot, deps);
    await flush();
    expect(drawnLast()[3].medianIn).toBe(0.36);

    pick(String(snapshot.rain_events.seasons[0]!.water_year));
    await flush();

    expect(drawnLast()[3].medianIn).toBe(0.42);
    expect(panel.querySelector(".ev-stats")!.textContent).toContain("0.42 in");
  });

  it("follows the default rule within a water year the open row is not in", async () => {
    const snapshot = raining();
    snapshot.rain_events.events[2]!.water_year = 2025;
    snapshot.rain_events.seasons.push({ ...snapshot.rain_events.seasons[0]!, water_year: 2025 });
    history.replaceState(null, "", `/?event=${EVENT_2}#events`);
    mountEvents(panel, snapshot, deps);

    pick("2025");

    expect(openStart()).toBe(EVENT_1);
    expect(location.search).toBe("");
    pick("2026");
    expect(openStart()).toBe(EVENT_3); // the live one
  });

  it("keeps a row closed by hand closed when the water year changes", () => {
    mountEvents(panel, raining(), deps);
    click(row(EVENT_3));

    pick(String(raining().rain_events.seasons[0]!.water_year));

    expect(openRows()).toHaveLength(0);
  });

  it("says a water year with no events has none, and keeps the filter bar", () => {
    const snapshot = raining();
    snapshot.rain_events.seasons.unshift({
      ...snapshot.rain_events.seasons[0]!,
      water_year: 2027,
      stats: { count: 0, total_in: 0, median_in: null, mean_duration_minutes: null },
    });
    mountEvents(panel, snapshot, deps);

    pick("2027");

    expect(rows()).toHaveLength(0);
    expect(panel.querySelector(".ev-empty")!.textContent).toBe(
      "No rain events in water year 2027 yet.",
    );
    expect(panel.querySelector(".ev-bar .ev-select")).not.toBeNull();
    expect(panel.querySelector(".ev-bar .ev-sort")).not.toBeNull();
  });
});

describe("a newer snapshot mounted over the tab", () => {
  const HOUR = 3_600_000;

  /**
   * The raining snapshot with `extra` more completed events, one a day before
   * event 1, so the list runs past one page. Built at `builtAtMs`.
   */
  function longer(extra: number, builtAtMs: number): DashboardSnapshot {
    const snapshot = raining();
    snapshot.built_at_ms = builtAtMs;
    const oldest = snapshot.rain_events.events.at(-1)!;
    for (let i = 1; i <= extra; i += 1) {
      const start = EVENT_1 - i * 24 * HOUR;
      snapshot.rain_events.events.push({ ...oldest, start_ms: start, end_ms: start + HOUR });
    }
    return snapshot;
  }

  it("keeps the open event and the rows Show more opened", async () => {
    const first = longer(30, 1);
    mountEvents(panel, first, deps);
    click(".ev-more");
    click(row(EVENT_2));
    await flush();
    expect(rows()).toHaveLength(33);
    const opened = finding();

    mountEvents(panel, longer(30, 2), deps);
    await flush();

    expect(rows()).toHaveLength(33);
    expect(openStart()).toBe(EVENT_2);
    expect(finding()).toBe(opened);
    expect(openRows()[0]!.textContent).not.toContain("not in the catalog");
  });

  it("shows the open event's row when newer events push it past the page", async () => {
    const first = longer(30, 1);
    mountEvents(panel, first, deps);
    const last = starts().at(-1)!; // row 20, the page's last
    click(row(last));
    await flush();

    const next = longer(30, 2);
    const newer = { ...next.rain_events.events[1]!, start_ms: EVENT_2 + HOUR };
    next.rain_events.events.splice(1, 0, newer);
    mountEvents(panel, next, deps);

    expect(starts()).toContain(last);
    expect(openStart()).toBe(last);
  });

  it("redraws a kept completed event from the curve it already read", async () => {
    mountEvents(panel, raining(), deps);
    click(row(EVENT_2));
    await flush();

    mountEvents(panel, raining(), deps);

    expect(charts().textContent).not.toContain("Loading");
    await flush();
    expect(deps.fetchCurve).toHaveBeenCalledTimes(1);
    expect(deps.draw).toHaveBeenCalledTimes(3); // the live event, event 2, event 2 again
    expect(drawnLast()[1]).toEqual(curveOf(EVENT_2));
  });

  it("keeps the water year, the order and the chart picked", async () => {
    const year = raining().rain_events.seasons[0]!.water_year;
    mountEvents(panel, raining(), deps);
    click("[data-sort=longest]");
    pick(String(year));
    click("[data-chart=rate]");

    mountEvents(panel, raining(), deps);
    await flush();

    expect(panel.querySelector("[data-sort=longest]")!.getAttribute("aria-pressed")).toBe("true");
    expect(panel.querySelector<HTMLSelectElement>(".ev-select")!.value).toBe(String(year));
    expect(panel.querySelector("[data-chart=rate]")!.getAttribute("aria-pressed")).toBe("true");
    expect(drawnLast()[3].chart).toBe("rate");
  });

  it("keeps a row closed by hand closed", () => {
    mountEvents(panel, raining(), deps);
    click(row(EVENT_3));
    expect(openRows()).toHaveLength(0);

    mountEvents(panel, raining(), deps);

    expect(openRows()).toHaveLength(0);
  });

  it("goes back to All when the new snapshot no longer has the year picked", () => {
    const year = raining().rain_events.seasons[0]!.water_year;
    mountEvents(panel, raining(), deps);
    pick(String(year));

    const next = raining();
    next.rain_events.seasons = [{ ...next.rain_events.seasons[0]!, water_year: year + 1 }];
    mountEvents(panel, next, deps);

    expect(panel.querySelector<HTMLSelectElement>(".ev-select")!.value).toBe("all");
    expect(starts()).toEqual([EVENT_3, EVENT_2, EVENT_1]);
  });

  it("falls back to the newest event, saying so, when the open one is gone", async () => {
    mountEvents(panel, raining(), deps);
    click(row(EVENT_1));
    await flush();

    const next = raining();
    next.rain_events.events = next.rain_events.events.filter((e) => e.start_ms !== EVENT_1);
    mountEvents(panel, next, deps);
    await flush();

    expect(openStart()).toBe(EVENT_3);
    expect(openRows()[0]!.textContent).toContain("The linked event is not in the catalog");
  });
});

describe("an empty catalog", () => {
  it("says there are no events yet, under the filter bar", () => {
    const snapshot = raining();
    snapshot.rain_events = {
      events: [],
      all_time: { count: 0, total_in: 0, median_in: null, mean_duration_minutes: null },
      seasons: [],
      open_curve: null,
    };

    mountEvents(panel, snapshot, deps);

    expect(panel.querySelector(".ev-title")!.textContent).toBe("No rain events in the catalog yet");
    expect(panel.querySelector(".ev-empty")!.textContent).toBe("No rain events on record yet.");
    expect(panel.querySelector(".ev-bar")).not.toBeNull();
  });
});

describe("each water year's gap status", () => {
  const status = (over: Partial<GapStatus>): GapStatus => ({
    key: "water_year",
    first_day: "2024-10-01",
    last_day: "2025-09-30",
    days: 365,
    missing_days: 0,
    status: "valid",
    reason: null,
    ...over,
  });

  function twoYears(older: GapStatus): DashboardSnapshot {
    const snapshot = raining();
    snapshot.rain_events.events[2]!.water_year = 2025;
    snapshot.rain_events.seasons = [
      {
        water_year: 2026,
        stats: { count: 2, total_in: 0.78, median_in: 0.39, mean_duration_minutes: 547.5 },
        status: status({ first_day: "2025-10-01", last_day: "2026-01-09", days: 101 }),
      },
      {
        water_year: 2025,
        stats: { count: 1, total_in: 0.36, median_in: 0.36, mean_duration_minutes: 530 },
        status: older,
      },
    ];
    return snapshot;
  }

  const option = (year: string) =>
    panel.querySelector<HTMLOptionElement>(`.ev-select option[value='${year}']`)!;

  it("names a non-comparable year's reason in its option and says why when it is picked", () => {
    const gap = status({ missing_days: 40, status: "invalid", reason: "gap" });
    mountEvents(panel, twoYears(gap), deps);

    expect(option("2025").dataset.gap).toBe("gap");
    expect(option("2025").textContent).toBe("2025, partial");
    expect(option("2025").title).toContain("n/a: gap in record");
    expect(option("2026").dataset.gap).toBeUndefined();
    expect(option("2026").textContent).toBe("2026");

    pick("2025");

    const note = panel.querySelector<HTMLElement>(".ev-season-gap")!;
    expect(note.hidden).toBe(false);
    expect(note.querySelector(".gap-hatch .gap-label")!.textContent).toBe("n/a: gap in record");
    expect(note.textContent).toContain(
      "Water year 2025 has days missing from the record, so the list can miss its events and their total is not the year's rain.",
    );
    // The events-only sum is still shown beside it, never hatched away.
    expect(panel.querySelector(".ev-stats")!.textContent).toContain("0.36 in");

    pick("2026");
    expect(panel.querySelector<HTMLElement>(".ev-season-gap")!.hidden).toBe(true);
  });

  it("reads a year the archive does not reach by the snapshot's own reason", () => {
    const missing = { missing_days: 365, status: "invalid", reason: "not_archived" } as const;
    mountEvents(panel, twoYears(status(missing)), deps);

    expect(option("2025").dataset.gap).toBe("not_archived");
    expect(option("2025").textContent).toBe("2025, not archived");
    pick("2025");
    expect(panel.querySelector(".ev-season-gap")!.textContent).toContain(
      "Water year 2025 is not archived yet",
    );
  });

  it("notes a prorated year without a mark or a ≈ on the events-only sum", () => {
    mountEvents(
      panel,
      twoYears(status({ missing_days: 3, status: "prorated", reason: "gap" })),
      deps,
    );

    expect(option("2025").textContent).toBe("2025");
    pick("2025");

    const note = panel.querySelector<HTMLElement>(".ev-season-gap")!;
    expect(note.querySelector(".gap-hatch")).toBeNull();
    expect(note.textContent).toBe(
      "3 of 365 days of water year 2025 are missing from the record, so the list can miss events in them.",
    );
    expect(panel.querySelector(".ev-stats")!.textContent).not.toContain("≈");
  });

  it("words a prorated year under All as it does for the one year", () => {
    mountEvents(
      panel,
      twoYears(status({ missing_days: 3, status: "prorated", reason: "gap" })),
      deps,
    );

    expect(panel.querySelector(".ev-season-gap")!.textContent).toBe(
      "3 of 365 days of water year 2025 are missing from the record, so the list can miss events in them.",
    );
  });

  it("names a year not covered and a prorated year together under All", () => {
    const snapshot = twoYears(status({ missing_days: 3, status: "prorated", reason: "gap" }));
    snapshot.rain_events.seasons[0]!.status = status({
      first_day: "2025-10-01",
      last_day: "2026-01-09",
      days: 101,
      missing_days: 40,
      status: "invalid",
      reason: "gap",
    });
    mountEvents(panel, snapshot, deps);

    expect(panel.querySelector(".ev-season-gap")!.textContent).toBe(
      "The record does not cover all of water year 2026, and 3 of 365 days of water year 2025 are missing from it, so the list can miss events in them.",
    );
  });

  it("names the years the record does not cover under All", () => {
    mountEvents(
      panel,
      twoYears(status({ missing_days: 40, status: "invalid", reason: "gap" })),
      deps,
    );

    expect(panel.querySelector(".ev-season-gap")!.textContent).toBe(
      "The record does not cover all of water year 2025, so the list can miss its events.",
    );
  });

  it("shows no note when every year is whole", () => {
    mountEvents(panel, twoYears(status({})), deps);

    expect(panel.querySelector<HTMLElement>(".ev-season-gap")!.hidden).toBe(true);
  });
});

describe("the filter bar", () => {
  it("pins under the sticky header", () => {
    document.body.innerHTML =
      "<div class=page><nav class=tabs style='position: sticky; top: 0px; height: 48px'></nav><section id=panel></section></div>";
    panel = document.getElementById("panel")!;
    const nav = document.querySelector<HTMLElement>(".tabs")!;
    nav.getBoundingClientRect = () => ({ height: 48 }) as DOMRect;

    mountEvents(panel, raining(), deps);

    const tile = panel.querySelector<HTMLElement>("[data-section=events]")!;
    expect(tile.style.getPropertyValue("--ev-bar-top")).toBe("48px");
    expect(tile.querySelector(".ev-bar")!.nextElementSibling!.className).toBe("ev-list");
  });
});
