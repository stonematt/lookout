// @vitest-environment happy-dom
/**
 * The Rain Events tab rendered from the golden snapshots, with the curve
 * source and the chart drawing faked (no network, no Plotly).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EventsDeps } from "../src/events/tab.js";
import { mountEvents } from "../src/events/tab.js";
import { ReadError } from "../src/load.js";
import { scrollEdges } from "../src/events/model.js";
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
const rows = () => [...panel.querySelectorAll<HTMLTableRowElement>(".ev-table tbody tr")];
const starts = () =>
  rows().map((r) => Number(r.querySelector<HTMLElement>(".ev-open")!.dataset.start));
const click = (selector: string) =>
  panel.querySelector<HTMLElement>(selector)!.dispatchEvent(
    new MouseEvent("click", { bubbles: true, cancelable: true }),
  );

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
    expect(rows()[0]!.classList.contains("pin")).toBe(true);
    expect(rows()[0]!.textContent).toContain("Live");
    expect(rows()[1]!.textContent).not.toContain("Live");
    expect(rows()[1]!.textContent).toContain("0.48");
    expect(rows()[1]!.textContent).toContain("excellent");
  });

  it("heads the list with the summary the snapshot computed", () => {
    mountEvents(panel, raining(), deps);

    expect(panel.querySelector(".ev-title")!.textContent).toBe(
      "The wettest of 3 events since Jan 4, 2026, brought 0.48 in, against 0.36 in for the median event",
    );
    const stats = panel.querySelector(".ev-stats")!.textContent!;
    expect(stats).toContain("3");
    expect(stats).toContain("0.36 in"); // the median event
  });

  it("sorts by total with the live event still pinned on top", () => {
    mountEvents(panel, raining(), deps);

    click("[data-sort=wettest]");

    expect(starts()).toEqual([EVENT_3, EVENT_2, EVENT_1]);
    expect(panel.querySelector("[data-sort=wettest]")!.getAttribute("aria-pressed")).toBe(
      "true",
    );
  });

  it("picks one water year's events and its summary", () => {
    const snapshot = raining();
    const section = snapshot.rain_events;
    section.events[2]!.water_year = 2025;
    section.seasons = [
      { ...section.seasons[0]!, stats: { count: 2, total_in: 0.78, median_in: 0.39, mean_duration_minutes: 547.5 } },
      { ...section.seasons[0]!, water_year: 2025, stats: { count: 1, total_in: 0.36, median_in: 0.36, mean_duration_minutes: 530 } },
    ];
    mountEvents(panel, snapshot, deps);

    click("[data-season='2025']");

    expect(starts()).toEqual([EVENT_1]);
    expect(panel.querySelector(".ev-title")!.textContent).toBe(
      "The one rain event in water year 2025 brought 0.36 in",
    );
  });
});

describe("the selected event", () => {
  it("opens the newest event when the URL names none, without writing it", async () => {
    mountEvents(panel, raining(), deps);
    await flush();

    expect(panel.querySelector(".ev-finding")!.textContent).toBe(
      "0.30 in so far over 2 h 25 m, and still raining",
    );
    expect(location.search).toBe("?fixture=raining");
  });

  it("charts an event still open but no longer live from the open curve, unmarked", async () => {
    // The catalog holds event 3 open; the live counter has ended it.
    const snapshot = raining();
    snapshot.rain_events.events[0]!.live = false;
    mountEvents(panel, snapshot, deps);
    await flush();

    expect(deps.fetchCurve).not.toHaveBeenCalled();
    expect(deps.draw.mock.calls[0]![1]).toBe(snapshot.rain_events.open_curve);
    expect(panel.querySelector(".ev-detail .ev-live")).toBeNull();
    expect(rows()[0]!.classList.contains("pin")).toBe(false);
  });

  it("charts the live event from the snapshot's open curve, not a fetch", async () => {
    const snapshot = raining();
    mountEvents(panel, snapshot, deps);
    await flush();

    expect(deps.fetchCurve).not.toHaveBeenCalled();
    expect(deps.draw).toHaveBeenCalledOnce();
    expect(deps.draw.mock.calls[0]![1]).toBe(snapshot.rain_events.open_curve);
    expect(panel.querySelector(".ev-detail .ev-live")).not.toBeNull();
  });

  it("answers a click once after a newer snapshot is mounted over it", async () => {
    // The page mounts the tab again whenever the snapshot changes.
    mountEvents(panel, raining(), deps);
    mountEvents(panel, raining(), deps);

    click(`.ev-open[data-start='${EVENT_2}']`);
    await flush();

    expect(deps.fetchCurve).toHaveBeenCalledTimes(1);
  });

  it("opens a completed event from its curve file and puts it in the URL", async () => {
    mountEvents(panel, raining(), deps);

    click(`.ev-open[data-start='${EVENT_2}']`);
    await flush();

    expect(location.search).toBe(`?fixture=raining&event=${EVENT_2}`);
    expect(location.hash).toBe("#events");
    expect(deps.fetchCurve).toHaveBeenCalledWith(EVENT_2);
    expect(deps.draw.mock.calls.at(-1)![1]).toEqual(curveOf(EVENT_2));
    expect(panel.querySelector(".ev-finding")!.textContent).toBe(
      "0.48 in over 15 h 50 m, the wettest of 3 events",
    );
    const current = panel.querySelector("tr[aria-current=true] .ev-open") as HTMLElement;
    expect(Number(current.dataset.start)).toBe(EVENT_2);
  });

  it("opens the event a shared link names", async () => {
    history.replaceState(null, "", `/?event=${EVENT_1}#events`);

    mountEvents(panel, raining(), deps);
    await flush();

    expect(deps.fetchCurve).toHaveBeenCalledWith(EVENT_1);
    expect(panel.querySelector(".ev-finding")!.textContent).toBe(
      "0.36 in over 8 h 50 m, the 2nd wettest of 3 events",
    );
  });

  it("gives each row a real link to its event", () => {
    mountEvents(panel, raining(), deps);

    const link = panel.querySelector<HTMLAnchorElement>(`.ev-open[data-start='${EVENT_1}']`)!;
    expect(link.getAttribute("href")).toBe(`?fixture=raining&event=${EVENT_1}#events`);
  });

  it("says so when a linked event is not in the catalog", async () => {
    history.replaceState(null, "", "/?event=1600000000000#events");

    mountEvents(panel, raining(), deps);
    await flush();

    expect(panel.querySelector(".ev-detail")!.textContent).toContain(
      "The linked event is not in the catalog",
    );
    expect(panel.querySelector(".ev-finding")!.textContent).toContain("still raining");
  });

  it("shows totals only, with the gap note, for an event with no archive data", async () => {
    const snapshot = raining();
    const row = snapshot.rain_events.events[2]!;
    Object.assign(row, { has_curve: false, quality: "invalid" });
    history.replaceState(null, "", `/?event=${EVENT_1}#events`);

    mountEvents(panel, snapshot, deps);
    await flush();

    expect(deps.fetchCurve).not.toHaveBeenCalled();
    expect(deps.draw).not.toHaveBeenCalled();
    const detail = panel.querySelector(".ev-detail")!;
    expect(detail.querySelector(".ev-gap.gap-hatch")!.textContent).toContain(
      "n/a: gap in record",
    );
    expect(detail.querySelector(".ev-facts")!.textContent).toContain("0.36 in");
    expect(detail.textContent).not.toMatch(/error/i);
    expect(rows()[2]!.textContent).toContain("no data");
  });

  it("says a curve not yet written is coming, not an error", async () => {
    deps.fetchCurve.mockRejectedValue(new ReadError("/event?start=1", 404));
    history.replaceState(null, "", `/?event=${EVENT_2}#events`);

    mountEvents(panel, raining(), deps);
    await flush();

    expect(panel.querySelector(".ev-charts")!.textContent).toContain("not published yet");
  });

  it("offers a retry when the curve cannot be read", async () => {
    deps.fetchCurve.mockRejectedValueOnce(new ReadError("/event?start=1", 502));
    history.replaceState(null, "", `/?event=${EVENT_2}#events`);
    mountEvents(panel, raining(), deps);
    await flush();

    click(".ev-retry");
    await flush();

    expect(deps.fetchCurve).toHaveBeenCalledTimes(2);
    expect(deps.draw).toHaveBeenCalledOnce();
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

  const finding = () => panel.querySelector(".ev-finding")!.textContent;
  const current = () =>
    Number(panel.querySelector<HTMLElement>("tr[aria-current=true] .ev-open")!.dataset.start);

  it("keeps the selected event and the rows Show more opened", async () => {
    const first = longer(30, 1);
    mountEvents(panel, first, deps);
    click(".ev-more");
    click(`.ev-open[data-start='${EVENT_2}']`);
    await flush();
    expect(rows()).toHaveLength(33);
    const opened = finding();

    mountEvents(panel, longer(30, 2), deps);
    await flush();

    expect(rows()).toHaveLength(33);
    expect(current()).toBe(EVENT_2);
    expect(finding()).toBe(opened);
    expect(panel.querySelector(".ev-detail")!.textContent).not.toContain("not in the catalog");
  });

  it("shows the selected event's row when newer events push it past the page", async () => {
    const first = longer(30, 1);
    mountEvents(panel, first, deps);
    const last = starts().at(-1)!; // row 20, the page's last
    click(`.ev-open[data-start='${last}']`);
    await flush();

    const next = longer(30, 2);
    const newer = { ...next.rain_events.events[1]!, start_ms: EVENT_2 + HOUR };
    next.rain_events.events.splice(1, 0, newer);
    mountEvents(panel, next, deps);

    expect(starts()).toContain(last);
    expect(current()).toBe(last);
  });

  it("redraws a kept completed event from the curve it already read", async () => {
    mountEvents(panel, raining(), deps);
    click(`.ev-open[data-start='${EVENT_2}']`);
    await flush();

    mountEvents(panel, raining(), deps);

    expect(panel.querySelector(".ev-charts")!.textContent).not.toContain("Loading");
    await flush();
    expect(deps.fetchCurve).toHaveBeenCalledTimes(1);
    expect(deps.draw).toHaveBeenCalledTimes(3); // the live event, event 2, event 2 again
    expect(deps.draw.mock.calls.at(-1)![1]).toEqual(curveOf(EVENT_2));
  });

  it("keeps the water year and the order picked", () => {
    mountEvents(panel, raining(), deps);
    click("[data-sort=wettest]");
    click(`[data-season='${raining().rain_events.seasons[0]!.water_year}']`);

    mountEvents(panel, raining(), deps);

    const pressed = (selector: string) => panel.querySelector(selector)!.getAttribute("aria-pressed");
    expect(pressed("[data-sort=wettest]")).toBe("true");
    expect(pressed(`[data-season='${raining().rain_events.seasons[0]!.water_year}']`)).toBe("true");
    expect(pressed("[data-season=all]")).toBe("false");
  });

  it("goes back to All when the new snapshot no longer has the year picked", () => {
    const year = raining().rain_events.seasons[0]!.water_year;
    mountEvents(panel, raining(), deps);
    click(`[data-season='${year}']`);

    const next = raining();
    next.rain_events.seasons = [{ ...next.rain_events.seasons[0]!, water_year: year + 1 }];
    mountEvents(panel, next, deps);

    expect(panel.querySelector("[data-season=all]")!.getAttribute("aria-pressed")).toBe("true");
    expect(starts()).toEqual([EVENT_3, EVENT_2, EVENT_1]);
  });

  it("falls back to the newest event, saying so, when the selected one is gone", async () => {
    mountEvents(panel, raining(), deps);
    click(`.ev-open[data-start='${EVENT_1}']`);
    await flush();

    const next = raining();
    next.rain_events.events = next.rain_events.events.filter((e) => e.start_ms !== EVENT_1);
    mountEvents(panel, next, deps);
    await flush();

    expect(current()).toBe(EVENT_3);
    expect(panel.querySelector(".ev-detail")!.textContent).toContain(
      "The linked event is not in the catalog",
    );
  });
});

describe("an empty catalog", () => {
  it("says there are no events yet", () => {
    const snapshot = raining();
    snapshot.rain_events = {
      events: [],
      all_time: { count: 0, total_in: 0, median_in: null, mean_duration_minutes: null },
      seasons: [],
      open_curve: null,
    };

    mountEvents(panel, snapshot, deps);

    expect(panel.querySelector(".ev-title")!.textContent).toBe(
      "No rain events in the catalog yet",
    );
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
      { water_year: 2026, stats: { count: 2, total_in: 0.78, median_in: 0.39, mean_duration_minutes: 547.5 }, status: status({ first_day: "2025-10-01", last_day: "2026-01-09", days: 101 }) },
      { water_year: 2025, stats: { count: 1, total_in: 0.36, median_in: 0.36, mean_duration_minutes: 530 }, status: older },
    ];
    return snapshot;
  }

  it("hatches a non-comparable year's button and says why when it is picked", () => {
    const gap = status({ missing_days: 40, status: "invalid", reason: "gap" });
    mountEvents(panel, twoYears(gap), deps);

    const button = panel.querySelector<HTMLElement>("[data-season='2025']")!;
    expect(button.dataset.gap).toBe("gap");
    expect(button.querySelector(".gap-hatch")).not.toBeNull();
    expect(button.textContent).toContain("n/a: gap in record");
    expect(button.title).toContain("n/a: gap in record");
    const fine = panel.querySelector<HTMLElement>("[data-season='2026']")!;
    expect(fine.dataset.gap).toBeUndefined();
    expect(fine.querySelector(".gap-hatch")).toBeNull();

    click("[data-season='2025']");

    const note = panel.querySelector<HTMLElement>(".ev-season-gap")!;
    expect(note.hidden).toBe(false);
    expect(note.querySelector(".gap-hatch .gap-label")!.textContent).toBe("n/a: gap in record");
    expect(note.textContent).toContain(
      "Water year 2025 has days missing from the record, so the list can miss its events and their total is not the year's rain.",
    );
    // The events-only sum is still shown beside it, never hatched away.
    expect(panel.querySelector(".ev-stats")!.textContent).toContain("0.36 in");

    click("[data-season='2026']");
    expect(panel.querySelector<HTMLElement>(".ev-season-gap")!.hidden).toBe(true);
  });

  it("reads a year the archive does not reach by the snapshot's own reason", () => {
    const missing = { missing_days: 365, status: "invalid", reason: "not_archived" } as const;
    mountEvents(panel, twoYears(status(missing)), deps);

    expect(panel.querySelector<HTMLElement>("[data-season='2025']")!.dataset.gap).toBe(
      "not_archived",
    );
    click("[data-season='2025']");
    expect(panel.querySelector(".ev-season-gap")!.textContent).toContain(
      "Water year 2025 is not archived yet",
    );
  });

  it("notes a prorated year without a hatch or a ≈ on the events-only sum", () => {
    mountEvents(panel, twoYears(status({ missing_days: 3, status: "prorated", reason: "gap" })), deps);

    const button = panel.querySelector<HTMLElement>("[data-season='2025']")!;
    expect(button.querySelector(".gap-hatch")).toBeNull();
    click("[data-season='2025']");

    const note = panel.querySelector<HTMLElement>(".ev-season-gap")!;
    expect(note.querySelector(".gap-hatch")).toBeNull();
    expect(note.textContent).toBe(
      "3 of 365 days of water year 2025 are missing from the record, so the list can miss events in them.",
    );
    expect(panel.querySelector(".ev-stats")!.textContent).not.toContain("≈");
  });

  it("words a prorated year under All as it does for the one year", () => {
    mountEvents(panel, twoYears(status({ missing_days: 3, status: "prorated", reason: "gap" })), deps);

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
    mountEvents(panel, twoYears(status({ missing_days: 40, status: "invalid", reason: "gap" })), deps);

    expect(panel.querySelector(".ev-season-gap")!.textContent).toBe(
      "The record does not cover all of water year 2025, so the list can miss its events.",
    );
  });

  it("shows no note when every year is whole", () => {
    mountEvents(panel, twoYears(status({})), deps);

    expect(panel.querySelector<HTMLElement>(".ev-season-gap")!.hidden).toBe(true);
  });
});

describe("the water-year control's scroll cue", () => {
  it("names the edges with years hidden past them", () => {
    expect(scrollEdges(0, 300, 300)).toBe("none");
    expect(scrollEdges(0, 600, 300)).toBe("end");
    expect(scrollEdges(150, 600, 300)).toBe("both");
    expect(scrollEdges(300, 600, 300)).toBe("start");
    expect(scrollEdges(299.5, 600, 300)).toBe("start");
  });
});
