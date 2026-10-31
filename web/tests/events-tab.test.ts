// @vitest-environment happy-dom
/**
 * The Rain Events tab rendered from the golden snapshots, with the curve
 * source and the chart drawing faked (no network, no Plotly).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EventsDeps } from "../src/events/tab.js";
import { mountEvents } from "../src/events/tab.js";
import { ReadError } from "../src/load.js";
import type { DashboardSnapshot, EventCurve } from "../src/snapshot.js";
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
      "3 rain events since Jan 4, 2026, 1.14 in of rain in those events",
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
      { water_year: 2026, stats: { count: 2, total_in: 0.78, median_in: 0.39, mean_duration_minutes: 547.5 } },
      { water_year: 2025, stats: { count: 1, total_in: 0.36, median_in: 0.36, mean_duration_minutes: 530 } },
    ];
    mountEvents(panel, snapshot, deps);

    click("[data-season='2025']");

    expect(starts()).toEqual([EVENT_1]);
    expect(panel.querySelector(".ev-title")!.textContent).toBe(
      "1 rain event in water year 2025, 0.36 in of rain in those events",
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
