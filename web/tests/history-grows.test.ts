// @vitest-environment happy-dom
/**
 * History grows (#470, spec #460 stories 28–32): a comparison the record
 * does not support is not drawn, one line names the next unlock, a gap
 * inside a shown period stays visible, and Full history is a row in the
 * window range chart. Rendered from the goldens: the Overview, and the
 * sections that left it with the accordion (#472) for the zooms.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { setLayout } from "../src/layout.js";
import { almanacGrid, rollingRain } from "../src/overview/almanac.js";
import { renderOverview } from "../src/overview/index.js";
import { againstNormal, temperaturePeriods, tenDays } from "../src/overview/temperature.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { golden, SCENARIOS } from "./golden.js";

type Doc = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

const doc = (scenario: Scenario): Doc => golden(scenario, "dashboard") as Doc;
const parsed = (d: Doc): DashboardSnapshot => parseDashboard(d);

let panel: HTMLElement;

beforeEach(() => {
  document.documentElement.removeAttribute("data-layout");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

/** The zooms' sections, in a tile grid of their own. */
const ZOOM_SECTIONS = [temperaturePeriods, againstNormal, tenDays, almanacGrid, rollingRain];
const show = (snapshot: DashboardSnapshot) => renderOverview(panel, snapshot, ZOOM_SECTIONS);

const all = (selector: string) => [...panel.querySelectorAll<Element>(selector)];
const texts = (selector: string) => all(selector).map((el) => el.textContent ?? "");
const periodRows = () => texts("[data-part=periods-chart] .tc-row");
const dayRows = () => texts("[data-part=days-chart] .tc-row");

/** Delete every `shown` on a gap status, anywhere in the document. */
function dropShown(value: unknown): void {
  if (Array.isArray(value)) value.forEach(dropShown);
  else if (value && typeof value === "object") {
    const record = value as Doc;
    if ("missing_days" in record && "status" in record) delete record.shown;
    Object.values(record).forEach(dropShown);
  }
}

describe("no n/a row on the Overview", () => {
  it.each(SCENARIOS.flatMap((s) => [[s, "phone"] as const, [s, "four"] as const]))(
    "%s at %s: nothing reads n/a",
    (scenario, layout) => {
      if (layout === "four") setLayout("four");
      renderOverview(panel, parsed(doc(scenario)));
      expect(panel.textContent).not.toMatch(/n\/a/);

      show(parsed(doc(scenario)));
      expect(panel.textContent).not.toMatch(/n\/a/);
    },
  );

  it("draws no window range chart while the archive holds none of its rows (dry)", () => {
    show(parsed(doc("dry")));

    // Today, Yesterday and Full history are all not archived yet (#606).
    expect(all("[data-part=periods-chart]")).toHaveLength(0);
    // The days before the record began are not drawn; the three it holds are.
    expect(dayRows()).toEqual(["Wed, Nov 19", "Tue, Nov 18", "Mon, Nov 17"]);
  });

  it("draws no rolling-window rain tile while the record covers none of its windows", () => {
    show(parsed(doc("raining")));

    expect(panel.querySelector("[data-section=rolling-rain]")).toBeNull();
    expect(panel.querySelector("[data-section=almanac]")).not.toBeNull();
  });

  it("hatches no almanac month or season the record does not cover (dry)", () => {
    show(parsed(doc("dry")));

    const almanac = panel.querySelector("[data-section=almanac]")!;
    expect(almanac.querySelectorAll(".gap-hatch")).toHaveLength(0);
    // Oct, and Nov so far (it began before the record), as before the archive.
    const kinds = [...almanac.querySelectorAll<HTMLElement>("button.alm-cell")].map(
      (m) => `${m.dataset.month} ${m.dataset.kind}`,
    );
    expect(kinds).toEqual(["2025-10 before", "2025-11 before"]);
  });

  it("hides the rolling table's columns the record does not cover (seasoned)", () => {
    setLayout("four");
    show(parsed(doc("seasoned")));

    const heads = texts("[data-section=rolling-rain] thead th").slice(1);
    const years = (doc("seasoned").almanac.rolling as Doc[])[0]!.years.map((y: Doc) =>
      String(y.year),
    );
    expect(heads).toEqual(years.slice(0, heads.length));
    // A cell the record does not cover is empty; a shown gap reads "gap".
    const notShown = (doc("seasoned").almanac.rolling as Doc[]).flatMap((w) =>
      (w.years as Doc[])
        .filter((y) => y.status.shown === false && heads.includes(String(y.year)))
        .map((y) => [w.key as string, String(y.year)] as const),
    );
    expect(notShown.length).toBeGreaterThan(0);
    for (const [key, year] of notShown) {
      const cell = panel.querySelectorAll(
        `[data-section=rolling-rain] tr[data-window="${key}"] td`,
      )[heads.indexOf(year)]!;
      expect(cell.textContent).toBe("");
      expect(cell.classList.contains("gap-hatch")).toBe(false);
    }
    expect(new Set(texts("[data-section=rolling-rain] .gap-label"))).toEqual(new Set(["gap"]));
  });
});

describe("a gap inside a shown period", () => {
  it("stays hatched, worded without n/a (seasoned)", () => {
    show(parsed(doc("seasoned")));

    expect(periodRows()).toEqual([
      "Today",
      "Yesterday",
      "7 days",
      "30 days",
      "90 days",
      "365 days",
      "Full history",
    ]);
    // 365 days and Full history: shown, with days missing. 90 days is whole
    // since the seasoned fixture spans water years 2025–26 (#539).
    expect(texts("[data-part=periods-chart] .tc-reason")).toEqual([
      "gap in record",
      "gap in record",
    ]);
  });

  it("draws no chart of the dry golden's shown but unarchived windows, not a hatched block (#606)", () => {
    show(parsed(doc("dry")));

    expect(all("[data-part=periods-chart]")).toHaveLength(0);
    expect(texts("[data-block=periods] .tc-reason")).toEqual([]);
  });
});

describe("Full history", () => {
  it("is a row in the window range chart that draws a range (raining)", () => {
    show(parsed(doc("raining")));

    expect(periodRows()).toEqual(["Today", "Yesterday", "Full history"]);
    expect(all("[data-part=periods-chart] .tc-whisker")).toHaveLength(3);
    expect(texts("[data-part=periods-chart] .tc-value").slice(-2)).toEqual(["40°", "60°"]);
    expect(texts("[data-block=periods] .ov-sr li").at(-1)).toBe(
      "Full history (Jan 4 to Jan 8): low 40.0°, middle half 42.6° to 57.1°, high 60.0°.",
    );
  });
});

describe("the next-unlock line", () => {
  it("names the comparison the snapshot says unlocks next (dry)", () => {
    show(parsed(doc("dry")));

    const lines = all("[data-part=next-unlock]");
    expect(lines).toHaveLength(1);
    expect(lines[0]!.textContent).toBe("Next to unlock: 7 days, on Nov 23, 2025.");
    expect(lines[0]!.closest("[data-section]")?.getAttribute("data-section")).toBe(
      "temperature-periods",
    );
  });

  it("says nothing when the snapshot names no unlock (seasoned)", () => {
    show(parsed(doc("seasoned")));

    expect(all("[data-part=next-unlock], .ov-unlock")).toHaveLength(0);
  });

  it("says nothing for a null next_unlock, as live on Lookout Post", () => {
    const d = doc("raining");
    d.record_depth.next_unlock = null;
    expect(() => show(parsed(d))).not.toThrow();

    expect(all("[data-part=next-unlock], .ov-unlock")).toHaveLength(0);
    expect(panel.textContent).not.toMatch(/unlock/i);
  });

  it("words an unlock with an unknown key without the key", () => {
    const d = doc("dry");
    d.record_depth.next_unlock = { key: "fortnight", date: "2025-12-01" };
    show(parsed(d));

    expect(texts("[data-part=next-unlock]")).toEqual([
      "More of the record unlocks on Dec 1, 2025.",
    ]);
  });

  it("is a contract error when malformed", () => {
    const d = doc("dry");
    d.record_depth.next_unlock = { key: "7d" };
    expect(() => parsed(d)).toThrow(SnapshotContractError);
  });
});

describe("a null almanac.grid_finding, as live on Salmon Run", () => {
  it.each(SCENARIOS)("%s: renders the almanac with its headline and no empty line", (scenario) => {
    const d = doc(scenario);
    d.almanac.grid_finding = null;
    expect(() => show(parsed(d))).not.toThrow();

    const almanac = panel.querySelector("[data-section=almanac]")!;
    const finding = almanac.querySelector("[data-part=finding]")!;
    expect(finding.textContent!.trim().length).toBeGreaterThan(0);
    for (const el of almanac.querySelectorAll("p, h2, h3")) {
      expect(el.textContent!.trim(), el.outerHTML).not.toBe("");
    }
  });
});

describe("a snapshot written before Release A (N−1)", () => {
  /** A golden with record_depth, full_history and every shown deleted. */
  function previous(scenario: Scenario): DashboardSnapshot {
    const d = doc(scenario);
    delete d.record_depth;
    delete d.temperature_context.full_history;
    dropShown(d);
    return parsed(d);
  }

  it.each(SCENARIOS)("%s: renders today's view without crashing", (scenario) => {
    expect(() => show(previous(scenario))).not.toThrow();

    // The dry golden's archive holds none of its windows, so it draws no
    // chart (#606); the others draw every window, hatched where they must.
    expect(periodRows()).toEqual(
      scenario === "dry" ? [] : ["Today", "Yesterday", "7 days", "30 days", "90 days", "365 days"],
    );
    expect(dayRows()).toHaveLength(10);
    expect(all("[data-part=next-unlock]")).toHaveLength(0);
  });

  it("keeps today's hatched rows in that fallback (raining)", () => {
    show(previous("raining"));

    expect(all("[data-part=periods-chart] .tc-hatch")).toHaveLength(4);
    expect(panel.querySelector("[data-section=rolling-rain]")).not.toBeNull();
  });

  it("reads a null full_history as no row", () => {
    const d = doc("raining");
    d.temperature_context.full_history = null;
    show(parsed(d));

    expect(periodRows()).toEqual(["Today", "Yesterday"]);
  });
});
