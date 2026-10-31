// @vitest-environment happy-dom
/**
 * The Solar tab (#85) against the golden snapshots (spec #66 seam 2), the
 * selected day's round trip through the URL, and its words against worked
 * examples.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { kwhPerM2 as kwh } from "../src/format.js";
import { shade, shadeKey } from "../src/solar/charts.js";
import { renderSolar } from "../src/solar/index.js";
import {
  dailyHeadline,
  dayHeadline,
  openingDay,
  readDay,
  selectedDay,
  slotText,
  traceHeadline,
  withSolarDay,
} from "../src/solar/text.js";
import type { DashboardSnapshot, SolarTrace } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { golden } from "./golden.js";

const raining = parseDashboard(golden("raining", "dashboard"));
const dry = parseDashboard(golden("dry", "dashboard"));

let panel: HTMLElement;

beforeEach(() => {
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const section = (id: string) => panel.querySelector(`[data-section=${id}]`)!;

/** Render with a page query, recording every query the tab writes. */
function render(snapshot: DashboardSnapshot, query = ""): string[] {
  const written: string[] = [];
  let current = query;
  renderSolar(
    panel,
    snapshot,
    () => current,
    (next) => {
      current = next;
      written.push(next);
    },
  );
  return written;
}

/** The value shown for a window in the strip. */
function stripValue(label: string): Element | undefined {
  const item = [...section("daily").querySelectorAll(".solar-stats div")].find((d) => text(d.querySelector("dt")) === label);
  return item?.querySelector("dd") ?? undefined;
}

describe("the contract", () => {
  it("parses both goldens with their Solar tab", () => {
    expect(raining.solar_tab?.days.first_day).toBe("2026-01-04");
    expect(dry.solar_tab?.recent.energy_wh_m2[0]).toHaveLength(96);
  });

  it("rejects a Solar tab whose days are not parallel", () => {
    const doc = golden("raining", "dashboard") as { solar_tab: { days: { reason: unknown[] } } };
    doc.solar_tab.days.reason.pop();
    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });

  it("rejects a Solar window without its own gap status", () => {
    const doc = golden("raining", "dashboard") as { solar_tab: { windows: { status?: unknown }[] } };
    delete doc.solar_tab.windows[0]!.status;
    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });

  it("allows a snapshot with no energy catalog", () => {
    const doc = golden("dry", "dashboard") as { solar_tab: unknown };
    doc.solar_tab = null;
    expect(parseDashboard(doc).solar_tab).toBeNull();
  });
});

describe("daily energy", () => {
  it("shows each window's energy from the snapshot", () => {
    render(raining);
    expect(text(stripValue("Today"))).toBe("3.79 kWh/m²");
    expect(text(stripValue("Yesterday"))).toBe("4.07 kWh/m²");
  });

  it("hatches a window that cannot be compared", () => {
    render(raining);
    expect(stripValue("7 days")?.classList.contains("gap-hatch")).toBe(true);
  });

  it("judges a window by solar's own status, which counts the catalog's missing days", () => {
    // The archive's status says valid; solar's says prorated.
    const copy = structuredClone(raining);
    const window = copy.solar_tab!.windows.find((w) => w.key === "today")!;
    expect(copy.gaps.windows.find((w) => w.key === "today")?.status).toBe("valid");
    window.status = { ...window.status, status: "prorated", reason: "gap", missing_days: 1 };
    render(copy);
    expect(text(stripValue("Today"))).toMatch(/^≈ 3\.79 kWh\/m²/);
  });

  it("hatches a window the catalog leaves invalid though the archive is whole", () => {
    const copy = structuredClone(raining);
    const window = copy.solar_tab!.windows.find((w) => w.key === "yesterday")!;
    expect(copy.gaps.windows.find((w) => w.key === "yesterday")?.status).toBe("valid");
    window.energy_kwh_m2 = null;
    window.status = { ...window.status, status: "invalid", reason: "gap", missing_days: 1 };
    render(copy);
    expect(stripValue("Yesterday")?.classList.contains("gap-hatch")).toBe(true);
  });

  it("draws the daily bars", () => {
    render(raining);
    expect(section("daily").querySelectorAll(".solar-bar").length).toBe(5);
  });
});

describe("the 24-hour view", () => {
  it("leads with the energy over the last 24 hours against the 24 before", () => {
    render(raining);
    expect(text(section("now").querySelector("h2"))).toBe(
      "3.95 kWh/m² of sun in the last 24 hours, about the same as 4.07 the 24 before",
    );
  });

  it("ends at the live reading, the same number as current conditions", () => {
    render(raining);
    expect(raining.current.solar_wm2).toBe(322.2);
    expect(text(section("now").querySelector(".solar-now-label"))).toBe("now 322 W/m²");
  });

  it("names the live reading when the 24 hours cannot be totalled", () => {
    render(dry);
    expect(text(section("now").querySelector("h2"))).toBe("Solar radiation is 0 W/m² now");
    expect(text(section("now").querySelector(".solar-source"))).toContain("The 24-hour energy is not totalled");
  });
});

describe("the selected day in the URL", () => {
  it("opens on the day the URL names", () => {
    render(raining, "?fixture=raining&solar_day=2026-01-06");
    expect(text(section("day").querySelector("h2"))).toBe("Jan 6, 2026: 4.07 kWh/m², brightest at noon");
  });

  it("opens on today without one, or with one outside the record", () => {
    render(raining, "?solar_day=2019-01-01");
    expect(text(section("day").querySelector("h2"))).toMatch(/^Jan 8, 2026 so far: 3\.79 kWh\/m²/);
    expect(openingDay(raining.solar_tab!, "")).toBe("2026-01-08");
  });

  it("writes a picked day back to the URL, keeping the other parameters", () => {
    const written = render(raining, "?fixture=raining&solar_day=2026-01-06");
    section("day").querySelector<HTMLButtonElement>(".solar-button")!.click(); // Previous day
    expect(written).toEqual(["?fixture=raining&solar_day=2026-01-05"]);
    expect(text(section("day").querySelector("h2"))).toMatch(/^Jan 5, 2026: /);
  });

  it("reads back what it writes", () => {
    const query = withSolarDay("?fixture=dry&age=3", "2025-11-17");
    expect(query).toBe("?fixture=dry&age=3&solar_day=2025-11-17");
    expect(selectedDay(query)).toBe("2025-11-17");
    expect(withSolarDay(query, null)).toBe("?fixture=dry&age=3");
    expect(selectedDay("?solar_day=yesterday")).toBeNull();
  });

  it("hatches a picked day that is not in the record, with its reason", () => {
    render(dry, "?solar_day=2025-11-18");
    const block = section("day").querySelector(".gap-hatch");
    expect(block).not.toBeNull();
    expect(text(block)).toContain("not archived yet");
  });

  it("gives the total alone for a day older than the 15-minute window", () => {
    const copy = structuredClone(raining);
    const tab = copy.solar_tab!;
    tab.recent.first_day = "2026-01-06";
    tab.recent.energy_wh_m2 = tab.recent.energy_wh_m2.slice(-3);
    render(copy, "?solar_day=2026-01-05");
    expect(text(section("day").querySelector(".solar-older"))).toBe(
      "Jan 5, 2026 gave 4.07 kWh/m². Its 15-minute energy is kept for the last 90 days only.",
    );
  });
});

describe("the heatmaps", () => {
  it("switches between its three views", () => {
    render(raining);
    const buttons = [...section("heatmaps").querySelectorAll<HTMLButtonElement>(".solar-seg button")];
    expect(buttons.map((b) => text(b))).toEqual(["Months", "Days", "15 minutes"]);
    expect(buttons[1]!.getAttribute("aria-pressed")).toBe("true");
    expect(text(section("heatmaps").querySelector("h2"))).toBe(
      "The brightest day on record was Jan 4, 2026, at 4.07 kWh/m²",
    );
    buttons[0]!.click();
    expect(buttons[0]!.getAttribute("aria-pressed")).toBe("true");
    // January is still under way, so no month can be compared yet.
    expect(text(section("heatmaps").querySelector("h2"))).toBe("Solar energy each month");
  });
});

describe("words", () => {
  it("writes energy with fewer decimals as it grows", () => {
    expect(kwh(2.414)).toBe("2.41 kWh/m²");
    expect(kwh(61.23)).toBe("61.2 kWh/m²");
    expect(kwh(182.4)).toBe("182 kWh/m²");
  });

  it("writes slots as clock times", () => {
    expect(slotText(0)).toBe("midnight");
    expect(slotText(41)).toBe("10:15 am");
    expect(slotText(48)).toBe("noon");
    expect(slotText(54)).toBe("1:30 pm");
  });

  it("compares the 24 hours with the 24 before", () => {
    const trace = { energy_kwh_m2: 3.1, previous_kwh_m2: 2.0 } as SolarTrace;
    expect(traceHeadline(trace, 500)).toBe("3.10 kWh/m² of sun in the last 24 hours, up from 2.00 the 24 before");
    expect(traceHeadline({ ...trace, previous_kwh_m2: 4.0 }, 500)).toContain("down from 4.00");
    expect(traceHeadline({ ...trace, previous_kwh_m2: null }, 500)).toBe("3.10 kWh/m² of sun in the last 24 hours");
  });

  it("compares the last 7 full days with the same dates in earlier years", () => {
    const copy = structuredClone(raining);
    const days = copy.solar_tab!.days;
    // Jan 1, 2024 to Jan 8, 2026: 2.00 a day, then 3.10 on Jan 1-7, 2026.
    const count = 739;
    days.first_day = "2024-01-01";
    days.energy_kwh_m2 = Array.from({ length: count }, (_, i) => (i >= count - 8 && i < count - 1 ? 3.1 : 2.0));
    days.reason = days.energy_kwh_m2.map(() => null);
    expect(dailyHeadline(copy)).toBe(
      "Brighter than this week in earlier years: 3.10 kWh/m² a day over the last 7 full days, against 2.00 (2025) and 2.00 (2024)",
    );
  });

  it("falls back to the last 30 days, or to no claim at all", () => {
    expect(dailyHeadline(raining)).toBe("Sun each day over the last 90 days");
  });

  it("names a day's brightest 15 minutes", () => {
    const day = readDay(raining.solar_tab!, "2026-01-05");
    expect(dayHeadline(day)).toBe("Jan 5, 2026: 4.07 kWh/m², brightest at noon");
  });

  it("shades on five steps and keys them", () => {
    expect([0, 0.1, 1, 2.5, 5].map((v) => shade(v, 5))).toEqual([0, 1, 1, 3, 5]);
    expect(shadeKey(5, "kWh/m²")[4]).toEqual(["solar-step-5", "up to 5.00 kWh/m²"]);
  });
});
