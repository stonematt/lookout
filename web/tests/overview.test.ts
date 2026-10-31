// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from "vitest";

import { conditionSentence } from "../src/overview/condition.js";
import { renderOverview, SECTIONS } from "../src/overview/index.js";
import { clearNowLabel, rainStep, ribbonClaim, ribbonTicks } from "../src/overview/ribbon.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const raining = parseDashboard(golden("raining", "dashboard"));
const dry = parseDashboard(golden("dry", "dashboard"));

let panel: HTMLElement;

beforeEach(() => {
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const text = (selector: string) =>
  panel.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim();
const all = (selector: string) => [...panel.querySelectorAll<HTMLElement>(selector)];

describe("the Overview", () => {
  it("opens with the first screen, in the Phone First-Screen order", () => {
    renderOverview(panel, raining);

    const first = panel.querySelector("[data-section=first-screen]")!;
    expect(panel.firstElementChild).toBe(first);
    const order = [...first.querySelectorAll("[data-part]")].map((el) =>
      el.getAttribute("data-part"),
    );
    expect(order).toEqual(["temperature", "condition", "right-now", "ribbon"]);
  });

  it("renders each section into its own host, in order", () => {
    const seen: string[] = [];
    renderOverview(panel, dry, [
      { id: "a", label: "A", render: (host) => void seen.push(host.dataset.section!) },
      { id: "b", label: "B", render: (host) => void seen.push(host.dataset.section!) },
    ]);

    expect(seen).toEqual(["a", "b"]);
    expect(all(".ov-section").map((el) => el.getAttribute("aria-label"))).toEqual([
      "A",
      "B",
    ]);
  });

  it("starts its section list with the first screen", () => {
    expect(SECTIONS[0]?.id).toBe("first-screen");
  });
});

describe("the current temperature and its condition sentence", () => {
  it("shows the live temperature large", () => {
    renderOverview(panel, raining);

    expect(text("[data-part=temperature]")).toBe("59.3°F");
  });

  it("says it is raining, with the rate and today's live total", () => {
    renderOverview(panel, raining);

    expect(text("[data-part=condition]")).toBe(
      "Mild and lightly breezy; raining at 0.12 in/hr, 0.30 in so far today.",
    );
  });

  it("names the last rain on a dry day", () => {
    const today = structuredClone(dry);
    today.current.rain_today_in = 0;
    renderOverview(panel, today);

    expect(text("[data-part=condition]")).toBe(
      "Cold, damp and still; no rain today, the last on Nov 17, 9:03 am.",
    );
  });

  it("says nothing of rain today when the newest reading is yesterday's", () => {
    // Built at midnight from a 23:55 reading: rain today is not known.
    renderOverview(panel, dry);

    expect(text("[data-part=condition]")).toBe("Cold, damp and still.");
  });

  it("names the trend against a day before, and rain earlier today", () => {
    const sentence = conditionSentence({
      ...dry.current,
      temperature_f: 82,
      humidity_pct: 30,
      wind_speed_mph: 14,
      temp_trend: "rising",
      rain_today_in: 0.04,
      last_rain_ms: dry.observed_at_ms! - 90 * 60_000,
    });

    expect(sentence).toBe(
      "Hot, dry and windy, warmer than this time yesterday; " +
        "0.04 in of rain today, the last at Nov 18, 10:25 pm.",
    );
  });

  it("says what it does not know rather than guessing", () => {
    const sentence = conditionSentence({
      ...dry.current,
      temperature_f: null,
      humidity_pct: null,
      wind_speed_mph: null,
      rain_today_in: null,
      rain_rate_in_hr: null,
      last_rain_ms: null,
    });

    expect(sentence).toBe("No conditions reported.");
  });
});

describe("the Right-now strip", () => {
  it("lists feels like, wind and rain today from the snapshot's reading", () => {
    renderOverview(panel, raining);

    const items = all("[data-part=right-now] > div").map((div) => [
      div.querySelector("dt")?.textContent,
      div.querySelector("dd")?.textContent,
    ]);
    expect(items).toEqual([
      ["feels like", "59.3°"],
      ["wind", "3.0 mph"],
      ["rain today", "0.30 in"],
    ]);
  });

  it("shows n/a for a value the station did not report", () => {
    renderOverview(panel, {
      ...dry,
      current: { ...dry.current, wind_speed_mph: null },
    });

    expect(all("[data-part=right-now] dd")[1]?.textContent).toBe("n/a");
  });
});

describe("the 48-hour condition ribbon", () => {
  it("draws night, day and rain slots, in four rain steps", () => {
    renderOverview(panel, raining);

    const rain = all(".ribbon-band rect[data-rain]").map((r) => r.dataset.rain);
    // Event 2 at 0.06 in/hr (step 2) and event 3 at 0.12 in/hr (step 3).
    expect(new Set(rain)).toEqual(new Set(["2", "3"]));
    expect(all(".ribbon-band rect[data-light=day]").length).toBeGreaterThan(0);
    expect(all(".ribbon-band rect[data-light=night]").length).toBeGreaterThan(0);
    expect(rainStep(0.0125)).toBe(1); // 0.05 in/hr
    expect(rainStep(0.025)).toBe(2); // 0.10 in/hr
    expect(rainStep(0.075)).toBe(3); // 0.30 in/hr
    expect(rainStep(0.1)).toBe(4);
    expect(rainStep(0)).toBeNull();
  });

  it("traces the temperature and ends at the live value", () => {
    renderOverview(panel, raining);

    expect(all(".ribbon-trace polyline").length).toBeGreaterThan(0);
    expect(text(".ribbon-now")).toBe("now 59.3°");
    expect(panel.querySelector(".ribbon-dot")).not.toBeNull();
  });

  it("hatches the span after the archive, with its reason", () => {
    renderOverview(panel, raining);

    const hatches = all(".ribbon-hatch");
    expect(hatches).toHaveLength(1);
    expect(hatches[0]?.dataset.reason).toBe("not archived yet");
    // The last two of 192 slots, anchored at the right edge.
    expect(hatches[0]?.style.right).toBe("0px");
    expect(parseFloat(hatches[0]!.style.width)).toBeCloseTo((2 / 192) * 100);
  });

  it("hatches a gap in the record as a gap", () => {
    renderOverview(panel, dry);

    const reasons = all(".ribbon-hatch").map((el) => el.dataset.reason);
    expect(reasons).toContain("gap in record");
    expect(reasons.at(-1)).toBe("not archived yet");
  });

  it("states the range and the rain hours above the band", () => {
    expect(ribbonClaim(raining.condition_ribbon!)).toBe(
      "40° to 60° over the last 48 hours, with rain in 9\u00a0h of the 47\u00a0h\u00a030\u00a0m on record.",
    );
    expect(ribbonClaim({ ...raining.condition_ribbon!, rain_hours: 0, known_hours: 48 })).toBe(
      "40° to 60° over the last 48 hours, with no rain.",
    );
  });

  it("ticks every six hours in Pacific time, clear of the now label", () => {
    const ticks = ribbonTicks(raining.condition_ribbon!).map((t) => t.label);

    // Jan 6, 4:30 pm to Jan 8, 4:30 pm Pacific.
    // Thursday noon (91% across) sits under the "now" label.
    expect(ticks).toEqual(["Wed", "6 am", "noon", "6 pm", "Thu", "6 am"]);
    // Thursday 6 am (78%) would run into it on a phone: marked late.
    expect(ribbonTicks(raining.condition_ribbon!).map((t) => t.late)).toEqual([
      false, false, false, false, false, true,
    ]);
  });

  it("hides any tick label the now label would run into, at any width", () => {
    renderOverview(panel, raining);
    const row = panel.querySelector<HTMLElement>(".ribbon-ticks")!;
    // Lay the row out by hand: ticks 30px wide every 80px, the now label at
    // 420-490px, over the last tick (400-430px).
    const box = (left: number, width: number) => () =>
      ({ left, right: left + width, width, top: 0, bottom: 16, height: 16 }) as DOMRect;
    const ticks = all(".ribbon-tick");
    ticks.forEach((tick, i) => (tick.getBoundingClientRect = box(i * 80, 30)));
    panel.querySelector<HTMLElement>(".ribbon-now")!.getBoundingClientRect = box(420, 70);

    clearNowLabel(row);

    // The one before it (320-350px) is clear by more than the 6px a label
    // keeps from its neighbour.
    expect(ticks.map((t) => t.classList.contains("ribbon-tick-hidden"))).toEqual([
      false, false, false, false, false, true,
    ]);

    // A wider now label, from 354px: within 6px of the fifth tick too.
    panel.querySelector<HTMLElement>(".ribbon-now")!.getBoundingClientRect = box(354, 136);
    clearNowLabel(row);
    expect(ticks.map((t) => t.classList.contains("ribbon-tick-hidden"))).toEqual([
      false, false, false, false, true, true,
    ]);
  });

  it("leaves the ticks alone before the row is laid out", () => {
    renderOverview(panel, raining);
    const row = panel.querySelector<HTMLElement>(".ribbon-ticks")!;

    clearNowLabel(row);

    expect(all(".ribbon-tick-hidden")).toHaveLength(0);
  });

  it("names the archive lag in the source line, not in a headline", () => {
    renderOverview(panel, raining);

    expect(text(".ov-source")).toBe(
      "Temperature, feels like, wind and rain today are the station's reading at " +
        "Jan 8, 4:25 pm. " +
        "The ribbon is the archive in 15-minute slots, to Jan 8, 3:55 pm; " +
        "the 30 m after it is hatched, as only the live total is known there.",
    );
    expect(panel.querySelector("[data-part=condition]")?.textContent).not.toMatch(
      /archive/,
    );
  });

  it("dates the strip's reading itself, so it never borrows a fresher one", () => {
    renderOverview(panel, raining);
    expect(text(".ov-source")).toMatch(
      /^Temperature, feels like, wind and rain today are the station's reading at Jan 8, 4:25 pm\./,
    );
    expect(text(".ov-source")).not.toMatch(/live reading/);
  });

  it("says when the strip has no reading at all", () => {
    renderOverview(panel, { ...dry, observed_at_ms: null, condition_ribbon: null });
    expect(text(".ov-source")).toBe(
      "Temperature, feels like, wind and rain today have no reading yet.",
    );
  });

  it("gives the band an accessible name", () => {
    renderOverview(panel, raining);

    expect(panel.querySelector(".ribbon-band")?.getAttribute("role")).toBe("img");
    expect(panel.querySelector(".ribbon-band")?.getAttribute("aria-label")).toMatch(
      /^48-hour condition ribbon: 40° to 60°/,
    );
  });

  it("is left out when there is no ribbon", () => {
    const none: DashboardSnapshot = { ...dry, condition_ribbon: null };
    renderOverview(panel, none);

    expect(panel.querySelector("[data-part=ribbon]")).toBeNull();
    expect(panel.querySelector("[data-part=temperature]")).not.toBeNull();
  });
});
