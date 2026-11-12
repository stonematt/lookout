// @vitest-environment happy-dom
/**
 * The feature tile (#469, spec #460 stories 10–17, 52–53): one tile under
 * the now tile, drawn from the snapshot's `feature` (ADR-0009), replacing
 * "What leads right now". Each kind renders from a golden or from a golden
 * copy edited here; the tile is one link named by its finding. The site
 * words and links; it never picks the feature.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { setLayout } from "../src/layout.js";
import { renderOverview } from "../src/overview/index.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard, SnapshotContractError } from "../src/snapshot.js";
import { golden, SCENARIOS } from "./golden.js";

type Doc = Record<string, unknown>;

const doc = (scenario: (typeof SCENARIOS)[number]) =>
  structuredClone(golden(scenario, "dashboard")) as Doc;

/** A golden copy with its feature (and anything else) replaced. */
function edited(scenario: (typeof SCENARIOS)[number], patch: Doc): DashboardSnapshot {
  return parseDashboard({ ...doc(scenario), ...patch });
}

const unusual = (reading: string, value: number, unit: string, percentile: number) => ({
  feature: { kind: "unusual", reading, value, unit, percentile },
});

let panel: HTMLElement;

beforeEach(() => {
  setLayout("four");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
});

const words = (node: Element | null | undefined) => node?.textContent?.replace(/\s+/g, " ").trim();
const tile = () => panel.querySelector<HTMLElement>("[data-section=feature]");
const link = () => tile()!.querySelector<HTMLAnchorElement>("a")!;

function draw(snapshot: DashboardSnapshot): void {
  renderOverview(panel, snapshot);
}

describe("the feature tile", () => {
  it.each(SCENARIOS)("follows the now tile on the %s golden, and What leads is gone", (s) => {
    draw(parseDashboard(golden(s, "dashboard")));
    const ids = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map(
      (t) => t.dataset.section,
    );

    expect(ids.slice(0, 3)).toEqual(["first-screen", "feature", "temperature-periods"]);
    expect(panel.querySelector("[data-section=hero-lead]")).toBeNull();
    expect(panel.querySelector("details[data-part=why]")).toBeNull();
    expect(panel.textContent).not.toMatch(/What leads|Why this leads|Leading:/);
  });

  it.each(SCENARIOS)("is one link on the %s golden, named by its finding", (s) => {
    draw(parseDashboard(golden(s, "dashboard")));
    const links = tile()!.querySelectorAll("a");

    expect(links).toHaveLength(1);
    // The whole tile is the link: the kicker, value and finding sit inside it.
    expect([...tile()!.children]).toEqual([links[0]]);
    const finding = words(tile()!.querySelector("[data-part=finding]"))!;
    const name = links[0]!.getAttribute("aria-label")!;
    expect(name.startsWith(tile()!.getAttribute("aria-label")!)).toBe(true);
    expect(name).toMatch(/\. (Open|Go to) .+\.$/);
    expect(name.toLowerCase()).toContain(finding.toLowerCase());
    // Nothing inside the link is interactive in its own right.
    expect(links[0]!.querySelector("a, button, details, input")).toBeNull();
  });

  it("spans half the four-column grid beside temperature, so every row is whole", () => {
    draw(parseDashboard(golden("seasoned", "dashboard")));
    const spans = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map((t) =>
      Number(t.dataset.span),
    );
    let row = 0;
    for (const span of spans) {
      row += span;
      if (row > 4) throw new Error(`row overflows at ${spans.join(",")}`);
      if (row === 4) row = 0;
    }
    expect(row).toBe(0);
    expect(tile()!.dataset.span).toBe("2");
  });

  it("draws a live rain event: its total so far against the usual event", () => {
    const raining = parseDashboard(golden("raining", "dashboard"));
    draw(raining);

    expect(raining.feature?.kind).toBe("rain");
    expect(tile()!.dataset.kind).toBe("rain");
    expect(words(tile()!.querySelector("[data-part=kicker]"))).toBe("Raining now · 2 h 25 m");
    expect(words(tile()!.querySelector("[data-part=value]"))).toBe("0.30 in");
    expect(link().getAttribute("aria-label")).toBe(
      "Raining now: 0.30 in so far, against 0.42 in for a usual event. Open the Rain Events tab.",
    );
    expect(link().getAttribute("href")).toBe("#events");
    expect(tile()!.querySelector("[data-part=live]")).not.toBeNull();
  });

  it("draws rain that ended: the newest event's total and how long ago it ended", () => {
    const base = doc("raining");
    const events = structuredClone(base.rain_events) as { events: Doc[] };
    for (const row of events.events) row.live = false;
    const newest = events.events[0]!;
    const snapshot = edited("raining", {
      active_rain_event: null,
      rain_events: { ...events, open_curve: null },
      built_at_ms: (newest.end_ms as number) + 3 * 3_600_000,
    });
    draw(snapshot);

    expect(words(tile()!.querySelector("[data-part=kicker]"))).toBe("Rain ended 3 h ago");
    expect(link().getAttribute("aria-label")).toMatch(
      /^Rain ended (about )?3 hours ago, after \d\.\d\d in(, against 0\.42 in for a usual event)?\. Open the Rain Events tab\.$/,
    );
    expect(link().getAttribute("href")).toBe("#events");
    expect(tile()!.querySelector("[data-part=live]")).toBeNull();
  });

  it("draws an unusual reading: its value and its percentile at this hour", () => {
    const seasoned = parseDashboard(golden("seasoned", "dashboard"));
    draw(seasoned);

    expect(seasoned.feature).toMatchObject({ kind: "unusual", reading: "temperature" });
    expect(words(tile()!.querySelector("[data-part=kicker]"))).toBe("Unusual · temperature");
    expect(words(tile()!.querySelector("[data-part=value]"))).toBe("72.0°");
    const name = link().getAttribute("aria-label")!;
    expect(name).toMatch(
      /^Temperature 72.0°, warmer than usual for (\d+ (am|pm)|noon|midnight): the 100th percentile/,
    );
    expect(name).toContain("in earlier years");
    expect(name).toMatch(/\. Go to Now against this date\.$/);
    expect(link().getAttribute("href")).toBe("#overview-against-normal");
  });

  it.each([
    ["humidity", 22, "pct", 3, "22%", "drier", "3rd", "#overview-air", "Go to Air"],
    ["wind_gust", 31.5, "mph", 98, "31.5 mph", "gustier", "98th", "#overview-wind", "Go to Wind"],
    [
      "pressure",
      29.41,
      "inhg",
      1,
      "29.41 inHg",
      "lower",
      "1st",
      "#overview-pressure",
      "Go to Pressure",
    ],
    ["solar", 140, "wm2", 2, "140 W/m²", "duller", "2nd", "#solar", "Open the Solar tab"],
  ])(
    "draws an unusual %s from a golden copy",
    (reading, value, unit, percentile, shown, word, ordinal, href, open) => {
      draw(edited("seasoned", unusual(reading, value, unit, percentile)));

      expect(words(tile()!.querySelector("[data-part=value]"))).toBe(shown);
      const name = link().getAttribute("aria-label")!;
      expect(name).toContain(`${shown}, ${word} than usual for `);
      expect(name).toContain(`the ${ordinal} percentile`);
      expect(name.endsWith(`. ${open}.`)).toBe(true);
      expect(link().getAttribute("href")).toBe(href);
    },
  );

  it("draws a long dry spell from a golden copy: its days against the usual spell", () => {
    draw(
      edited("dry", {
        feature: { kind: "dry_spell", reading: null, value: null, unit: null, percentile: null },
        dry_spell: { days_since_rain: 9.4, usual_days: 3 },
      }),
    );

    expect(words(tile()!.querySelector("[data-part=kicker]"))).toBe("Dry spell");
    expect(words(tile()!.querySelector("[data-part=value]"))).toBe("9 days");
    expect(link().getAttribute("aria-label")).toBe(
      "About 9 days since it last rained, against a usual dry spell of 3 days. Open the Rain tab.",
    );
    expect(link().getAttribute("href")).toBe("#rain");
  });

  it("draws a quiet day as a usual day, with the temperature now", () => {
    const dry = parseDashboard(golden("dry", "dashboard"));
    draw(dry);

    expect(dry.feature?.kind).toBe("quiet");
    expect(words(tile()!.querySelector("[data-part=kicker]"))).toBe("Today");
    const t = dry.current.temperature_f!.toFixed(1);
    expect(words(tile()!.querySelector("[data-part=value]"))).toBe(`${t}°`);
    expect(link().getAttribute("aria-label")).toBe(
      `A usual day, ${t}° now. Go to Now against this date.`,
    );
    expect(link().getAttribute("href")).toBe("#overview-against-normal");
  });

  it("gives every Overview tile an id the in-page links reach", () => {
    draw(parseDashboard(golden("dry", "dashboard")));
    for (const id of ["against-normal", "wind", "pressure", "air"]) {
      expect(panel.querySelector(`#overview-${id}`), id).not.toBeNull();
    }
  });

  it("opens the finding row it points to on a phone", () => {
    setLayout("phone");
    draw(parseDashboard(golden("dry", "dashboard")));
    const target = panel.querySelector<HTMLDetailsElement>("#overview-against-normal")!;

    expect(target.tagName).toBe("DETAILS");
    expect(target.open).toBe(false);
    link().click();
    expect(target.open).toBe(true);
  });

  it("is in the phone's fixed top, an open tile, never a finding row", () => {
    setLayout("phone");
    draw(parseDashboard(golden("raining", "dashboard")));

    expect(tile()!.tagName).toBe("SECTION");
    expect(tile()!.dataset.phoneTop).toBe("true");
  });
});

describe("the feature in a snapshot before Release A (N−1, ADR-0008)", () => {
  it.each(SCENARIOS)("draws no feature tile and no What leads for %s, and does not throw", (s) => {
    const old = doc(s);
    delete old.feature;
    delete old.dry_spell;
    const snapshot = parseDashboard(old);
    draw(snapshot);

    expect(tile()).toBeNull();
    expect(panel.querySelector("[data-section=hero-lead]")).toBeNull();
    expect(panel.querySelector("[data-section=first-screen]")).not.toBeNull();
    expect(panel.querySelector("[data-section=temperature-periods]")).not.toBeNull();
  });

  it.each([
    [{ kind: "lead" }],
    [{ kind: "unusual", reading: "temperature", value: null, unit: "f", percentile: 95 }],
    [{ kind: "unusual", reading: "dew_point", value: 50, unit: "f", percentile: 95 }],
    [{ kind: "quiet", reading: "temperature", value: 50, unit: "f", percentile: 50 }],
  ])("refuses a feature present but malformed: %j", (feature) => {
    expect(() => edited("dry", { feature })).toThrow(SnapshotContractError);
  });

  it("refuses a dry spell present but malformed", () => {
    expect(() => edited("dry", { dry_spell: { days_since_rain: "3" } })).toThrow(
      SnapshotContractError,
    );
  });
});
