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
import { nameFrom } from "./accessible-name.js";
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

/** The link's accessible name; the hidden spans leave a space before a stop. */
const nameOf = (el: Element) => nameFrom(el).replace(/ ([.,])/g, "$1");
const linkName = () => nameOf(link());

function draw(snapshot: DashboardSnapshot): void {
  renderOverview(panel, snapshot);
}

describe("the feature tile", () => {
  it.each(SCENARIOS)("follows the now tile on the %s golden, and What leads is gone", (s) => {
    draw(parseDashboard(golden(s, "dashboard")));
    const ids = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map(
      (t) => t.dataset.section,
    );

    expect(ids.slice(0, 3)).toEqual(["first-screen", "feature", "temperature"]);
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
    const name = nameOf(links[0]!);
    // The tile is no named region, so the link alone says the finding (#472).
    expect(tile()!.hasAttribute("aria-label")).toBe(false);
    // The content names the link: no aria-label replaces it, and the finding
    // is read once, the words round it being hidden text (#543).
    expect(tile()!.querySelector("[aria-label]")).toBeNull();
    expect(name).toMatch(/\. (Open|Go to) .+\.$/);
    expect(name.toLowerCase()).toContain(finding.toLowerCase());
    expect(name.toLowerCase().split(finding.toLowerCase())).toHaveLength(2);
    // Nothing inside the link is interactive in its own right.
    expect(links[0]!.querySelector("a, button, details, input")).toBeNull();
  });

  /** The four-column grid's cells the tiles fill: the feature 2×2, the rest their span. */
  const cells = () =>
    [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map((t) =>
      t.dataset.section === "feature" ? 4 : Number(t.dataset.span),
    );

  it("spans two columns of the four-column grid, beside the first four mosaic tiles", () => {
    draw(parseDashboard(golden("seasoned", "dashboard")));
    const ids = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map(
      (t) => t.dataset.section,
    );

    expect(tile()!.dataset.span).toBe("2");
    expect(ids.slice(1, 6)).toEqual([
      "feature",
      "temperature",
      "rain",
      "last-10-days",
      "dry-spell",
    ]);
    lastRowIsDaylightAndInside();
  });

  /**
   * Every row of the four-column grid is full but the last, which holds
   * Daylight and Inside with two empty cells (spec #460 "Overview layout").
   */
  function lastRowIsDaylightAndInside(): void {
    const ids = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map(
      (t) => t.dataset.section,
    );
    const before = cells().slice(0, -2);
    expect(ids.slice(-2)).toEqual(["daylight", "inside"]);
    expect(before.reduce((a, b) => a + b, 0) % 4, cells().join(",")).toBe(0);
  }

  it.each(SCENARIOS)(
    "fills each four-column row but the last, Daylight and Inside, on the %s golden without a feature (N-1)",
    (s) => {
      const withoutFeature = doc(s);
      delete withoutFeature.feature;
      draw(parseDashboard(withoutFeature));

      expect(tile()).toBeNull();
      lastRowIsDaylightAndInside();
    },
  );

  it("draws a live rain event: its total so far against the usual event", () => {
    const raining = parseDashboard(golden("raining", "dashboard"));
    draw(raining);

    expect(raining.feature?.kind).toBe("rain");
    expect(tile()!.dataset.kind).toBe("rain");
    expect(words(tile()!.querySelector("[data-part=kicker]"))).toBe("Raining now · 2 h 25 m");
    expect(words(tile()!.querySelector("[data-part=value]"))).toBe("0.30 in");
    expect(linkName()).toBe(
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
    expect(linkName()).toMatch(
      /^Rain ended (about )?3 hours ago, after \d\.\d\d in over [^,]+(, against 0\.42 in for a usual event)?\. Open the Rain Events tab\.$/,
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
    const name = linkName();
    expect(name).toMatch(
      /^Temperature 72.0°, warmer than usual for (\d+ (am|pm)|noon|midnight): the 100th percentile/,
    );
    expect(name).toContain("in earlier years");
    expect(name).toMatch(/\. Open the temperature view\.$/);
    expect(link().getAttribute("href")).toBe("#overview/temperature");
  });

  it.each([
    // Humidity and pressure open their zooms (#477), not the tile's old id.
    [
      "humidity",
      22,
      "pct",
      3,
      "22%",
      "drier",
      "3rd",
      "#overview/humidity",
      "Open the humidity view",
    ],
    ["wind_gust", 31.5, "mph", 98, "31.5 mph", "gustier", "98th", "#overview-wind", "Go to Wind"],
    [
      "pressure",
      29.41,
      "inhg",
      1,
      "29.41 inHg",
      "lower",
      "1st",
      "#overview/pressure",
      "Open the pressure view",
    ],
    ["solar", 140, "wm2", 2, "140 W/m²", "duller", "2nd", "#solar", "Open the Solar tab"],
  ])(
    "draws an unusual %s from a golden copy",
    (reading, value, unit, percentile, shown, word, ordinal, href, open) => {
      draw(edited("seasoned", unusual(reading, value, unit, percentile)));

      expect(words(tile()!.querySelector("[data-part=value]"))).toBe(shown);
      const name = linkName();
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
    expect(words(tile()!.querySelector("[data-part=value]"))).toBe("9 d 9 h");
    expect(linkName()).toBe(
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
    expect(linkName()).toBe(`a usual day, ${t}° now. Open the temperature view.`);
    expect(link().getAttribute("href")).toBe("#overview/temperature");
  });

  it("gives every Overview tile an id the in-page links reach", () => {
    draw(parseDashboard(golden("dry", "dashboard")));
    for (const id of ["temperature", "wind", "pressure", "humidity", "daylight", "inside"]) {
      expect(panel.querySelector(`#overview-${id}`), id).not.toBeNull();
    }
  });

  it("points at the temperature zoom on a phone, its tile open, the Overview having no finding rows", () => {
    setLayout("phone");
    draw(parseDashboard(golden("dry", "dashboard")));
    const target = panel.querySelector<HTMLElement>("#overview-temperature")!;

    expect(link().getAttribute("href")).toBe("#overview/temperature");
    expect(target.tagName).toBe("SECTION");
    expect(panel.querySelector("details.tile-row")).toBeNull();
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
    expect(panel.querySelector("[data-section=temperature]")).not.toBeNull();
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
