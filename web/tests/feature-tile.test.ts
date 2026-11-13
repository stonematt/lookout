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
/** The finding as it shows, without the hidden words that run on from it. */
function findingWords(): string | undefined {
  const finding = tile()!.querySelector("[data-part=finding]")?.cloneNode(true) as Element;
  finding?.querySelectorAll(".ov-sr-run").forEach((run) => run.remove());
  return words(finding);
}

/**
 * The link's accessible name, exactly: the hidden words that follow the
 * finding run on from it inside its own element, so no space comes before
 * their stop or comma.
 */
const nameOf = (el: Element) => nameFrom(el);
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
    const finding = findingWords()!;
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

  /** The four-column grid's cells the tiles fill: each tile's span. */
  const cells = () =>
    [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map((t) =>
      Number(t.dataset.span),
    );

  it("spans the whole four-column row, the mosaic in rows of four under it", () => {
    draw(parseDashboard(golden("seasoned", "dashboard")));
    const ids = [...panel.querySelectorAll<HTMLElement>(".tile-grid > .tile")].map(
      (t) => t.dataset.section,
    );

    expect(tile()!.dataset.span).toBe("4");
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

  describe("finds an ended event by the feature's event_start_ms (#523)", () => {
    /** The raining golden with every event ended and the feature naming `start`. */
    function ended(start: number | null | undefined): { snapshot: DashboardSnapshot; rows: Doc[] } {
      const base = doc("raining");
      const events = structuredClone(base.rain_events) as { events: Doc[] };
      events.events.forEach((row, i) => {
        row.live = false;
        row.total_in = 0.11 * (i + 1);
      });
      const feature = { ...(base.feature as Doc) };
      if (start === undefined) delete feature.event_start_ms;
      else feature.event_start_ms = start;
      const snapshot = edited("raining", {
        active_rain_event: null,
        rain_events: { ...events, open_curve: null },
        feature,
        built_at_ms: (events.events[1]!.end_ms as number) + 3_600_000,
      });
      return { snapshot, rows: events.events };
    }
    const value = () => words(tile()!.querySelector("[data-part=value]"));

    it("draws the event the feature names, not the catalog's newest", () => {
      const probe = ended(undefined);
      const second = probe.rows[1]!;
      draw(ended(second.start_ms as number).snapshot);

      expect(value()).toBe("0.22 in");
    });

    it("keeps the catalog's newest ended event in a snapshot without the field (N−1)", () => {
      draw(ended(undefined).snapshot);

      expect(value()).toBe("0.11 in");
    });

    it("draws rain lately, no number, when no row has the named start", () => {
      draw(ended(1).snapshot);

      expect(tile()!.querySelector("[data-part=value]")).toBeNull();
      expect(words(tile()!.querySelector("[data-part=kicker]"))).toBe("Rain");
    });

    it("keeps the catalog's newest ended event when the field is null", () => {
      draw(ended(null).snapshot);

      expect(value()).toBe("0.11 in");
    });
  });

  it("refuses a feature whose event_start_ms is not a time", () => {
    const base = doc("raining");
    const feature = { ...(base.feature as Doc), event_start_ms: "soon" };

    expect(() => parseDashboard({ ...base, feature })).toThrow(SnapshotContractError);
  });

  it("draws an unusual reading: its value and its side of usual at this hour", () => {
    const seasoned = parseDashboard(golden("seasoned", "dashboard"));
    draw(seasoned);

    expect(seasoned.feature).toMatchObject({ kind: "unusual", reading: "temperature" });
    expect(words(tile()!.querySelector("[data-part=kicker]"))).toBe("Unusual · temperature");
    expect(words(tile()!.querySelector("[data-part=value]"))).toBe("72.0°");
    const name = linkName();
    // The golden's reading sits at the 100th percentile: past every earlier
    // reading for this hour, which is said in words, never as "100th" (#563).
    expect(name).toBe(
      "Temperature 72.0°. Warmer than usual for 3 pm: highest for this hour on record. " +
        "Open the temperature view.",
    );
    expect(link().getAttribute("href")).toBe("#overview/temperature");
  });

  it.each([
    [100, "humidity", 99, "pct", "More humid than usual for 3 pm: highest for this hour on record"],
    [0, "humidity", 22, "pct", "Drier than usual for 3 pm: lowest for this hour on record"],
    [0, "temperature", 31.0, "f", "Cooler than usual for 3 pm: lowest for this hour on record"],
  ])(
    "words the %sth percentile as past every reading for this hour, never as a percentile",
    (percentile, reading, value, unit, finding) => {
      draw(edited("seasoned", unusual(reading, value, unit, percentile)));
      expect(findingWords()).toBe(finding);
      expect(tile()!.textContent).not.toMatch(/\b(0th|100th)\b|percentile/);
      expect(linkName()).not.toMatch(/\b(0th|100th)\b|percentile/);
    },
  );

  it("names the reading's day when it is from before midnight", () => {
    // The dry golden is built at midnight from yesterday's 11:55 pm reading.
    draw(edited("dry", unusual("temperature", 31.0, "f", 4)));
    expect(findingWords()).toBe("Cooler than usual for 11 pm yesterday");
  });

  // The voice pass (#595): each unusual reading, on each side of its usual
  // range, says the reading and its side in plain words, never the
  // percentile the snapshot decided it by; the method is the footer's.
  it.each([
    // reading, value, unit, percentile, shown, the spoken lead, finding, zoom
    ["temperature", 64, "f", 96, "64.0°", "Temperature 64.0°", "Warmer than usual", "temperature"],
    ["temperature", 41, "f", 4, "41.0°", "Temperature 41.0°", "Cooler than usual", "temperature"],
    ["humidity", 91, "pct", 97, "91%", "Humidity 91%", "More humid than usual", "humidity"],
    ["humidity", 22, "pct", 3, "22%", "Humidity 22%", "Drier than usual", "humidity"],
    [
      "pressure",
      30.41,
      "inhg",
      95,
      "30.41 inHg",
      "30.41 inHg",
      "Pressure higher than usual",
      "pressure",
    ],
    [
      "pressure",
      29.41,
      "inhg",
      1,
      "29.41 inHg",
      "29.41 inHg",
      "Pressure lower than usual",
      "pressure",
    ],
    ["solar", 910, "wm2", 62, "910 W/m²", "Sun 910 W/m²", "Brighter than usual", "sun"],
    ["solar", 140, "wm2", 2, "140 W/m²", "Sun 140 W/m²", "Dimmer than usual", "sun"],
    // Wind gust is no longer judged (#533) but the site still reads it.
    ["wind_gust", 31.5, "mph", 98, "31.5 mph", "Wind gust 31.5 mph", "Gustier than usual", "wind"],
    ["wind_gust", 2.5, "mph", 5, "2.5 mph", "Wind gust 2.5 mph", "Calmer than usual", "wind"],
  ])(
    "words an unusual %s at %s on its side, with no percentile",
    (reading, value, unit, percentile, shown, lead, side, zoom) => {
      draw(edited("seasoned", unusual(reading, value, unit, percentile)));

      expect(words(tile()!.querySelector("[data-part=value]"))).toBe(shown);
      expect(findingWords()).toBe(`${side} for 3 pm`);
      expect(linkName()).toBe(`${lead}. ${side} for 3 pm. Open the ${zoom} view.`);
      expect(link().getAttribute("href")).toBe(`#overview/${zoom}`);
      expect(`${tile()!.textContent} ${linkName()}`).not.toMatch(/percentile|\b\d+(st|nd|rd|th)\b/);
    },
  );

  // The hold (#533, #578): Release A3 adds `since_ms`, `held` and
  // `outside_usual`. The tile keys its "unusual" wording on `outside_usual`,
  // absent counting as outside, so snapshots from before A3 read as today.
  describe("an unusual reading and the hold", () => {
    const built = 1779314460000; // the seasoned golden's build, 3 pm Pacific
    const since = built - 2 * 3_600_000; // 1 pm Pacific
    const tileText = () => ({
      kicker: words(tile()!.querySelector("[data-part=kicker]")),
      value: words(tile()!.querySelector("[data-part=value]")),
      finding: findingWords(),
      name: linkName(),
    });
    const today = () => {
      draw(parseDashboard(golden("seasoned", "dashboard")));
      return tileText();
    };

    it("reads exactly as today with none of the hold's fields", () => {
      const expected = today();
      draw(edited("seasoned", unusual("temperature", 72, "f", 100)));

      expect(tileText()).toEqual(expected);
      expect(expected.kicker).toBe("Unusual · temperature");
    });

    it.each([
      ["held", true],
      ["not held", false],
    ])("reads as today while outside usual, %s, with no since", (_label, held) => {
      const expected = today();
      draw(
        edited("seasoned", {
          feature: {
            ...unusual("temperature", 72, "f", 100).feature,
            since_ms: since,
            held,
            outside_usual: true,
          },
        }),
      );

      expect(tileText()).toEqual(expected);
      expect(tileText().name).not.toMatch(/ was | at 1 pm/);
    });

    it.each([
      [
        "temperature",
        64,
        "f",
        62,
        "Temperature was outside its usual range at 1 pm; back inside now",
        "64.0°",
      ],
      [
        "humidity",
        48,
        "pct",
        30,
        "Humidity was outside its usual range at 1 pm; back inside now",
        "48%",
      ],
    ])(
      "says a held %s back inside its band was outside, and when, naming no side",
      (reading, value, unit, percentile, finding, shown) => {
        draw(
          edited("seasoned", {
            feature: {
              ...unusual(reading, value, unit, percentile).feature,
              since_ms: since,
              held: true,
              outside_usual: false,
            },
          }),
        );
        const text = tileText();

        expect(text.finding).toBe(finding);
        expect(text.value).toBe(shown);
        expect(text.kicker).toBe(`Earlier · ${reading}`);
        expect(text.name).toContain(finding);
        const all = `${text.kicker} ${text.finding} ${text.name}`;
        expect(all).not.toMatch(/unusual|than usual|percentile|held|above|below/i);
      },
    );
  });

  it("refuses the hold's fields present but malformed", () => {
    const base = unusual("temperature", 72, "f", 100).feature;
    for (const patch of [{ since_ms: "1" }, { held: 1 }, { outside_usual: "no" }]) {
      expect(() => edited("seasoned", { feature: { ...base, ...patch } })).toThrow(
        SnapshotContractError,
      );
    }
  });

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
      "About 9 days since it last rained, against a usual dry spell of 3 days. Open the dry spell view.",
    );
    expect(link().getAttribute("href")).toBe("#overview/dry-spell");
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
