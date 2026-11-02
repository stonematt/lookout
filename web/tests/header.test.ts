// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from "vitest";

import { figuresLag, figuresTime, mountHeader } from "../src/header.js";
import type { CurrentReading } from "../src/snapshot.js";
import { SCHEMA_VERSION, parseCurrent, parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const MIN = 60_000;
const HOUR = 60 * MIN;
const raining = parseDashboard(golden("raining", "dashboard"));
const dry = parseDashboard(golden("dry", "dashboard"));
const RAINING_OBSERVED = 1767918300000; // Jan 8, 4:25 pm Pacific

let host: HTMLElement;
let clock: number;

beforeEach(() => {
  document.body.innerHTML = "<div id=host></div>";
  host = document.getElementById("host")!;
  clock = RAINING_OBSERVED + 3 * MIN;
});

const text = (selector: string) => host.querySelector(selector)?.textContent?.trim();
const stale = (selector = ".masthead-reading") =>
  host.querySelector(selector)?.getAttribute("data-stale");
const figuresShown = () =>
  !host.querySelector<HTMLElement>(".masthead-figures")?.hidden;

/** A `/current` body whose live reading was taken at `observedAtMs`. */
function live(
  observedAtMs: number | null,
  station: CurrentReading["station"] = raining.station,
): CurrentReading {
  return parseCurrent({
    schema_version: SCHEMA_VERSION,
    kind: "current",
    station,
    observed_at_ms: observedAtMs,
    built_at_ms: null,
    last_data: observedAtMs === null ? null : { dateutc: observedAtMs },
    context: null,
  });
}

describe("the header", () => {
  it("shows the station, the reading time and a fresh data age", () => {
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);

    expect(text("h1")).toBe("Fixture Station");
    expect(text(".masthead-reading time")).toBe("Jan 8, 4:25 pm");
    expect(host.querySelector("time")?.getAttribute("datetime")).toBe(
      "2026-01-09T00:25:00.000Z",
    );
    expect(text(".masthead-reading .masthead-age")).toBe("3 m ago");
    expect(stale()).toBe("false");
  });

  it("turns stale past 15 minutes as the clock runs, with no new data", () => {
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);

    clock = RAINING_OBSERVED + 15 * MIN;
    header.tick();
    expect(stale()).toBe("false");

    clock = RAINING_OBSERVED + 16 * MIN;
    header.tick();
    expect(stale()).toBe("true");
    expect(text(".masthead-reading .masthead-age")).toBe("Stale: 16 m old");
  });

  it("counts from the newest observation, not the snapshot build", () => {
    // The dry golden was built long ago: the age keeps counting from it.
    clock = Date.UTC(2026, 9, 30);
    const header = mountHeader(host, () => clock);
    header.showSnapshot(dry);
    expect(stale()).toBe("true");
    expect(text(".masthead-reading .masthead-age")).toMatch(/^Stale: \d+ d \d+ h old$/);
  });

  it("never moves back to an older observation", () => {
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    header.showCurrent(live(RAINING_OBSERVED - 60 * MIN));
    expect(text(".masthead-reading .masthead-age")).toBe("3 m ago");
    expect(text("h1")).toBe("Fixture Station");
  });

  it("moves forward to a newer live observation", () => {
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    header.showCurrent(live(RAINING_OBSERVED + 2 * MIN));
    expect(text(".masthead-reading .masthead-age")).toBe("1 m ago");
  });

  it("keeps a known station name when a later reading has none", () => {
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    header.showCurrent(live(RAINING_OBSERVED, { mac: raining.station.mac, name: null }));
    expect(text("h1")).toBe("Fixture Station");
  });

  it("names an unnamed station Lookout", () => {
    const header = mountHeader(host, () => clock);
    header.showSnapshot({
      ...raining,
      station: { ...raining.station, name: null },
      observed_at_ms: null,
    });
    expect(text("h1")).toBe("Lookout");
    expect(text(".masthead-reading .masthead-age")).toBe("No reading yet");
    expect(stale()).toBe("true");
    expect(figuresShown()).toBe(false);
  });

  it("says when the snapshot could not be loaded, and offers a retry", () => {
    const header = mountHeader(host, () => clock);
    let retried = 0;
    header.fail(() => retried++);
    expect(text(".masthead-reading")).toContain("could not be loaded");
    expect(text(".masthead-reading .masthead-age")).toBe("");
    expect(figuresShown()).toBe(false);
    const retry = host.querySelector<HTMLButtonElement>(".masthead-retry")!;
    expect(retry.textContent).toBe("Try again");
    retry.click();
    expect(retried).toBe(1);
  });

  it("keeps the failure when only /current answers", () => {
    const header = mountHeader(host, () => clock);
    header.fail(() => undefined);
    header.showCurrent(live(clock - MIN));
    expect(text(".masthead-reading")).toContain("could not be loaded");
    expect(host.querySelector(".masthead-retry")).not.toBeNull();
    expect(text(".masthead-reading .masthead-age")).toBe("");

    // The retry brings the snapshot; the live reading it kept is not lost.
    header.showSnapshot(raining);
    expect(text(".masthead-reading .masthead-age")).toBe("1 m ago");
  });

  it("drops the failure once a reading arrives", () => {
    const header = mountHeader(host, () => clock);
    header.fail(() => undefined);
    header.showSnapshot(raining);
    expect(host.querySelector(".masthead-retry")).toBeNull();
    expect(text(".masthead-reading .masthead-age")).toBe("3 m ago");
  });
});

describe("the figures' own age", () => {
  it("stays out of the way when the snapshot and the live reading are both fresh", () => {
    clock = RAINING_OBSERVED + 4 * MIN;
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    header.showCurrent(live(RAINING_OBSERVED + 3 * MIN));

    expect(text(".masthead-reading .masthead-age")).toBe("1 m ago");
    expect(stale()).toBe("false");
    expect(figuresShown()).toBe(false);
  });

  it("names a 6-hour-old snapshot under a fresh live reading", () => {
    clock = RAINING_OBSERVED + 6 * HOUR;
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    header.showCurrent(live(clock - 2 * MIN));

    // The live reading is fresh, and says so.
    expect(text(".masthead-reading .masthead-age")).toBe("2 m ago");
    expect(stale()).toBe("false");
    // The page's figures do not borrow its freshness.
    expect(figuresShown()).toBe(true);
    expect(text(".masthead-figures time")).toBe("Jan 8, 4:25 pm");
    expect(text(".masthead-figures")).toMatch(/^Numbers below from Jan 8, 4:25 pm/);
    // Two lines: the reading line names its source, so neither time is mistaken.
    expect(text(".masthead-reading .masthead-when")).toMatch(/^Station last reported /);
    expect(text(".masthead-figures .masthead-age")).toBe("Stale: 6 h old");
    expect(stale(".masthead-figures")).toBe("true");
  });

  it("keeps counting an old snapshot when /current fails", () => {
    clock = RAINING_OBSERVED + 6 * HOUR;
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    // No showCurrent: /current could not be read.

    // One age, the snapshot's own, stale on its own clock.
    expect(text(".masthead-reading .masthead-when")).toBe("Reading Jan 8, 4:25 pm");
    expect(text(".masthead-reading .masthead-age")).toBe("Stale: 6 h old");
    expect(stale()).toBe("true");
    expect(figuresShown()).toBe(false);
  });

  it("appears once the figures go stale while the live reading stays fresh", () => {
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    expect(figuresShown()).toBe(false);

    // The page stays open; /current keeps arriving, the snapshot does not.
    clock = RAINING_OBSERVED + 16 * MIN;
    header.showCurrent(live(clock - 1 * MIN));
    expect(stale()).toBe("false");
    expect(figuresShown()).toBe(true);
    expect(text(".masthead-figures .masthead-age")).toBe("Stale: 16 m old");

    clock = RAINING_OBSERVED + 31 * MIN;
    header.tick();
    expect(text(".masthead-figures .masthead-age")).toBe("Stale: 31 m old");
  });

  it("stays hidden while both are stale and close together", () => {
    clock = RAINING_OBSERVED + 40 * MIN;
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    header.showCurrent(live(RAINING_OBSERVED + 10 * MIN));
    expect(stale()).toBe("true");
    expect(figuresShown()).toBe(false);
  });

  it("shows when both are stale but far apart", () => {
    clock = RAINING_OBSERVED + 6 * HOUR;
    const header = mountHeader(host, () => clock);
    header.showSnapshot(raining);
    header.showCurrent(live(clock - HOUR));
    expect(text(".masthead-reading .masthead-age")).toBe("Stale: 1 h old");
    expect(figuresShown()).toBe(true);
    expect(text(".masthead-figures .masthead-age")).toBe("Stale: 6 h old");
  });

  it("decides the figures' line from the two times alone", () => {
    const now = RAINING_OBSERVED + HOUR;
    expect(figuresLag(now - MIN, now - 10 * MIN, now)).toBe(false); // both fresh
    expect(figuresLag(now - MIN, now - 16 * MIN, now)).toBe(true); // figures stale
    expect(figuresLag(now - 20 * MIN, now - 30 * MIN, now)).toBe(false); // close
    expect(figuresLag(now - 20 * MIN, now - 40 * MIN, now)).toBe(true); // far apart
    expect(figuresLag(null, now - HOUR, now)).toBe(false);
    expect(figuresLag(now, null, now)).toBe(false);
  });

  it("dates the figures by the snapshot's observation, else its build", () => {
    expect(figuresTime(raining)).toBe(RAINING_OBSERVED);
    expect(figuresTime({ ...raining, observed_at_ms: null })).toBe(raining.built_at_ms);
  });
});
