// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from "vitest";

import { mountHeader } from "../src/header.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const MIN = 60_000;
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
const stale = () => host.querySelector(".masthead-reading")?.getAttribute("data-stale");

describe("the header", () => {
  it("shows the station, the reading time and a fresh data age", () => {
    const header = mountHeader(host, () => clock);
    header.show({ station: raining.station, observedAtMs: raining.observed_at_ms });

    expect(text("h1")).toBe("Fixture Station");
    expect(text("time")).toBe("Jan 8, 4:25 pm");
    expect(host.querySelector("time")?.getAttribute("datetime")).toBe(
      "2026-01-09T00:25:00.000Z",
    );
    expect(text(".masthead-age")).toBe("3 m ago");
    expect(stale()).toBe("false");
  });

  it("turns stale past 15 minutes as the clock runs, with no new data", () => {
    const header = mountHeader(host, () => clock);
    header.show({ station: raining.station, observedAtMs: raining.observed_at_ms });

    clock = RAINING_OBSERVED + 15 * MIN;
    header.tick();
    expect(stale()).toBe("false");

    clock = RAINING_OBSERVED + 16 * MIN;
    header.tick();
    expect(stale()).toBe("true");
    expect(text(".masthead-age")).toBe("Stale: 16 m old");
  });

  it("counts from the newest observation, not the snapshot build", () => {
    // The dry golden was built long ago: the age keeps counting from it.
    clock = Date.UTC(2026, 9, 30);
    const header = mountHeader(host, () => clock);
    header.show({ station: dry.station, observedAtMs: dry.observed_at_ms });
    expect(stale()).toBe("true");
    expect(text(".masthead-age")).toMatch(/^Stale: \d+ d \d+ h old$/);
  });

  it("never moves back to an older observation", () => {
    const header = mountHeader(host, () => clock);
    header.show({ station: raining.station, observedAtMs: raining.observed_at_ms });
    header.show({ station: null, observedAtMs: RAINING_OBSERVED - 60 * MIN });
    expect(text(".masthead-age")).toBe("3 m ago");
    expect(text("h1")).toBe("Fixture Station");
  });

  it("moves forward to a newer live observation", () => {
    const header = mountHeader(host, () => clock);
    header.show({ station: raining.station, observedAtMs: raining.observed_at_ms });
    header.show({ station: null, observedAtMs: RAINING_OBSERVED + 2 * MIN });
    expect(text(".masthead-age")).toBe("1 m ago");
  });

  it("names an unnamed station Lookout", () => {
    const header = mountHeader(host, () => clock);
    header.show({ station: { mac: "00:11:22:33:44:55", name: null }, observedAtMs: null });
    expect(text("h1")).toBe("Lookout");
    expect(text(".masthead-age")).toBe("No reading yet");
    expect(stale()).toBe("true");
  });

  it("says when the snapshot could not be loaded, and offers a retry", () => {
    const header = mountHeader(host, () => clock);
    let retried = 0;
    header.fail(() => retried++);
    expect(text(".masthead-reading")).toContain("could not be loaded");
    expect(text(".masthead-age")).toBe("");
    const retry = host.querySelector<HTMLButtonElement>(".masthead-retry")!;
    expect(retry.textContent).toBe("Try again");
    retry.click();
    expect(retried).toBe(1);
  });

  it("drops the failure once a reading arrives", () => {
    const header = mountHeader(host, () => clock);
    header.fail(() => undefined);
    header.show({ station: raining.station, observedAtMs: raining.observed_at_ms });
    expect(host.querySelector(".masthead-retry")).toBeNull();
    expect(text(".masthead-age")).toBe("3 m ago");
  });
});
