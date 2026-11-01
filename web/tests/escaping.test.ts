// @vitest-environment happy-dom
/**
 * Snapshot values reach the page as text, never as markup: a snapshot whose
 * strings (and fields typed as numbers) hold HTML renders that HTML as text.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import { escapeHtml } from "../src/dom.js";
import { mountEvents } from "../src/events/tab.js";
import type { EventsDeps } from "../src/events/tab.js";
import { mountHeader } from "../src/header.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const HOSTILE = `<img src=x onerror="alert(1)">"'&`;
const NOW = Date.UTC(2026, 0, 9, 0, 30);

/** The raining golden with markup in every interpolated value it can hold. */
function hostile(): DashboardSnapshot {
  const doc = structuredClone(golden("raining", "dashboard")) as {
    station: { name: string };
    rain_events: {
      seasons: { water_year: unknown }[];
      events: { water_year: unknown; quality: unknown }[];
    };
  };
  doc.station.name = HOSTILE;
  for (const s of doc.rain_events.seasons) s.water_year = HOSTILE;
  for (const e of doc.rain_events.events) {
    e.water_year = HOSTILE;
    e.quality = HOSTILE;
  }
  return parseDashboard(doc);
}

let host: HTMLElement;

beforeEach(() => {
  history.replaceState(null, "", "/?fixture=raining#events");
  document.body.innerHTML = "<section id=host></section>";
  host = document.getElementById("host")!;
});

describe("escaping", () => {
  it("escapes the five HTML-significant characters, and numbers pass", () => {
    expect(escapeHtml(`<a href="x">'&'</a>`)).toBe(
      "&#60;a href=&#34;x&#34;&#62;&#39;&#38;&#39;&#60;/a&#62;",
    );
    expect(escapeHtml(2026)).toBe("2026");
  });

  it("renders the Rain Events tab's snapshot values as text", () => {
    const deps: EventsDeps = {
      fetchCurve: vi.fn(),
      draw: vi.fn(async () => {}),
      nowMs: () => NOW,
    };
    mountEvents(host, hostile(), deps);

    expect(host.querySelector("img")).toBeNull();
    expect(host.querySelector("[onerror]")).toBeNull();
    expect(host.querySelector(".ev-kicker")!.textContent).toContain(HOSTILE);
    expect(host.querySelector(".ev-chip")!.textContent).toBe(HOSTILE);
    const season = host.querySelectorAll<HTMLButtonElement>("[data-season]")[1]!;
    expect(season.textContent).toBe(HOSTILE);
    expect(season.dataset.season).toBe(HOSTILE);
    expect(season.title).toContain(HOSTILE);
  });

  it("renders the station name in the header as text", () => {
    const header = mountHeader(host, () => NOW);
    header.showSnapshot(hostile());

    expect(host.querySelector("img")).toBeNull();
    expect(host.querySelector("h1")!.textContent).toBe(HOSTILE);
  });
});
