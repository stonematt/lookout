// @vitest-environment happy-dom
/**
 * Snapshot values reach the page as text, never as markup: a snapshot whose
 * strings (and fields typed as numbers) hold HTML renders that HTML as text.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import { escapeHtml, html } from "../src/dom.js";
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

  it("escapes every value interpolated into html, unless html built it", () => {
    const name = HOSTILE;
    const cell = html`<dd>${name}</dd>`;
    const block = html`<dl title="${name}">${cell}${[cell, html`<b>${7}</b>`]}${name}</dl>`;
    const escaped = escapeHtml(HOSTILE);

    expect(String(block)).toBe(
      `<dl title="${escaped}"><dd>${escaped}</dd><dd>${escaped}</dd><b>7</b>${escaped}</dl>`,
    );
    // Markup that only looks like a fragment, a plain string, is escaped.
    expect(String(html`${String(cell)}`)).toBe(escapeHtml(`<dd>${escaped}</dd>`));
    document.body.innerHTML = html`${block}`;
    expect(document.querySelector("img")).toBeNull();
    expect(document.querySelector("dd")!.textContent).toBe(HOSTILE);
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
    expect(host.querySelector(".ev-chip")!.textContent).toBe(HOSTILE);
    const season = host.querySelectorAll<HTMLOptionElement>(".ev-select option")[1]!;
    // The golden's water year starts before the archive: partial, and why.
    expect(season.textContent).toBe(`${HOSTILE}, partial`);
    expect(season.title).toContain("n/a: before the archive");
    expect(season.value).toBe(HOSTILE);
    expect(season.title).toContain(HOSTILE);
  });

  it("renders the station name in the header as text", () => {
    const header = mountHeader(host, () => NOW);
    header.showSnapshot(hostile());

    expect(host.querySelector("img")).toBeNull();
    expect(host.querySelector("h1")!.textContent).toBe(HOSTILE);
  });
});
