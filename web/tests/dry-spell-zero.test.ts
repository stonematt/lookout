// @vitest-environment happy-dom
/**
 * A version 5 dry spell of 0 whole days (#694 item 15, product lead on #647,
 * 2026-11-24): no whole dry day has passed since the last rain, so there is
 * no dry spell yet. Every dry-spell reader (the Dry spell tile and its zoom,
 * the feature, the Rain tab, the Scriptable widget) reads "No dry spell yet"
 * with "last rain 9 h ago" under it, from the snapshot's last rain
 * (`rain_now.last_rain_ms`), never "0 dry days"; at 1 it reads "1 dry day".
 * Documents are version 5, built from the goldens (`goldenNext`); the goldens
 * stay version 4.
 */
import { createRequire } from "node:module";

import { beforeEach, describe, expect, it } from "vitest";

import { setLayout } from "../src/layout.js";
import { featureView } from "../src/overview/feature.js";
import { renderOverview } from "../src/overview/index.js";
import { drySpellTile } from "../src/overview/mosaic-tiles.js";
import { renderRain } from "../src/rain/index.js";
import { dryHeadline, drySpellShown } from "../src/rain/text.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { goldenNext } from "./golden.js";

type Doc = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

const HOUR = 3_600_000;

/**
 * The seasoned golden (dry mode, built Wed May 20 2026, 3:01 pm) as a
 * version 5 document: `days` whole dry days, last rain `hoursAgo` before
 * the build.
 */
function v5Doc(days: number, hoursAgo: number): Doc {
  const doc = goldenNext("seasoned", "dashboard", days) as Doc;
  doc.dry_spell.usual_days = 2;
  doc.rain_now = { ...doc.rain_now, last_rain_ms: doc.built_at_ms - hoursAgo * HOUR };
  return doc;
}

const v5 = (days: number, hoursAgo: number): DashboardSnapshot =>
  parseDashboard(v5Doc(days, hoursAgo));

/** Rain this morning at 6:01 am: no whole dry day yet. */
const rainToday = () => v5(0, 9);
/** Rain late yesterday, 11:31 pm, none today: still no whole dry day. */
const rainLateYesterday = () => v5(0, 15.5);
/** Exactly one whole dry day. */
const oneDay = () => v5(1, 40);

let panel: HTMLElement;

beforeEach(() => {
  setLayout("phone");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
  history.replaceState(null, "", "/");
});

const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();

describe("the dry spell reading on a version 5 document", () => {
  it.each([
    ["rain today", rainToday, "last rain 9 h ago"],
    ["rain late yesterday", rainLateYesterday, "last rain 15 h ago"],
  ])("reads no dry spell yet with %s, never 0 dry days", (_, build, under) => {
    const shown = drySpellShown(build())!;
    expect(shown).toMatchObject({ value: "No dry spell yet", unit: null, under });
    expect(dryHeadline(build())).toBe(`No dry spell yet, ${under}`);
  });

  it("reads 1 dry day in the singular", () => {
    expect(drySpellShown(oneDay())).toMatchObject({ value: "1", unit: "day", under: null });
    expect(dryHeadline(oneDay())).toBe(
      "1 dry day since it last rained, against a usual dry spell of 2 days",
    );
  });
});

describe("every dry-spell reader at 0 whole days", () => {
  it("draws the Dry spell tile as no dry spell yet, last rain under it", () => {
    expect(drySpellTile.read(rainToday())).toMatchObject({
      value: "No dry spell yet",
      delta: "last rain 9 h ago",
    });
    renderOverview(panel, rainToday());
    const tile = panel.querySelector("[data-section=dry-spell]")!;
    expect(words(tile.querySelector("[data-part=value]"))).toBe("No dry spell yet");
    expect(words(tile.querySelector("[data-part=delta]"))).toBe("last rain 9 h ago");
    expect(words(panel)).not.toMatch(/\b0 dry days?\b|\b0 days\b/);
  });

  it("reads 1 day on the tile at exactly one whole dry day", () => {
    expect(drySpellTile.read(oneDay())).toMatchObject({ value: "1", unit: "day" });
  });

  it("puts no dry spell yet in the Rain tab's stat, last rain under it", () => {
    renderRain(panel, rainLateYesterday());
    const stats = [...panel.querySelectorAll(".rain-dry-stats div")];
    const since = stats.find((s) => words(s.querySelector("dt")) === "Since it last rained")!;
    expect([...since.querySelectorAll("dd")].map(words)).toEqual([
      "No dry spell yet",
      "last rain 15 h ago",
    ]);
    expect(words(panel.querySelector("[data-section=dry]"))).not.toMatch(/\b0 (dry )?days\b/);
  });

  it("words a dry spell feature the same way", () => {
    const doc = v5Doc(0, 9);
    doc.feature = { kind: "dry_spell", reading: null, value: null, unit: null, percentile: null };
    const view = featureView(parseDashboard(doc))!;
    expect(view.value).toBeNull();
    expect(view.finding).toBe("No dry spell yet, last rain 9 h ago");
  });

  it("draws the widget as no dry spell yet, never 0 days", () => {
    const widget = createRequire(import.meta.url)("../../widgets/scriptable/Lookout.js") as {
      widgetModel(doc: unknown): { view: string; sentence: string; daysDry: number };
    };
    const doc = v5Doc(0, 9);
    doc.widget_view = "dry_spell";
    const model = widget.widgetModel(doc);
    expect(model.view).toBe("dry_spell");
    expect(model.sentence).toBe("No dry spell yet, last rain 9 h ago");
    const one = v5Doc(1, 40);
    one.widget_view = "dry_spell";
    expect(widget.widgetModel(one).sentence).toBe("Dry for 1 day, longer than the usual 2");
  });
});
