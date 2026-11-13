// @vitest-environment happy-dom
/**
 * The wind zoom (#476, spec #460; boards WindZoom, WindZoomDark): the wind
 * rose over the window a segmented control picks, the calm line, the share
 * by compass point and the live wind sentence, all read from the snapshot's
 * `wind_rose` (#465) and `current`. Nothing is computed in the browser: the
 * control only switches which snapshot window is drawn.
 */
import { afterEach, describe, expect, it } from "vitest";

import type { DashboardSnapshot } from "../src/snapshot.js";
import { SnapshotContractError, parseDashboard } from "../src/snapshot.js";
import { isZoomRoute, zoomFor } from "../src/overview/zoom.js";
import { ZOOMS } from "../src/overview/zooms/index.js";
import { windTile } from "../src/overview/mosaic-conditions.js";
import { windZoom } from "../src/overview/zooms/wind.js";
import type { Scenario } from "./golden.js";
import { golden } from "./golden.js";

const raw = (scenario: Scenario) =>
  structuredClone(golden(scenario, "dashboard")) as Record<string, unknown>;
const parsed = (scenario: Scenario): DashboardSnapshot => parseDashboard(raw(scenario));

const panel = (id: string) => windZoom.panels.find((p) => p.id === id)!;

/** Render one panel of the wind zoom into a fresh host in the page. */
function render(id: string, snapshot: DashboardSnapshot): HTMLElement {
  document.body.innerHTML = "";
  const host = document.createElement("section");
  document.body.append(host);
  panel(id).render(host, snapshot);
  return host;
}

const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();
const segments = (host: HTMLElement) => [
  ...host.querySelectorAll<HTMLButtonElement>(".wind-seg button"),
];
const pressed = (host: HTMLElement) =>
  segments(host).find((b) => b.getAttribute("aria-pressed") === "true");
const share = (host: HTMLElement, point: string) =>
  words(host.querySelector(`.wind-bars [data-point="${point}"] .wind-bar-value`));

afterEach(() => {
  document.body.innerHTML = "";
});

describe("the wind zoom's registration", () => {
  it("is registered under the wind tile's slug, after temperature", () => {
    // In the mosaic's order: after temperature, rain and dry spell (#475).
    const slugs = ZOOMS.map((z) => z.slug);
    expect(slugs.indexOf("wind")).toBeGreaterThan(slugs.indexOf("temperature"));
    expect(slugs.indexOf("wind")).toBeGreaterThan(slugs.indexOf("dry-spell"));
    expect(isZoomRoute("overview/wind")).toBe(true);
    expect(zoomFor("#overview/wind")?.title).toBe("Wind");
  });

  it("leads with the wind tile's reading", () => {
    expect(windZoom.lead!(parsed("seasoned"))).toEqual({
      value: "5.0",
      unit: "mph",
      delta: "from the WSW, gusts to 8.5 mph",
    });
  });
});

describe("the wind rose, seasoned station", () => {
  it("offers 24 h, 7 days and 30 days, 7 days pressed first", () => {
    const host = render("rose", parsed("seasoned"));
    expect(segments(host).map(words)).toEqual(["24 h", "7 days", "30 days"]);
    expect(words(pressed(host))).toBe("7 days");
    expect(host.querySelector(".wind-seg")!.getAttribute("role")).toBe("group");
  });

  it("draws a gust wedge and an average wedge for each direction with readings", () => {
    const host = render("rose", parsed("seasoned"));
    // 7 days: ESE through NW have readings, ten of the sixteen bins.
    expect(host.querySelectorAll(".wind-rose .wind-gust")).toHaveLength(10);
    expect(host.querySelectorAll(".wind-rose .wind-average")).toHaveLength(10);
    // A ring every 5 mph to the highest gust (11.5 mph): 5, 10 and 15.
    expect(host.querySelectorAll(".wind-rose .wind-ring")).toHaveLength(3);
  });

  it("marks now with one orange dot and its label", () => {
    const host = render("rose", parsed("seasoned"));
    expect(host.querySelectorAll(".wind-rose .wind-now-dot")).toHaveLength(1);
    expect(words(host.querySelector(".wind-rose .wind-now-label"))).toBe("now 5.0 mph");
  });

  it("names the calm share by what the snapshot counts", () => {
    const host = render("rose", parsed("seasoned"));
    expect(words(host.querySelector(".wind-calm"))).toBe(
      "Calm 24% of readings in the last 7 days: under 1 mph or no direction. " +
        "Calm readings have no direction, so the rose and the bars leave them out.",
    );
  });

  it("shares out the readings that were not calm by compass point", () => {
    const host = render("rose", parsed("seasoned"));
    expect(words(host.querySelector(".wind-bars-title"))).toBe(
      "Readings that were not calm, by direction",
    );
    expect(host.querySelectorAll(".wind-bars [data-point]")).toHaveLength(8);
    expect(share(host, "S")).toBe("25%");
    expect(share(host, "SW")).toBe("35%");
    expect(share(host, "N")).toBe("0%");
    // Each bar is drawn against 100%, so its length is its share.
    const fill = host.querySelector<HTMLElement>('.wind-bars [data-point="SW"] .wind-bar-fill')!;
    expect(fill.style.width).toBe("35%");
  });

  it("never implies a speed threshold such as windy", () => {
    const host = render("rose", parsed("seasoned"));
    expect(host.textContent).not.toMatch(/windy/i);
  });

  it("switches windows from the snapshot when a segment is pressed", () => {
    const host = render("rose", parsed("seasoned"));
    segments(host)[2]!.click();
    expect(words(pressed(host))).toBe("30 days");
    expect(words(host.querySelector(".wind-calm"))).toContain(
      "Calm 23% of readings in the last 30 days",
    );
    expect(share(host, "SW")).toBe("34%");
    // 30 days: E through NW, eleven bins.
    expect(host.querySelectorAll(".wind-rose .wind-gust")).toHaveLength(11);
    segments(host)[0]!.click();
    expect(words(host.querySelector(".wind-calm"))).toContain(
      "Calm 32% of readings in the last 24 hours",
    );
    expect(share(host, "SW")).toBe("52%");
    // The choice outlives a redraw; put it back for the tests after this one.
    segments(host)[1]!.click();
  });

  it("keeps the window pressed when the snapshot redraws", () => {
    const host = render("rose", parsed("seasoned"));
    segments(host)[2]!.click();
    const again = render("rose", parsed("seasoned"));
    expect(words(pressed(again))).toBe("30 days");
    segments(again)[1]!.click();
  });

  it("gives screen readers a table in place of the rose, which it hides", () => {
    const host = render("rose", parsed("seasoned"));
    expect(host.querySelector(".wind-rose svg")!.getAttribute("aria-hidden")).toBe("true");
    const table = host.querySelector(".wind-sr table")!;
    expect(words(table.querySelector("caption"))).toBe(
      "Average wind and highest gust from each direction, last 7 days",
    );
    expect(words(host.querySelector(".wind-sr > p"))).toBe(
      "Wind rose for the last 7 days: each direction's average wind and highest gust, " +
        "listed in the table that follows.",
    );
    const rows = [...table.querySelectorAll("tbody tr")].map((r) =>
      [...r.children].map((cell) => cell.textContent).join(" "),
    );
    expect(rows).toHaveLength(16);
    expect(rows[0]).toBe("N no readings");
    expect(rows[10]).toBe("SW 4.5 mph 10.6 mph");
    // The bars' graphics are hidden; their words are read.
    for (const bar of host.querySelectorAll(".wind-bar-fill")) {
      expect(bar.getAttribute("aria-hidden")).toBe("true");
    }
  });
});

describe("the wind rose, young and live stations", () => {
  it("offers only the windows the record covers, and no control for one", () => {
    const host = render("rose", parsed("raining"));
    expect(segments(host)).toHaveLength(0);
    expect(words(host.querySelector(".wind-calm"))).toContain(
      "Calm 0% of readings in the last 24 hours",
    );
    expect(share(host, "S")).toBe("100%");
    expect(host.querySelectorAll(".wind-rose .wind-gust")).toHaveLength(1);
  });

  it("hatches a window the record does not hold, with its reason, and draws no rose", () => {
    const host = render("rose", parsed("dry"));
    expect(host.querySelector(".wind-rose svg")).toBeNull();
    expect(host.querySelector(".wind-bars")).toBeNull();
    expect(host.querySelector(".gap-hatch, [data-gap]")).not.toBeNull();
    expect(host.textContent).toContain("not archived yet");
  });

  it("hides the rose for a snapshot without one, and still shows the live wind", () => {
    const doc = raw("seasoned");
    delete doc.wind_rose;
    const snapshot = parseDashboard(doc);
    expect(panel("rose").shown!(snapshot)).toBe(false);
    const nowPanel = panel("now");
    expect(nowPanel.shown?.(snapshot) ?? true).toBe(true);
    expect(words(render("now", snapshot))).toBe("From the WSW, gusts to 8.5 mph.");
  });

  it("draws no bars for a window with no readings, which the calm line says", () => {
    const snapshot = parsed("raining");
    const w = snapshot.wind_rose!.windows[0]!;
    Object.assign(w, { readings: 0, calm_readings: 0, calm_pct: null });
    for (const p of w.points) Object.assign(p, { readings: 0, share_pct: null });
    const host = render("rose", snapshot);
    expect(words(host.querySelector(".wind-calm"))).toBe("No wind readings in the last 24 hours.");
    expect(host.querySelector(".wind-bars")).toBeNull();
  });

  it("refuses a malformed rose", () => {
    const doc = raw("seasoned");
    doc.wind_rose = { calm_below_mph: 1, windows: [{ key: "24h" }] };
    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });
});

describe("the live wind sentence", () => {
  it("says where it blows from and its gust", () => {
    expect(words(render("now", parsed("dry")))).toBe("From the SE, gusts to 2.5 mph.");
  });

  it("says calm at a speed of exactly 0, with no direction (#598)", () => {
    const snapshot = parsed("dry");
    Object.assign(snapshot.current, { wind_speed_mph: 0, wind_gust_mph: 0 });
    expect(snapshot.current.wind_dir_deg).not.toBeNull();
    expect(words(render("now", snapshot))).toBe("Calm.");
    expect(windZoom.lead!(snapshot)).toEqual({ value: "calm", delta: "no gusts" });
    snapshot.current.wind_gust_mph = 2.5;
    expect(words(render("now", snapshot))).toBe("Calm; gusts to 2.5 mph.");
    snapshot.current.wind_gust_mph = null;
    expect(words(render("now", snapshot))).toBe("Calm; gust not known.");
  });

  it("draws no now dot on the rose at 0 mph, and does at a non-zero speed (#598)", () => {
    const snapshot = parsed("seasoned");
    snapshot.current.wind_speed_mph = 0;
    expect(render("rose", snapshot).querySelectorAll(".wind-rose .wind-now-dot")).toHaveLength(0);
    expect(render("rose", snapshot).textContent).not.toContain("now 0.0");
    snapshot.current.wind_speed_mph = 1.2;
    expect(render("rose", snapshot).querySelectorAll(".wind-rose .wind-now-dot")).toHaveLength(1);
  });

  it("says what is not known, never zero", () => {
    const snapshot = parsed("dry");
    snapshot.current.wind_speed_mph = null;
    snapshot.current.wind_dir_deg = null;
    snapshot.current.wind_gust_mph = null;
    expect(words(render("now", snapshot))).toBe(
      "Wind speed not known. Direction and gust not known.",
    );
  });
});

describe("the Wind tile's line in the sources footer", () => {
  it("describes the rose and its calm threshold, from the snapshot", () => {
    const snapshot = parsed("seasoned");
    snapshot.wind_rose!.calm_below_mph = 0.5;
    const { line, method } = windTile.source(snapshot);
    expect(line).toBe("live speed, direction and gust; the rose from each window's readings.");
    expect(method).toContain("each direction's average wind and its highest gust");
    expect(method).toContain("A reading under 0.5 mph, or with no direction, is calm");
  });

  it("describes the live reading alone before the rose (N-1)", () => {
    const snapshot = parsed("seasoned");
    delete snapshot.wind_rose;
    expect(windTile.source(snapshot).line).toBe("live speed, direction and gust.");
    expect(windTile.source(snapshot).method).not.toContain("rose");
  });
});
