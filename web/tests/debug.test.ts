// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";

import { renderDump, renderRelease, renderTree, stationPicker } from "../src/debug.js";
import { SITE_RELEASE } from "../src/release.js";
import { FIXTURE_MAC, SALMON_MAC, oneStation, twoStations } from "./golden.js";

const BODY = {
  schema_version: 3,
  kind: "current",
  station: { mac: "00:11:22:33:44:55", name: "Fixture Station" },
  observed_at_ms: 1767918300000,
  last_data: { dateutc: 1767918300000, tempf: 59.5, battrain: "1", note: "<img src=x>" },
  context: { station: { tz: "America/Los_Angeles" }, current: null },
};

const kickers = (grid: HTMLElement) =>
  [...grid.querySelectorAll(".tile-kicker")].map((k) => k.textContent);

describe("renderDump", () => {
  it("names the station and the read in the header", () => {
    const [header] = renderDump("/current", 200, JSON.stringify(BODY));
    expect(header!.querySelector("h1")!.textContent).toBe("Fixture Station");
    expect(header!.textContent).toContain("GET /current · 200");
  });

  it("puts lastData first, then context, the document's fields, and the raw body", () => {
    const [, grid] = renderDump("/current", 200, JSON.stringify(BODY));
    expect(kickers(grid!)).toEqual(["last_data", "context", "station", "document", "raw JSON"]);
  });

  it("keeps the whole body as received, indented", () => {
    const [, grid] = renderDump("/current", 200, JSON.stringify(BODY));
    expect(JSON.parse(grid!.querySelector("pre")!.textContent!)).toEqual(BODY);
  });

  it("puts values in as text, never as markup", () => {
    const [, grid] = renderDump("/current", 200, JSON.stringify(BODY));
    expect(grid!.querySelector("img")).toBeNull();
    expect(grid!.textContent).toContain("<img src=x>");
  });

  it("shows a body that is not JSON as it came, and marks the failed read", () => {
    const [header, grid] = renderDump("/current", 502, "Bad Gateway");
    expect(header!.querySelector<HTMLElement>(".debug-request")!.dataset.ok).toBe("false");
    expect(kickers(grid!)).toEqual(["raw JSON"]);
    expect(grid!.querySelector("pre")!.textContent).toBe("Bad Gateway");
  });
});

describe("stationPicker", () => {
  it("lists every indexed station with the shown one selected", () => {
    const wrap = stationPicker(twoStations(), SALMON_MAC, () => undefined)!;
    const select = wrap.querySelector("select")!;
    expect(wrap.className).toBe("masthead-picks");
    expect([...select.options].map((o) => o.value)).toEqual([FIXTURE_MAC, SALMON_MAC]);
    expect(select.value).toBe(SALMON_MAC);
  });

  it("hands the chosen MAC on", () => {
    const chosen: string[] = [];
    const select = stationPicker(twoStations(), FIXTURE_MAC, (mac) =>
      chosen.push(mac),
    )!.querySelector("select")!;
    select.value = SALMON_MAC;
    select.dispatchEvent(new Event("change"));
    expect(chosen).toEqual([SALMON_MAC]);
  });

  it("draws nothing for one station", () => {
    expect(stationPicker(oneStation(), FIXTURE_MAC, () => undefined)).toBeNull();
  });

  it("takes the station name's place in the header", () => {
    const [header] = renderDump("/current", 200, JSON.stringify(BODY), twoStations());
    const select = header!.querySelector("h1 select")!;
    expect((select as HTMLSelectElement).value).toBe(FIXTURE_MAC);
  });
});

describe("renderRelease", () => {
  const SITE = { version: "2026.11.12", sha: "93e95b8" };
  const status = (version: string, sha: string | null) =>
    JSON.stringify({ kind: "status", version, sha, outcome: "ok" });
  const spans = (line: HTMLElement) => [...line.children].map((s) => s.textContent);

  it("sets the site's version and SHA beside the refresh run's", () => {
    const line = renderRelease(status("2026.11.12", "93e95b8a0c1d2e3f4a5b"), SITE);
    expect(spans(line)).toEqual(["Site 2026.11.12 · 93e95b8", "Refresh run 2026.11.12 · 93e95b8"]);
    expect(line.dataset.match).toBe("true");
  });

  it("says the two differ when the versions do", () => {
    const line = renderRelease(status("2026.11.1", "93e95b8a0c1d2e3f4a5b"), SITE);
    expect(spans(line)).toEqual([
      "Site 2026.11.12 · 93e95b8",
      "Refresh run 2026.11.1 · 93e95b8",
      "differ",
    ]);
    expect(line.dataset.match).toBe("false");
  });

  it("says the two differ when the SHAs do", () => {
    const line = renderRelease(status("2026.11.12", "0123456789abcdef"), SITE);
    expect(line.dataset.match).toBe("false");
  });

  it("compares versions only for a local build or a run that cannot tell its SHA", () => {
    const local = renderRelease(status("2026.11.12", "0123456789abcdef"), {
      ...SITE,
      sha: "local",
    });
    expect(local.dataset.match).toBe("true");
    const noSha = renderRelease(status("2026.11.12", null), SITE);
    expect(spans(noSha)[1]).toBe("Refresh run 2026.11.12 · unknown");
    expect(noSha.dataset.match).toBe("true");
  });

  it("still shows the site's release when the status cannot be read", () => {
    for (const text of [null, "Bad Gateway", JSON.stringify({ error: "x" })]) {
      const line = renderRelease(text, SITE);
      expect(spans(line)).toEqual(["Site 2026.11.12 · 93e95b8", "Refresh run unknown"]);
      expect(line.dataset.match).toBe("unknown");
    }
  });

  it("shows this build's release by default", () => {
    expect(renderRelease(null).textContent).toBe(
      `Site ${SITE_RELEASE.version} · local` + "Refresh run unknown",
    );
  });
});

describe("renderTree", () => {
  it("writes each value as JSON does, typed, with epoch-ms given a local time", () => {
    const tree = renderTree(BODY.last_data, "America/Los_Angeles");
    const values = [...tree.querySelectorAll("dd")];
    expect(values.map((v) => v.className)).toEqual([
      "json-value json-number",
      "json-value json-number",
      "json-value json-string",
      "json-value json-string",
    ]);
    // a string stays quoted, so "1" never reads as the number 1
    expect(values[2]!.textContent).toBe('"1"');
    expect(values[0]!.querySelector(".json-time")!.textContent).toContain("PST");
  });

  it("folds nested blocks open, with their size", () => {
    const tree = renderTree({ station: { mac: "x", name: null } });
    const fold = tree.querySelector("details")!;
    expect(fold.open).toBe(true);
    expect(fold.querySelector("summary")!.textContent).toBe("station{2}");
  });
});
