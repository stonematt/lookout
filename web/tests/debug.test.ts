// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";

import { renderDump, renderTree } from "../src/debug.js";

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
