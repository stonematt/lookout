// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";

import { renderDump } from "../src/debug.js";

describe("renderDump", () => {
  it("shows the URL and status, then the body as indented JSON", () => {
    const body = JSON.stringify({ kind: "current", last_data: { dateutc: 1, tempf: 59.5 } });
    const [meta, pre] = renderDump("/current?station=X", 200, body);
    expect(meta!.textContent).toBe("GET /current?station=X → 200");
    expect(pre!.textContent).toBe(JSON.stringify(JSON.parse(body), null, 2));
  });

  it("puts values in as text, never as markup", () => {
    const [, pre] = renderDump("/current", 200, JSON.stringify({ note: "<img src=x>" }));
    expect(pre!.querySelector("img")).toBeNull();
    expect(pre!.textContent).toContain("<img src=x>");
  });

  it("shows a body that is not JSON as it came", () => {
    const [meta, pre] = renderDump("/current", 502, "Bad Gateway");
    expect(meta!.textContent).toContain("502");
    expect(pre!.textContent).toBe("Bad Gateway");
  });
});
