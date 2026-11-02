/**
 * The security headers `vercel.json` sets on every response. The page is
 * checked against the same policy in a browser by `scripts/csp-check.mjs`.
 */
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

interface HeaderRule {
  source: string;
  headers: { key: string; value: string }[];
}

const config = JSON.parse(
  readFileSync(new URL("../vercel.json", import.meta.url), "utf8"),
) as { headers: HeaderRule[] };

/** The headers the catch-all rule sets, by lower-case name. */
function everyResponse(): Map<string, string> {
  const rule = config.headers.find((r) => r.source === "/(.*)");
  expect(rule, "a header rule for every path").toBeDefined();
  return new Map(rule!.headers.map((h) => [h.key.toLowerCase(), h.value]));
}

/** The CSP as directive name to its sources. */
function csp(): Map<string, string[]> {
  const value = everyResponse().get("content-security-policy") ?? "";
  return new Map(
    value
      .split(";")
      .map((d) => d.trim().split(/\s+/))
      .filter((parts) => parts[0])
      .map(([name, ...sources]) => [name!, sources]),
  );
}

describe("the security headers", () => {
  it("stop MIME sniffing and keep the referrer to this origin", () => {
    const headers = everyResponse();
    expect(headers.get("x-content-type-options")).toBe("nosniff");
    expect(headers.get("referrer-policy")).toBe("strict-origin-when-cross-origin");
  });

  it("forbid framing the site", () => {
    expect(csp().get("frame-ancestors")).toEqual(["'none'"]);
    expect(everyResponse().get("x-frame-options")).toBe("DENY");
  });

  it("allow scripts and connections to this origin only", () => {
    const policy = csp();
    expect(policy.get("default-src")).toEqual(["'self'"]);
    expect(policy.get("script-src")).toEqual(["'self'"]);
    expect(policy.get("connect-src")).toEqual(["'self'"]);
    expect(policy.get("object-src")).toEqual(["'none'"]);
    expect(policy.get("base-uri")).toEqual(["'none'"]);
  });

  it("allow inline styles only, which Plotly and the SVG charts set", () => {
    const policy = csp();
    expect(policy.get("style-src")).toEqual(["'self'", "'unsafe-inline'"]);
    expect(policy.get("img-src")).toEqual(["'self'"]);
    expect(policy.get("font-src")).toEqual(["'self'"]);
  });

  it("never allow inline or evaluated script", () => {
    const value = everyResponse().get("content-security-policy")!;
    for (const [name, sources] of csp()) {
      if (name.startsWith("script-src") || name === "default-src") {
        expect(sources, name).not.toContain("'unsafe-inline'");
        expect(sources, name).not.toContain("'unsafe-eval'");
      }
    }
    expect(value).not.toContain("*");
  });
});
