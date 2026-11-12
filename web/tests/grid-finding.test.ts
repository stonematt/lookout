/**
 * The snapshot's `almanac.grid_finding` (#467) names the month `gridHeadline`
 * leads with and the earlier months it sets it against: the browser's rule,
 * now decided in Python. Each golden's key must reproduce the sentence the
 * site builds today.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { gridHeadline } from "../src/overview/almanac-text.js";
import type { Almanac } from "../src/snapshot.js";
import { goldenText, SCENARIOS } from "./golden.js";

type Finding = { month: string; earlier: string[] } | null;
type WithFinding = Almanac & { grid_finding: Finding };

const ROOT = resolve(import.meta.dirname, "../..");
const october: WithFinding = JSON.parse(
  readFileSync(`${ROOT}/tests/golden/almanac/october.json`, "utf8"),
).almanac;

const cases: [string, WithFinding][] = [
  ["october", october],
  ...SCENARIOS.map((name): [string, WithFinding] => [
    name,
    JSON.parse(goldenText(name, "dashboard")).almanac,
  ]),
];

describe("almanac.grid_finding", () => {
  it.each(cases)("%s: names the month and years gridHeadline leads with", (_name, almanac) => {
    const headline = gridHeadline(almanac);
    const finding = almanac.grid_finding;
    if (finding === null) {
      expect(headline).toBe("Rain by month, each water year from Oct 1");
      return;
    }
    expect(headline).toContain(`${finding.month.slice(0, 4)} had`);
    const named = [...headline.matchAll(/\((\d{4})\)/g)].map((m) => m[1]);
    expect(finding.earlier.map((key) => key.slice(0, 4))).toEqual(named);
    for (const key of finding.earlier) expect(key.slice(5)).toBe(finding.month.slice(5));
  });
});
