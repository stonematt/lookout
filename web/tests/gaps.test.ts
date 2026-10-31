// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";

import { gapElement, gapLabel, gapSentence, gapValue, hatchGap } from "../src/gaps.js";
import type { GapStatus } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const raining = parseDashboard(golden("raining", "dashboard"));
const dry = parseDashboard(golden("dry", "dashboard"));
const window = (doc: typeof raining, key: string): GapStatus =>
  doc.gaps.windows.find((w) => w.key === key)!;

const status = (over: Partial<GapStatus>): GapStatus => ({
  key: "30d",
  first_day: "2025-12-13",
  last_day: "2026-01-11",
  days: 30,
  missing_days: 0,
  status: "valid",
  reason: null,
  ...over,
});
const gapInRecord = status({ status: "invalid", reason: "gap", missing_days: 4 });
const prorated = status({ status: "prorated", reason: "gap", missing_days: 3 });

describe("the gap statuses in the snapshot", () => {
  it("carry every standard window, in order", () => {
    expect(raining.gaps.windows.map((w) => w.key)).toEqual([
      "today",
      "yesterday",
      "7d",
      "30d",
      "90d",
      "365d",
      "water_year",
    ]);
  });

  it("read the raining station's week as before the archive began", () => {
    expect(window(raining, "today").status).toBe("valid");
    expect(window(raining, "7d")).toMatchObject({
      status: "invalid",
      reason: "before_archive",
    });
  });

  it("read the dry station's today as not archived yet", () => {
    expect(window(dry, "today")).toMatchObject({
      status: "invalid",
      reason: "not_archived",
    });
    expect(gapLabel(window(dry, "today"))).toBe("not archived yet");
  });
});

describe("gap labels", () => {
  it("say why in place, short", () => {
    expect(gapLabel(gapInRecord)).toBe("n/a: gap in record");
    expect(gapLabel(status({ status: "invalid", reason: "not_archived" }))).toBe(
      "not archived yet",
    );
    expect(gapLabel(status({ status: "invalid", reason: "low_coverage" }))).toBe(
      "n/a: low coverage",
    );
  });

  it("are null for a window that can be compared", () => {
    expect(gapLabel(status({}))).toBeNull();
    expect(gapLabel(prorated)).toBeNull();
  });

  it("have a full sentence naming the days", () => {
    expect(gapSentence(gapInRecord)).toBe(
      "Dec 13, 2025 to Jan 11, 2026: 4 of 30 days are missing from the record, " +
        "so it is not compared.",
    );
    expect(gapSentence(prorated)).toBe(
      "Dec 13, 2025 to Jan 11, 2026: 3 of 30 days are missing from the record, " +
        "so its total is scaled up from the other 27 (≈).",
    );
    expect(
      gapSentence(
        status({
          first_day: "2026-01-08",
          last_day: "2026-01-08",
          days: 1,
          missing_days: 1,
          status: "invalid",
          reason: "not_archived",
        }),
      ),
    ).toBe("Jan 8, 2026: the archive does not reach this far yet.");
  });
});

describe("gap values", () => {
  it("are never shown for an invalid window", () => {
    expect(gapValue(gapInRecord, "3.2 in")).toBeNull();
  });

  it("mark a prorated value with ≈ and leave a valid one alone", () => {
    expect(gapValue(prorated, "3.2 in")).toBe("≈ 3.2 in");
    expect(gapValue(status({}), "3.2 in")).toBe("3.2 in");
  });
});

describe("hatching", () => {
  it("replaces an invalid cell's content with the hatch and the reason", () => {
    const cell = document.createElement("td");
    cell.textContent = "3.2 in";

    expect(hatchGap(cell, gapInRecord)).toBe(true);

    expect(cell.classList.contains("gap-hatch")).toBe(true);
    expect(cell.dataset.gap).toBe("gap");
    expect(cell.textContent).not.toContain("3.2");
    expect(cell.querySelector(".gap-label")?.textContent).toBe("n/a: gap in record");
    expect(cell.querySelector(".gap-sr")?.textContent?.trim()).toBe(
      gapSentence(gapInRecord),
    );
    expect(cell.title).toBe(gapSentence(gapInRecord));
  });

  it("reads just n/a in a compact cell, with the reason kept for screen readers", () => {
    const cell = document.createElement("td");

    hatchGap(cell, gapInRecord, { compact: true });

    expect(cell.querySelector(".gap-label")?.textContent).toBe("n/a");
    expect(cell.querySelector(".gap-label")?.getAttribute("aria-hidden")).toBe("true");
    expect(cell.querySelector(".gap-sr")?.textContent?.trim()).toBe(
      gapSentence(gapInRecord),
    );
  });

  it("leaves a comparable cell's content, marking a prorated one", () => {
    const valid = document.createElement("td");
    valid.textContent = "3.2 in";
    const scaled = document.createElement("td");
    scaled.textContent = "≈ 3.2 in";

    expect(hatchGap(valid, status({}))).toBe(false);
    expect(hatchGap(scaled, prorated)).toBe(false);

    expect(valid.textContent).toBe("3.2 in");
    expect(valid.classList.contains("gap-hatch")).toBe(false);
    expect(scaled.dataset.gap).toBe("prorated");
    expect(scaled.title).toBe(gapSentence(prorated));
    expect(scaled.querySelector(".gap-sr")?.textContent?.trim()).toBe(
      gapSentence(prorated),
    );
    expect(valid.querySelector(".gap-sr")).toBeNull();
  });

  it("re-hatches an element without stacking its spoken text", () => {
    const cell = document.createElement("td");
    cell.textContent = "≈ 3.2 in";

    hatchGap(cell, prorated);
    hatchGap(cell, prorated);

    expect(cell.querySelectorAll(".gap-sr")).toHaveLength(1);
  });

  it("lifts the hatch when a status becomes comparable", () => {
    const cell = document.createElement("td");
    hatchGap(cell, gapInRecord);

    expect(hatchGap(cell, status({}))).toBe(false);

    expect(cell.classList.contains("gap-hatch")).toBe(false);
    expect(cell.title).toBe("");
    expect(cell.textContent).toBe("");
    expect(cell.dataset.gap).toBe("valid");
  });

  it("builds a standalone hatched block for a chart row", () => {
    const block = gapElement(window(dry, "7d"));

    expect(block.tagName).toBe("SPAN");
    expect(block.classList.contains("gap-hatch")).toBe(true);
    expect(block.querySelector(".gap-label")?.textContent).toBe("not archived yet");
  });
});
