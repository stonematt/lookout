// @vitest-environment happy-dom
/**
 * The year-by-month heatmap's month under way (#727): shaded from its
 * projection with the outline, labelled with the total so far, "≈" on the
 * days-elapsed fallback, hatched when the projection is null or missing, and
 * never moving the scale.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { drawYearMonth } from "../src/solar/charts.js";
import type { SolarMonth, SolarTab } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const base = parseDashboard(golden("raining", "dashboard")).solar_tab!;

let host: HTMLElement;
let said: string[];

beforeEach(() => {
  document.body.innerHTML = "<div id=host></div>";
  host = document.getElementById("host")!;
  said = [];
});

/** The tab with its month under way given these fields. */
function withCurrent(fields: Partial<SolarMonth>): { tab: SolarTab; key: string } {
  const months = base.months.map((m) =>
    m.status.reason === "in_progress" ? { ...m, ...fields } : m,
  );
  const key = months.find((m) => m.status.reason === "in_progress")!.key;
  return { tab: { ...base, months }, key };
}

/** Draw and return the cells with their text. */
function draw(tab: SolarTab): SVGRectElement[] {
  drawYearMonth(host, tab, 700, (t) => said.push(t));
  return [...host.querySelectorAll<SVGRectElement>("rect.solar-current")];
}

const cellTexts = () => [...host.querySelectorAll(".solar-cell-partial")].map((t) => t.textContent);

describe("the month under way on the Months heatmap", () => {
  it("is shaded from its projection, outlined, and labelled with the total so far", () => {
    const { tab } = withCurrent({
      to_date_kwh_m2: 24.4,
      projected_kwh_m2: 30,
      projection: "past_years",
    });
    const [cell] = draw(tab);
    expect(draw(tab)).toHaveLength(1);
    expect(cell!.getAttribute("class")).toMatch(/solar-cell solar-step-\d solar-current/);
    expect(cell!.getAttribute("fill")).toBeNull();
    expect(cellTexts()).toEqual(["24+"]);
    expect(cell!.querySelector("title")!.textContent).toMatch(
      /so far: 24\.4 kWh\/m², on course for about 30\.0 kWh\/m²\./,
    );
  });

  it("marks the days-elapsed fallback with ≈", () => {
    const { tab } = withCurrent({
      to_date_kwh_m2: 24.4,
      projected_kwh_m2: 30,
      projection: "days_elapsed",
    });
    draw(tab);
    expect(cellTexts()).toEqual(["≈24+"]);
  });

  it("clamps a projection above the scale to the top shade, leaving the scale alone", () => {
    const { tab } = withCurrent({
      to_date_kwh_m2: 5,
      projected_kwh_m2: 500,
      projection: "past_years",
    });
    const [cell] = draw(tab);
    expect(cell!.getAttribute("class")).toContain("solar-step-5");
    // The legend's top step comes from finished months only (index.ts), so a
    // projection cannot move it; here, the cell is capped at step 5, not above.
    expect(cell!.getAttribute("class")).not.toContain("solar-step-6");
  });

  it("stays hatched with the outline when the projection is null", () => {
    const { tab } = withCurrent({
      to_date_kwh_m2: 0.3,
      projected_kwh_m2: null,
      projection: null,
    });
    const [cell] = draw(tab);
    expect(cell!.getAttribute("fill")).toMatch(/^url\(/);
    expect(cellTexts()).toEqual([]);
  });

  it("stays hatched on an older snapshot without the fields", () => {
    const { tab } = withCurrent({});
    const months = tab.months.map((m) => {
      const { to_date_kwh_m2: _a, projected_kwh_m2: _b, projection: _c, ...old } = m;
      return old as SolarMonth;
    });
    const [cell] = draw({ ...tab, months });
    expect(cell!.getAttribute("fill")).toMatch(/^url\(/);
  });
});
