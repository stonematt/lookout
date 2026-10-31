// @vitest-environment happy-dom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import { heroLead, leadHeadline } from "../src/overview/hero.js";
import { renderOverview, SECTIONS } from "../src/overview/index.js";
import type { DashboardSnapshot, HeroLead, LeadKind } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

const raining = parseDashboard(golden("raining", "dashboard"));
const dry = parseDashboard(golden("dry", "dashboard"));

/**
 * The dry dashboard with one lead's golden section in place of its own, as
 * `tests/test_snapshot_lead.py` writes it to `tests/golden/leads/`.
 */
function withLead(kind: LeadKind): DashboardSnapshot {
  const path = resolve(import.meta.dirname, `../../tests/golden/leads/${kind}.json`);
  const lead = JSON.parse(readFileSync(path, "utf8")) as HeroLead;
  return {
    ...dry,
    observed_at_ms: lead.now_ms,
    archive_newest_ms: lead.now_ms === null ? null : lead.now_ms - 5 * 60_000,
    active_rain_event: null,
    hero_lead: lead,
  };
}

let host: HTMLElement;

beforeEach(() => {
  document.body.innerHTML = "<section id=host></section>";
  host = document.getElementById("host")!;
});

const text = (selector: string) =>
  host.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim();
const rules = () =>
  [...host.querySelectorAll<HTMLElement>("[data-part=why] li")].map((li) => ({
    rule: li.dataset.rule,
    met: li.dataset.met,
    lead: li.dataset.lead === "true",
    text: li.textContent?.replace(/\s+/g, " ").trim(),
  }));

describe("the hero lead's place", () => {
  it("follows the first screen", () => {
    expect(SECTIONS.map((s) => s.id).slice(0, 2)).toEqual(["first-screen", "hero-lead"]);
  });

  it("renders after the first screen in the Overview", () => {
    document.body.innerHTML = "<div id=panel></div>";
    const panel = document.getElementById("panel")!;
    renderOverview(panel, raining);

    const sections = [...panel.children].map((el) => el.getAttribute("data-section"));
    expect(sections.slice(0, 2)).toEqual(["first-screen", "hero-lead"]);
  });
});

describe("the headline names the finding with its number", () => {
  it("rain, while the event is live", () => {
    heroLead.render(host, raining);

    expect(text("[data-part=headline]")).toBe(
      "Raining now: 0.30 in so far, against 0.42 in for a usual event",
    );
    expect(host.dataset.lead).toBe("rain");
  });

  it("temperature, naming the compared years", () => {
    expect(leadHeadline(withLead("temperature"))).toBe(
      "10.0° warmer than usual for noon in January (2025)",
    );
  });

  it("solar", () => {
    expect(leadHeadline(withLead("solar"))).toBe("Strong sun: 400 W/m² now");
  });

  it("calm, against the usual dry spell", () => {
    expect(leadHeadline(withLead("calm"))).toBe(
      "2 d 12 h since it last rained, against a usual dry spell of 4 days",
    );
  });

  it("calm, without a usual dry spell when the year is not comparable", () => {
    expect(leadHeadline(dry)).toBe("1 d 14 h since it last rained");
  });
});

describe("Why this leads", () => {
  it("lists every rule, in order, with its live value and whether it was met", () => {
    heroLead.render(host, raining);

    expect(text("[data-part=why] summary")).toBe("Why this leads");
    expect(rules()).toEqual([
      {
        rule: "rain",
        met: "true",
        lead: true,
        text: "Rain event live or ended in the last 12 h: raining now, met: this leads",
      },
      {
        rule: "temperature",
        met: "false",
        lead: false,
        text:
          "Temperature at least 10° from the January median at 4 pm: not compared, " +
          "as no earlier January is complete in the record, not met",
      },
      {
        rule: "solar",
        met: "false",
        lead: false,
        text: "Sun at least 400 W/m²: 322 W/m² now, not met",
      },
      {
        rule: "calm",
        met: "false",
        lead: false,
        text:
          "Otherwise, calm: under 1 m since it last rained; the usual dry spell " +
          "is not compared, not met",
      },
    ]);
  });

  it("gives the temperature rule its median and years", () => {
    heroLead.render(host, withLead("temperature"));

    const temperature = rules().find((r) => r.rule === "temperature")!;
    expect(temperature.text).toBe(
      "Temperature at least 10° from the January median at noon: +10.0° now, " +
        "against 50.0° (2025), met: this leads",
    );
    const rain = rules().find((r) => r.rule === "rain")!;
    expect(rain.text).toBe(
      "Rain event live or ended in the last 12 h: the last event ended 1 d 17 h ago, not met",
    );
  });

  it("gives calm its usual dry spell when it leads", () => {
    heroLead.render(host, withLead("calm"));

    expect(rules().at(-1)).toEqual({
      rule: "calm",
      met: "true",
      lead: true,
      text:
        "Otherwise, calm: 2 d 12 h since it last rained, against a usual dry " +
        "spell of 4 days, met: this leads",
    });
    expect(rules().find((r) => r.rule === "rain")!.text).toBe(
      "Rain event live or ended in the last 12 h: no rain event on record, not met",
    );
  });
});

describe("the hero chart", () => {
  it("draws the bold now over the usual band, named in words", () => {
    heroLead.render(host, raining);

    const chart = host.querySelector("[data-part=chart] svg")!;
    expect(chart.getAttribute("role")).toBe("img");
    expect(chart.getAttribute("aria-label")).toContain("Raining now");
    expect(chart.querySelector(".hero-band")).not.toBeNull();
    expect(chart.querySelector(".hero-median")).not.toBeNull();
    expect(chart.querySelector(".hero-line")).not.toBeNull();
    expect(chart.querySelector(".hero-dot")).not.toBeNull();
    expect(chart.querySelector(".hero-note")?.textContent).toBe("now 0.30 in");
  });

  it("hatches the span only the live reading covers", () => {
    heroLead.render(host, raining);

    expect(host.querySelector(".hero-hatch")).not.toBeNull();
    expect(host.querySelector(".hero-join")).not.toBeNull();
    expect(text("[data-part=source]")).toContain("only the live reading is known");
  });

  it("draws temperature over each hour's band", () => {
    heroLead.render(host, withLead("temperature"));

    expect(host.querySelector(".hero-line-warm")).not.toBeNull();
    expect(host.querySelector(".hero-band")?.getAttribute("d")).toMatch(/^M/);
    expect(host.querySelector(".hero-hatch")).toBeNull();
  });

  it("draws the dry spell as a bar against the usual spells", () => {
    heroLead.render(host, withLead("calm"));

    expect(host.querySelector(".hero-line-calm")).not.toBeNull();
    expect(host.querySelector(".hero-band")).not.toBeNull();
    const facts = [...host.querySelectorAll("[data-part=facts] div")].map((item) => [
      item.querySelector("dd")?.textContent,
      item.querySelector("dt")?.textContent,
    ]);
    expect(facts).toEqual([
      ["2 d 12 h", "since rain"],
      ["4 days", "usual dry spell"],
      ["72", "dry spells compared"],
    ]);
  });

  it("says why calm has no usual dry spell, in the source line", () => {
    heroLead.render(host, dry);

    expect(host.querySelector(".hero-band")).toBeNull();
    // The 365-day window's own gap sentence.
    expect(text("[data-part=source]")).toContain(
      "Nov 20, 2024 to Nov 19, 2025: the archive does not reach this far yet.",
    );
  });
});
