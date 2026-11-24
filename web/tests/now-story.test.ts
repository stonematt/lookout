// @vitest-environment happy-dom
/**
 * One rain moment, one wording (#694 item 1): the Overview's rain tile and
 * the Rain tab head both tell the now story from `rain_now` through one
 * helper (`now-story.ts`), so for one snapshot they never disagree. The
 * format is the boards': minutes under an hour, whole hours after ("Rain
 * stopped 40 min ago", "Last rain 9 h ago"); the rate in "in/hr" (#647
 * call 1).
 */
import { createRequire } from "node:module";

import { beforeEach, describe, expect, it } from "vitest";

import { setLayout } from "../src/layout.js";
import { agoWords } from "../src/now-story.js";
import { renderOverview } from "../src/overview/index.js";
import { renderRain } from "../src/rain/index.js";
import type { DashboardSnapshot } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { golden } from "./golden.js";

type Doc = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

/** The raining golden, rain stopped `minutes` before the build. */
function stopped(minutes: number): DashboardSnapshot {
  const d = structuredClone(golden("raining", "dashboard")) as Doc;
  d.rain_now = {
    ...d.rain_now,
    raining: false,
    last_rain_ms: d.built_at_ms - minutes * 60_000,
    rate_in_hr: 0.03,
  };
  return parseDashboard(d);
}

const raining = () => parseDashboard(golden("raining", "dashboard"));

let panel: HTMLElement;

beforeEach(() => {
  setLayout("phone");
  document.body.innerHTML = "<div id=panel></div>";
  panel = document.getElementById("panel")!;
  history.replaceState(null, "", "/");
});

const words = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim();

/** The rain tile's story on the Overview. */
function overviewStory(snapshot: DashboardSnapshot): string | undefined {
  renderOverview(panel, snapshot);
  const tile = panel.querySelector(".tile[data-section=rain]:not([data-mosaic])");
  return words(tile?.querySelector("a[data-part=now] [data-part=finding]"));
}

/** The Rain tab head's now line. */
function rainTabStory(snapshot: DashboardSnapshot): string | undefined {
  panel.innerHTML = "";
  renderRain(panel, snapshot);
  return words(panel.querySelector("[data-section=years] [data-part=finding]"));
}

describe("the now story on the Overview and the Rain tab", () => {
  it.each([40, 130, 170])("says the same thing %i minutes after rain stopped", (minutes) => {
    const snapshot = stopped(minutes);
    const tab = rainTabStory(snapshot);
    expect(overviewStory(snapshot)).toBe(tab);
  });

  it("words a stop over an hour ago in whole hours, as the boards do", () => {
    expect(rainTabStory(stopped(130))).toBe("Rain stopped 2 h ago · 0.30 in since 2:00 pm");
    expect(overviewStory(stopped(130))).toBe("Rain stopped 2 h ago · 0.30 in since 2:00 pm");
  });

  it("gives the rate in in/hr on the Rain tab, as on the Overview", () => {
    expect(rainTabStory(raining())).toBe("Raining · 0.12 in/hr · 0.30 in since 2:00 pm");
  });

  it("words how long ago as the Scriptable widget does (it cannot import it)", () => {
    const widget = createRequire(import.meta.url)("../../widgets/scriptable/Lookout.js") as {
      agoWords(ms: number): string;
    };
    for (const minutes of [0, 40, 59, 60, 130, 47 * 60 + 59, 48 * 60, 73 * 60]) {
      expect(widget.agoWords(minutes * 60_000)).toBe(agoWords(minutes * 60_000));
    }
  });
});
