/**
 * Overview C, the rain tile (#661, spec #647 "Overview C"; boards MainDark,
 * Desktop, StoppedC, StoppedCDark, DesktopStopped, approved by Matt): while
 * rain leads, one tile under the now tile carries the now story on top and
 * the season at its foot, in place of the feature tile. Rain leads exactly
 * while the Feature Rule picks rain and the snapshot tells the now story
 * (last rain within 3 hours, ADR-0011, `rainLeads`); the site decides none
 * of it.
 *
 * - The now part, one link to the rain zoom (`#overview/rain`): the label,
 *   a filled dot and "Raining" while raining, a hollow ring and "Rain" once
 *   stopped (#660); the story, "0.08 in/hr" over "0.31 in since 2:40 pm",
 *   or "Rain stopped 40 min ago · 0.31 in since 2:40 pm"; then the dual
 *   chart (`rain-lead-chart.ts`) and its legend.
 * - The season foot, one link to the Rain tab's season section
 *   (`#rain/season`): the two wet-hour bars and their hatch key
 *   (`wet-meter.ts`), hidden in dry mode, then Rain since Oct 1 with the
 *   water year's verdict and the usual rain by now.
 *
 * While the tile is drawn the Wet hours and Rain since Oct 1 tiles leave
 * the mosaic (`overviewSections`), and the now tile's foot draws no bars,
 * so no count shows twice (Call 4 on #647). The tile's section id is the
 * rain zoom's slug, so the zoom opens under it as under any tile.
 *
 * Every value is the snapshot's: `rain_now` (its `last_3_hours` for the
 * chart), `wet_hours`, and `rain_tab`. Snapshot text reaches the page as
 * text, never as HTML.
 *
 * Offers `rainLeads`, `rainLeadTile`, `SEASON_ROUTE`, `isSeasonRoute` and
 * `followSeason`.
 */
import "./rain-lead.css";

import { make, spoken } from "../dom.js";
import { inches, inchesPerHour } from "../format.js";
import { nowStory, toldStory } from "../now-story.js";
import type { ToldStory } from "../now-story.js";
import type { DashboardSnapshot, Verdict } from "../snapshot.js";
import type { OverviewSection } from "./index.js";
import { rainTile } from "./mosaic-tiles.js";
import { storyChart } from "./rain-lead-chart.js";
import { meterParts, meterShown } from "./wet-meter.js";

/** The Rain tab's season section, which the season foot opens. */
export const SEASON_ROUTE = "rain/season";

/**
 * Whether a hash is the season route, which the tab shell keeps.
 *
 * @param hash - A URL hash, with or without its `#`.
 * @returns True for `rain/season`.
 */
export function isSeasonRoute(hash: string): boolean {
  return hash.replace(/^#/, "") === SEASON_ROUTE;
}

/**
 * Bring the Rain tab's season part into view when the route names it; the
 * page calls this each time the Rain tab opens. A head without a season
 * part (no wet hours) leaves the tab at its top.
 *
 * @param panel - The Rain tab's panel.
 */
export function followSeason(panel: HTMLElement): void {
  if (!isSeasonRoute(location.hash) || panel.hidden) return;
  panel.querySelector<HTMLElement>("[data-part=season]")?.scrollIntoView?.({ block: "start" });
}

/**
 * The now story while rain leads, or null.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The story.
 */
function storyOf(snapshot: DashboardSnapshot): ToldStory | null {
  return snapshot.feature?.kind === "rain" ? toldStory(snapshot) : null;
}

/**
 * Whether rain leads: the Feature Rule picked rain and the snapshot tells
 * the now story (Python's 3 hours, ADR-0011). A snapshot before #659 never
 * leads here, and the Overview keeps the feature tile.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns True while the rain tile is drawn.
 */
export function rainLeads(snapshot: DashboardSnapshot): boolean {
  return storyOf(snapshot) !== null;
}

/** A run of words that never wraps inside itself. */
function run(...parts: (string | Node)[]): HTMLElement {
  const node = make("span", "rain-lead-run");
  node.append(...parts);
  return node;
}

/**
 * The story's finding, in the one now story's words (`now-story.ts`, #694):
 * raining, the rate in Fact type over the event's rain; stopped, one line in
 * two runs that wrap after the dot.
 *
 * @param snapshot - The dashboard snapshot.
 * @param rain - The story.
 * @returns The story's elements.
 */
function story(snapshot: DashboardSnapshot, rain: ToldStory): HTMLElement[] {
  const told = nowStory(snapshot);
  const event = told?.event ?? null;
  const eventRun = (words: [string, string]) => run(make("b", "", words[0]), ` ${words[1]}`);
  const finding = make("h2", "tile-finding tile-finding-lead rain-lead-finding");
  finding.dataset.part = "finding";
  if (rain.raining) {
    const rate = make("p", "rain-lead-rate");
    rate.dataset.part = "value";
    rate.setAttribute("aria-hidden", "true");
    const unit = make("span", "rain-lead-unit", "in/hr");
    rate.append(inchesPerHour(rain.rate_in_hr).replace(/ in\/hr$/, ""), unit);
    finding.append(event ? eventRun(event) : "in the last hour");
    const body = make("div", "rain-lead-body");
    body.append(rate, spoken(`Raining, ${inchesPerHour(rain.rate_in_hr)}: `), finding);
    return [body];
  }
  const stopped = told?.lead ?? "";
  if (event) finding.append(run(`${stopped} ·`), " ", eventRun(event));
  else finding.append(run(stopped));
  return [finding];
}

/**
 * The tile's label: a filled dot and "Raining", or a hollow ring and "Rain".
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The label's words.
 */
function label(snapshot: DashboardSnapshot): string {
  return storyOf(snapshot)?.raining ? "Raining" : "Rain";
}

/** The water year's verdict in words: "drier than usual". */
function verdictWords(verdict: Verdict): [string, string] {
  return [verdict, verdict === "about the same" ? " as usual" : " than usual"];
}

/**
 * Rain since Oct 1 at the foot: the Rain since Oct 1 tile's number, then
 * the verdict and the usual rain by now, or the tile's own delta without a
 * verdict.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The block.
 */
function rainSince(snapshot: DashboardSnapshot): HTMLElement {
  const block = make("div", "rain-lead-year");
  block.dataset.part = "rain-since";
  const reading = rainTile.read(snapshot);
  const left = make("div", "rain-lead-year-number");
  left.append(make("span", "rain-lead-year-name", rainTile.title));
  if (reading.value !== null) {
    const value = make("span", "rain-lead-year-value", reading.value);
    if (reading.unit) value.append(make("span", "rain-lead-year-unit", ` ${reading.unit}`));
    left.append(value);
  }
  const usual = snapshot.rain_tab.usual_to_date ?? null;
  const right = make("div", "rain-lead-year-verdict");
  if (usual?.verdict) {
    const [word, rest] = verdictWords(usual.verdict);
    const line = make("span", "rain-lead-verdict");
    line.dataset.part = "verdict";
    line.append(make("b", "", word), rest);
    right.append(line, make("span", "rain-lead-usual", `usual by now ${inches(usual.usual_in)}`));
  } else if (reading.delta) {
    right.append(make("span", "rain-lead-verdict", reading.delta));
  }
  block.append(left, right);
  return block;
}

/**
 * Fill the rain tile: the now link (label, story, chart), then the season
 * foot's link (bars, Rain since Oct 1).
 *
 * @param host - The tile, its kicker already drawn.
 * @param snapshot - The dashboard snapshot.
 */
function render(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const rain = storyOf(snapshot);
  if (!rain) return;
  host.dataset.raining = String(rain.raining);

  const now = make("a", "feature-link mosaic-link rain-lead-now");
  now.href = `#overview/${rainTile.slug}`;
  now.dataset.part = "now";
  const top = make("div", "feature-top");
  const kicker = host.querySelector<HTMLElement>(":scope > [data-part=kicker]");
  if (kicker) {
    const mark = make("span", rain.raining ? "feature-live-dot" : "rain-lead-ring");
    mark.dataset.part = rain.raining ? "live" : "stopped";
    mark.setAttribute("aria-hidden", "true");
    kicker.prepend(mark);
    // The story says it in full; the label is for the eye.
    kicker.setAttribute("aria-hidden", "true");
    top.append(kicker);
  }
  const chevron = make("span", "feature-chevron");
  chevron.setAttribute("aria-hidden", "true");
  top.append(chevron);
  now.append(top, ...story(snapshot, rain));
  const slots = rain.last_3_hours ?? null;
  const chart = slots ? storyChart(rain, slots, snapshot.built_at_ms) : null;
  if (chart) now.append(chart.block);
  now.append(spoken("Open the rain view."));

  const season = make("a", "rain-lead-season");
  season.href = `#${SEASON_ROUTE}`;
  season.dataset.part = "season";
  const wet = snapshot.wet_hours ?? null;
  if (wet && meterShown(snapshot)) {
    const bars = make("div", "wet-meter rain-lead-bars");
    bars.dataset.part = "wet-meter";
    bars.append(...meterParts(wet));
    season.append(bars);
  }
  season.append(rainSince(snapshot), spoken("Open the Rain tab at the season."));

  host.replaceChildren(now, season);
  chart?.start();
}

/** The rain tile's line and method in the Overview's sources footer. */
function source(snapshot: DashboardSnapshot) {
  const bars = meterShown(snapshot)
    ? " The wet-hour bars count the station-local clock hours of the last 7 and 30 days, to the last whole hour: an hour with at least one tip of the gauge (0.01 in) is wet, of the hours the record can count. An hour with no reading or a gap over 30 minutes is missing, hatched at the bar's end, never dry."
    : "";
  return {
    line: "rain each 5 minutes, the event's total, wet hours, rain to date.",
    method:
      "Last rain is the gauge's last tip; while it is within 3 hours, rain leads. The columns are the rain each 5 minutes of the last 3 hours, from the daily counter; the line is the event's running total, ending on the station's own event total, on an axis ten times the columns'. A 5 minutes with no reading is hatched, never read as dry." +
      bars +
      ` ${rainTile.source(snapshot).method}`,
  };
}

/** The rain tile, as an Overview section. */
export const rainLeadTile: OverviewSection = {
  // The rain zoom's slug, so its zoom opens under this tile.
  id: rainTile.slug,
  label: "Rain",
  kicker: label,
  span: 4,
  phoneTop: true,
  shown: rainLeads,
  source,
  render,
};
