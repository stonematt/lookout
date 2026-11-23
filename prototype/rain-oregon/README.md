# Prototype: rain should feel like Oregon (#647), Nov 23, 2026

Design canvas for the prototype pass on stonematt/lookout#647. Branch `prototype/rain-oregon`: keep unmerged, like `prototype/mosaic-overview`. No PR into dev.

- **Canvas (private):** https://claude.ai/artifact/5XYWsospyUdu72Ax2LpNF4
- **Files:** `canvas/project/` (canvas.json + one `.dc.html` per board). Publish with the Artifact tool: `url` above, `root` = `prototype/rain-oregon/canvas`, `file_path` = the changed file.
- **Design system:** Lookout (`design-system/`, mirror https://claude.ai/artifact/Bmri2bVCvPqPumexcZkMXc), installed at `project/ds/lookout/tokens.json`. Boards use inline hexes from `web/src/styles/tokens.css`, as Mosaic did.
- **Data:** `backtest/` is a copy of the grill job's back-test tables (Lookout Post, corrections applied). The 30-day wet hours on every board are the real Oct 25 to Nov 22 days, plus 3 h for the rain scenario on Nov 23. Nov 5 is hatched only to show a missing day.

## Sample numbers (keep consistent)

Mon Nov 23, 2026, Lookout Post. Raining since 2:40 pm; now 4:35 pm; 0.08 in/h (last 60 min); 0.31 in since 2:40. Last 7 days: 19 wet h of 168. Last 30 days: 112 wet h of 696 countable (24 h of Nov 5 hatched). Since Oct 1: 3.98 in, 113 wet h of 1,292; earlier years by Nov 23: 7.58 in (2026), 10.08 (2025), 7.90 (2024), so "drier". Wet share usual by date: 11–15%.
"Later" board: 11:40 am, last rain 2:40 am (9 h ago), 17 wet h this week, 110 h of 30 days.
"DryDay": Fri Aug 14, 2026, dry spell 23 days (real: last rain Jul 22), usual 3; WY2026 to date 26.1 in vs 35.5 / 40.6.

## Boards

| Page | Board | State |
|---|---|---|
| Overview (Q2) | Main (A, raining), Stopped, Later (A) | done: dual feature chart, two wet-hour bars, Wet hours beside Temperature |
| | DryDay (A, dry mode, no meter) | done: Dry spell beside Temperature, Rain since Oct 1 last |
| | AltB (meter in hero top right) | done: two hour bars |
| | AltC (one rain tile: now over season) | **Matt likes it**; done |
| | MainDark | done |
| Meter (Q1, Q3) | Meter (24 h + 7 d vs 7 d + 30 d on three real days, plus a countable-hours example) | done, in hours |
| | TileNames (a: Wet hours / Dry spell, b: Rain lately, c: Wet time / Dry time) | done |
| Rain tab (Q4) | RainA, RainB, RainC (Matt agrees with C), RainCDark | done: wet time in hours; the Now card of C has the accumulation line |
| Rain Events (Q5) | EventsB (beeswarm per water year; earlier years are sample shapes), EventsC (every event of the water year; filter line on one row), EventsBC (B's scatter unchanged over C's list on one page; the filter line sits on the list), Glyphs (three options) | done; EventsA (timeline) dropped on Matt's comment; EventsB's dot caption removed |
| Desktop | Desktop (1440, dark, raining, rows of four, feature full width) | done; Wind, Pressure, Humidity and Sun values are made up |

## Matt's comments on the canvas (Nov 23) and what they changed

1. **Feature chart:** a dual chart. Columns are rain every 5 min (left axis, .01/.02/.03); a line is the running total since the start (right axis, .1/.2/.3 in, sharing the gridlines); "last hour 0.08 in" bracket under the baseline; legend under the chart. Done on Main. Apply to Stopped (bars shifted −68.9, line flat at 0.31 after 4:35), MainDark and AltC (replace its compact ribbon with this chart).
2. **Hours beat percent.** The wet-time meter is now two bars, 7 days and 30 days. Each bar is every hour in the window; the dark fill is the wet hours; the label reads "19 h of 168" / "112 h of 696". Header: "Wet hours" / "each bar is every hour; dark is wet". This replaces the decided "one track 0–100%, two markers": report it as a change Matt's comment drove. Done on every Overview board, Meter and the Rain tab.
3. **Tile order:** Wet hours moves up to row 1 beside Temperature; Rain since Oct 1 moves down to row 2. Done on Main with CSS `order` (Temperature −2, Wet hours −1). Apply to Stopped, Later, MainDark.
4. **C (one rain tile):** Matt likes it. Treat it as his lean for Q2.

The six comment threads were auto-acknowledged; resolve each with a short reply once its change is on every board.

## Recommendations so far (for the report)

- **Q1 meter windows:** 7 d + 30 d (24 h reads 0% on 49% of Dec–Feb days; the now story already covers the last hours).
- **Q2 stories on the Overview:** recommended A, but Matt likes C. Report C as the lead.
- **Q3 tile name:** follows the mode: "Wet hours" / "Dry spell".
- **Q4 Rain tab:** C, now then season then usual. The headline becomes the verdict over the water-year curves; the window strip becomes a dot-on-band chart (each window's rain on its own usual range). Cost: breaks Mosaic decision 14's "zoom = tab's lead".
- **Q5 Rain Events:** the first recommendation was A, the timeline. Matt picked B and C's list instead and dropped A ("drop this, polish C"). The four-step glyph stands.
- Flags: the meter hides in dry mode; the feature and the meter both link to the rain view.

## Countable hours

Countable hours: a label reads wet h of countable h; uncountable hours are hatched at the end of the bar, never counted as dry (GLOSSARY wet hour, wet share). Sample: the 30-day bar reads "112 h of 696", with the 24 hours of Nov 5 hatched; the 7-day bar reads "19 h of 168" (no gap). The Meter board has a week with an 18-hour gap: "19 h of 150".

## Picks and open questions

Picks so far:

- Q1, meter windows: 7 d + 30 d (Matt, Nov 23).
- Q2, the Overview: C, one rain tile.
- Q3, the tile name: (a), it follows the mode, "Wet hours" or "Dry spell".
- Q4, the Rain tab: C, now then season then usual.
- Q5, Rain Events: B + C on one page (EventsBC): B's scatter as drawn, then C's list with the filter line on it, the four-step glyph on each row. Matt: "this is the one to pick" (thread c97346e3). Every Rain Events link opens EventsBC.
- Tile order: row 1 is Temperature and the wet-or-dry tile; Rain since Oct 1 is always in row 2.
- Wet time is shown in hours, never percent, and every label reads "wet h of countable h".
- The feature tile has the dual chart.
- The Rain Events list: C's list with the reworked filter line (water year, sort, size), on one row: "2026–27", "Newest", "Showers and up".
- Rain Events: A, the timeline, is dropped (thread a7488a22). 

Open:

- Tile titles: whether every tile's title names what its number counts (thread a35b05e1).

Picking this up in a new session: read this README, then use ArtifactComments (action read) on the canvas URL to read any open threads.

## Still to do

1. Open: thread a35b05e1 (TileNames, "I like this row"): ask Matt whether the naming pattern becomes the default for the other tiles. It stays open until he answers.
2. Not done: the since-Oct-1 bars carry no hatched gaps (the gaps in Oct are not drawn on them).
