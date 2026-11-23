# Prototype: rain should feel like Oregon (#647), Nov 23, 2026

Design canvas for the prototype pass on stonematt/lookout#647. Branch `prototype/rain-oregon`: keep unmerged, like `prototype/mosaic-overview`. No PR into dev.

- **Canvas (private):** https://claude.ai/artifact/5XYWsospyUdu72Ax2LpNF4
- **Files:** `canvas/project/` (canvas.json + one `.dc.html` per board). Publish with the Artifact tool: `url` above, `root` = `prototype/rain-oregon/canvas`, `file_path` = the changed file.
- **Design system:** Lookout (`design-system/`, mirror https://claude.ai/artifact/Bmri2bVCvPqPumexcZkMXc), installed at `project/ds/lookout/tokens.json`. Boards use inline hexes from `web/src/styles/tokens.css`, as Mosaic did.
- **Data:** `backtest/` is a copy of the grill job's back-test tables (Lookout Post, corrections applied). The 30-day wet hours on every board are the real Oct 25 to Nov 22 days, plus 3 h for the rain scenario on Nov 23. Nov 5 is hatched only to show a missing day.

## Sample numbers (keep consistent)

Mon Nov 23, 2026, Lookout Post. Raining since 2:40 pm; now 4:35 pm; 0.08 in/h (last 60 min); 0.31 in since 2:40. Last 7 days: 19 wet h of 168 (11%). Last 30 days: 112 wet h of 720 (16%). Since Oct 1: 3.98 in, 113 wet h (9%); earlier years by Nov 23: 7.58 in (2026), 10.08 (2025), 7.90 (2024), so "drier". Wet share usual by date: 11–15%.
"Later" board: 11:40 am, last rain 2:40 am (9 h ago), 17 wet h this week, 110 h of 30 days.
"DryDay": Fri Aug 14, 2026, dry spell 23 days (real: last rain Jul 22), usual 3; WY2026 to date 26.1 in vs 35.5 / 40.6.

## Boards

| Page | Board | State |
|---|---|---|
| Overview (Q2) | Main (A, raining) | **updated for Matt's comments** |
| | Stopped, Later (A) | need the comment changes (below) |
| | DryDay (A, dry mode, no meter) | done |
| | AltB (meter in hero top right, season pair first row) | done |
| | AltC (one rain tile: now over season) | **Matt likes it**; needs the comment changes |
| | MainDark | needs the comment changes |
| Meter (Q1, Q3) | Meter (24 h + 7 d vs 7 d + 30 d on three real days) | done; the meter form changed after (see below) |
| | TileNames (a: Wet hours / Dry spell, b: Rain lately, c: Wet time / Dry time) | done |
| Rain tab (Q4) | RainA (water year leads), RainB (season calendar leads), RainC (now, season, usual), RainCDark | done |
| Rain Events (Q5) | EventsA (timeline + step glyph rows) | done |
| | EventsB (events by size, beeswarm per water year), EventsC (last ten events as bars), Glyphs (520x620: drop count / four steps / sized ring, x sprinkle, shower, rain, storm) | **not written yet**; frames are in canvas.json |
| Desktop | Desktop (1440, dark, raining, rows of four, feature full width per Mosaic decision 23) | **not written yet** |

## Matt's comments on the canvas (Nov 23) and what they changed

1. **Feature chart:** a dual chart. Columns are rain every 5 min (left axis, .01/.02/.03); a line is the running total since the start (right axis, .1/.2/.3 in, sharing the gridlines); "last hour 0.08 in" bracket under the baseline; legend under the chart. Done on Main. Apply to Stopped (bars shifted −68.9, line flat at 0.31 after 4:35), MainDark and AltC (replace its compact ribbon with this chart).
2. **Hours beat percent.** The wet-time meter is now two bars, 7 days and 30 days. Each bar is every hour in the window; the dark fill is the wet hours; the label reads "19 h of 168" / "112 h of 720". Header: "Wet hours" / "each bar is every hour; dark is wet". This replaces the decided "one track 0–100%, two markers": report it as a change Matt's comment drove. Done on Main; apply to Stopped, Later (17 h, 110 h), AltC, MainDark. The Rain tab's "Share of hours wet" rows should follow (hours, not %).
3. **Tile order:** Wet hours moves up to row 1 beside Temperature; Rain since Oct 1 moves down to row 2. Done on Main with CSS `order` (Temperature −2, Wet hours −1). Apply to Stopped, Later, MainDark.
4. **C (one rain tile):** Matt likes it. Treat it as his lean for Q2.

The six comment threads were auto-acknowledged; resolve each with a short reply once its change is on every board.

## Recommendations so far (for the report)

- **Q1 meter windows:** 7 d + 30 d (24 h reads 0% on 49% of Dec–Feb days; the now story already covers the last hours).
- **Q2 stories on the Overview:** recommended A, but Matt likes C. Report C as the lead.
- **Q3 tile name:** follows the mode: "Wet hours" / "Dry spell".
- **Q4 Rain tab:** C, now then season then usual. The headline becomes the verdict over the water-year curves; the window strip becomes a dot-on-band chart (each window's rain on its own usual range). Cost: breaks Mosaic decision 14's "zoom = tab's lead".
- **Q5 Rain Events:** A, the timeline (stems at each event's start, height = total, newest named, missing days hatched on the axis, "4 days missing" plus a "How events are found" fold) and the four-step glyph.
- Flags: the meter hides in dry mode; the feature and the meter both link to the rain view.

## Still to do

1. Apply the comment changes to Stopped, Later, AltC, MainDark; publish each; resolve the six threads.
2. Write EventsB, EventsC, Glyphs and Desktop (dark); publish each.
3. Update the canvas notes n1/n3 (meter is now hours bars; Matt leans C).
4. Commit and push this branch.
5. SendMessage to session `rain-oregon-principles`: the canvas link, the branch, and for each question the options and the recommendation (Q2 noting Matt likes C); then wait for Matt's picks.
