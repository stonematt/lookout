# Prototype: Mosaic Overview (Nov 2026)

The design round behind the visual-first Overview initiative. The issue holds the decisions to grill; this folder holds the evidence: the canvas boards, rendered images, and the site as it stood before.

Nothing here is built into the site. Every number is a made-up sample shaped like the snapshot, and the boards do not fully agree with each other.

## Live artifact (private until shared)

- **Lookout Mosaic Overview canvas:** https://claude.ai/artifact/1P46EUEZNzrDxWcdv4qyJF. Its files are copied in `canvas/` as published on Nov 11, 2026. It uses the Lookout design system (`design-system/` on `dev`), light and dark boards.

## The problem

Tiles That Talk (2026.11.6) made every tile a stack of kicker, finding sentence, standfirst, chart, source and chips. On a desktop the Overview reads as a wall of text around twelve charts; "archive 30 m behind" appears eight times. On a phone the context, the thing the page is for, sits in an accordion of closed sentence rows: "Now against normal" (the old label) is a claim with no picture until a tap. The FT posture works when the headline sits on a visible chart; a finding that describes a chart you cannot see is just text.

`images/before/`: the Overview on the raining golden, phone and 1440 px.

## Decisions so far

1. **A finding always shows its evidence, sized to the space it has.** An open view keeps the FT pattern: a number-led headline over the chart, with the method folded. A closed or small view carries no sentence: a number, a short delta, and a mini chart that makes the comparison. Standfirsts (how-to-read lines) go; charts are labelled directly. This amends Product Principle 3 and the Phone First-Screen Rule.
2. **Now, on top, in context.** The 48-hour ribbon is replaced by today hour by hour against the usual band for this date: the middle half and median of readings at each hour over ±7 days around this date in earlier years (`SPREAD_DAYS`, `lookout/snapshot/temperature.py`). Today's line ends at the orange now dot; rain shows as quarter-hour bars along the foot. The band runs to midnight: measured history, not a forecast. The condition sentence and stat strip shrink to one line ("+8° on usual for 2 pm").
3. **One feature tile.** Full width under the now tile, chosen by the refresh run. A live rain event always wins (the Raining board); otherwise something unusual (the dry-day board shows a warm spell). The rule for a dry day is open.
4. **A mosaic, not a list.** A two-column grid of half-width tiles, each a number, a delta and a mini chart: temperature, rain since Oct 1, last 10 days, dry spell, wind, pressure, sun, daylight. About two phone screens; no accordion on the Overview.
5. **Tap zooms.** A tile opens its full view (the Zoom board), where the finding headline, long charts, windows and year-over-year live. Back returns to the mosaic.
6. **A tile is a crop of its zoom's main chart.** The temperature tile is two rows (usual, today) of the zoom's window range chart; the zoom adds Yesterday, 7, 30, 90 and 365 days and Full history. The same rule sets the rain tile (the water-year curves) and the sun tile (the solar almanac).
7. **History grows.** Each comparison uses all the history the station has and switches when there is enough. The usual band uses the station's own recent days (up to 14) until it has an earlier year, then ±7 days in earlier years; under about 20 days of readings it draws only the median. Window rows and tiles appear only once the record covers them (Salmon Run, from Nov 3, 2026, shows Today, Yesterday, 7 days and Full history). One line names the next unlock instead of hatched "n/a" rows. Python decides all of it; the site draws what the snapshot says (ADR-0003).
8. **Full history** joins the windows: the archive's first day to now.
9. **Widgets reuse the same pieces** (later phase): small, the temperature plus one middle-half bar; medium, temperature plus the water-year curves; large, three range-bar rows plus the now line; iOS lock screen, a circular gauge of now within today's low and high.

## Second pass (Nov 11, after the grill)

The grill on the issue settled the tiles, the Feature Rule and the zoom; this pass drew what the spec still needed. Labels now say "usual" everywhere, on the old boards too, and the phone mosaic has the ninth tile (Humidity, with dew point; it takes the Air card).

10. **Desktop is the phone order on four columns.** Row-first flow, the feature spanning 2×2 at the left. The phone's first two pairs (Temperature | Rain, Last 10 days | Dry spell) become the 2×2 block beside the feature; the next two pairs (Wind | Pressure, Humidity | Sun) fill row 3. Nothing reorders between phone and desktop, so the grid does not fight the phone's two-column order. Board: `Desktop`.
11. **A zoom opens in place on desktop** as a full-width row under the tapped tile's row, at the same `#overview/<tile>` URL as the phone's full-screen zoom. For a tile beside the feature, that is under the feature block. The zoom's charts sit two by two; Close returns to the mosaic.
12. **The usual band reads in dark in the hero, with one change.** The band (`--temp-band` dark on the now surface, 1.9:1) reads better than in light (1.3:1). The median line must use `--ink-on-now` (8.1:1). The `--median` token is #11161c in both themes, so on the dark now surface it measures 1.8:1 and the line all but vanishes. Board: `MainDark`, with the failing version as an inset.
13. **A band drawn under bars is too heavy in dark.** On `--temp-band` dark (#606d7a) the middle-half bars measure 2.6:1 and the band's label is lost (feature chart, day by day, year by year). A band at #38444f keeps bars at 4.9:1 or more and the label at 4.8:1, and sits 1.6:1 off the tile (light: 1.2:1). That is a DESIGN.md token change, so it is a proposal here. Board: `ZoomDark`, with #38444f as an inset.
14. **The Rain tab's top section cannot lead the rain zoom unchanged.** Today the top section is "Rain, by window": the window strip over each day of the last 30 or 90 days (`images/before/Rain-top-390.png`). The rain tile draws the water-year curves, which sit second on the Rain tab. For a tile to be a crop of its zoom's lead (decision 6), the Rain tab must change:
    - The water year becomes the top section, with the window strip under its chart.
    - #252's period selector sits on that chart. It is drawn as To date, Oct–Mar and Full year, on water years only.
    - No standfirst and no lag chips (release 1). A window the record does not cover hides instead of showing "n/a".
    - The daily bars move down the Rain tab. The zoom ends at "More in Rain".

    Board: `RainZoom`.

## Open questions

- Nine tiles leave Daylight alone on the phone's last row and three empty cells on desktop. Add a tenth tile, merge Sun and Daylight, or accept the gap?
- What is #252's selector for: narrowing the dates (drawn) or choosing which years to compare?
- Adopt a lighter dark band for bands drawn under bars (#38444f, or a new token beside `--temp-band`)?
- A hidden tile (history missing) shifts every later tile by one cell, so pairs change partners on the phone. Accept it, or hold each tile's slot?

## Contract impact

New snapshot fields, so schema bumps shipped in two releases each (ADR-0008): the usual band by hour, the Full history window, each comparison's history depth and next-unlock date, and the feature choice. The ten-day list and the this-hour strips already compute the ±7-day usual, so the band reuses their code.

## Files

- `canvas/`: `canvas.json` plus the boards. First pass: `Main.dc.html` (dry day, the whole mosaic), `Raining.dc.html` (raining, first screen), `Zoom.dc.html` (the temperature view). Second pass: `Desktop.dc.html` (1440 px, temperature zoom open in place), `MainDark.dc.html` and `ZoomDark.dc.html` (dark), `RainZoom.dc.html` (the rain zoom). Design-canvas components; opened as plain HTML they render close to the canvas without the `support.js` runtime, so links do nothing.
- `images/proposed/`: each phone board at 390 px, 2×; `Desktop.png` at 1440 px, 1×. Rendered with the sticky tab bar unpinned so it does not cover tiles.
- `images/before/`: the current Overview on the raining golden at 390 px (2×) and 1440 px, and the current Rain tab's top section on the dry golden at 390 px (`Rain-top-390.png`).

## Known gaps

- No widget board.
- Charts are SVGs scaled to their box, so on the desktop board their text grows with the tile (zoom labels near 20 px, tile labels near 12 px). The site redraws at width (`drawSized`), so this is a drawing artefact, not a proposal.
- The light feature chart's "usual range" label still sits under the bars.
- Only the temperature and rain zooms are drawn; the rain zoom is light only.
- The Raining board's tiles below the fold are stubs, and it has no humidity tile.
- The dark boards are the light boards with the DESIGN.md -dark values swapped in; no dark-specific layout was tried.
