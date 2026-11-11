# Prototype: Mosaic Overview (Nov 2026)

The design round behind the visual-first Overview initiative. The issue holds the decisions to grill; this folder holds the evidence: the canvas boards, rendered images, and the site as it stood before.

Nothing here is built into the site. Every number is a made-up sample shaped like the snapshot, and the boards do not fully agree with each other.

## Live artifact (private until shared)

- **Lookout Mosaic Overview canvas:** https://claude.ai/artifact/1P46EUEZNzrDxWcdv4qyJF. Its files are copied in `canvas/` as published on Nov 11, 2026. It uses the Lookout design system (`design-system/` on `dev`), light theme only.

## The problem

Tiles That Talk (2026.11.6) made every tile a stack of kicker, finding sentence, standfirst, chart, source and chips. On a desktop the Overview reads as a wall of text around twelve charts; "archive 30 m behind" appears eight times. On a phone the context, the thing the page is for, sits in an accordion of closed sentence rows: "Now against normal" is a claim with no picture until a tap. The FT posture works when the headline sits on a visible chart; a finding that describes a chart you cannot see is just text.

`images/before/`: the Overview on the raining golden, phone and 1440 px.

## Decisions so far

1. **A finding always shows its evidence, sized to the space it has.** An open view keeps the FT pattern: a number-led headline over the chart, with the method folded. A closed or small view carries no sentence: a number, a short delta, and a mini chart that makes the comparison. Standfirsts (how-to-read lines) go; charts are labelled directly. This amends Product Principle 3 and the Phone First-Screen Rule.
2. **Now, on top, in context.** The 48-hour ribbon is replaced by today hour by hour against the usual for this date: the middle half and median of readings at each hour over ±7 days around this date in earlier years (`SPREAD_DAYS`, `lookout/snapshot/temperature.py`). Today's line ends at the orange now dot; rain shows as quarter-hour bars along the foot. The band runs to midnight: measured history, not a forecast. The condition sentence and stat strip shrink to one line ("+8° on usual for 2 pm").
3. **One feature tile.** Full width under the now tile, chosen by the refresh run. A live rain event always wins (the Raining board); otherwise something unusual (the dry-day board shows a warm spell). The rule for a dry day is open.
4. **A mosaic, not a list.** A two-column grid of half-width tiles, each a number, a delta and a mini chart: temperature, rain since Oct 1, last 10 days, dry spell, wind, pressure, sun, daylight. About two phone screens; no accordion on the Overview.
5. **Tap zooms.** A tile opens its full view (the Zoom board), where the finding headline, long charts, windows and year-over-year live. Back returns to the mosaic.
6. **A tile is a crop of its zoom's main chart.** The temperature tile is two rows (usual, today) of the zoom's window range chart; the zoom adds Yesterday, 7, 30, 90 and 365 days and Full history. The same rule sets the rain tile (the water-year curves) and the sun tile (the solar almanac).
7. **History grows.** Each comparison uses all the history the station has and switches when there is enough. The usual band uses the station's own recent days (up to 14) until it has an earlier year, then ±7 days in earlier years; under about 20 days of readings it draws only the median. Window rows and tiles appear only once the record covers them (Salmon Run, from Nov 3, 2026, shows Today, Yesterday, 7 days and Full history). One line names the next unlock instead of hatched "n/a" rows. Python decides all of it; the site draws what the snapshot says (ADR-0003).
8. **Full history** joins the windows: the archive's first day to now.
9. **Widgets reuse the same pieces** (later phase): small, the temperature plus one middle-half bar; medium, temperature plus the water-year curves; large, three range-bar rows plus the now line; iOS lock screen, a circular gauge of now within today's low and high.

## Open questions

- What earns the feature slot on a dry day: the biggest departure from normal, a priority list with thresholds, or a fixed seasonal chart?
- Which eight tiles, and in what order?
- Zoom as a full-screen view (drawn) or a bottom sheet?
- Desktop: the same tiles on a four-column grid with the feature spanning two by two (not drawn).
- Do the Rain, Rain Events and Solar tabs keep their accordions, or become their tiles' zoom views?

## Contract impact

New snapshot fields, so schema bumps shipped in two releases each (ADR-0008): the usual band by hour, the Full history window, each comparison's history depth and next-unlock date, and the feature choice. The ten-day list and the this-hour strips already compute the ±7-day usual, so the band reuses their code.

## Files

- `canvas/`: `canvas.json` plus `Main.dc.html` (dry day, the whole mosaic), `Raining.dc.html` (raining, first screen) and `Zoom.dc.html` (the temperature view). Design-canvas components; opened as plain HTML they render close to the canvas without the `support.js` runtime, so links do nothing.
- `images/proposed/`: each board rendered at 390 px, 2×.
- `images/before/`: the current Overview on the raining golden at 390 px (2×) and 1440 px.

## Known gaps

- Light theme only; no dark boards.
- No desktop board and no widget board.
- On the dry-day board the feature chart's "normal range" label sits under the bars, and the board runs taller than its content.
- Only the temperature tile and the feature tile link to a zoom; only the temperature zoom is drawn.
- The Raining board's tiles below the fold are stubs.
