# Prototype: Mosaic Overview (Nov 2026)

The design round behind the visual-first Overview initiative. The issue holds the decisions to grill; this folder holds the evidence: the canvas boards, rendered images, and the site as it stood before.

**Status: final, and the implementation target (Matt, Nov 11, 2026).** This branch, `prototype/mosaic-overview`, stays unmerged and complete as the reference for building the Mosaic Overview. Never merge it, rebase it or delete it. The spec on the issue states what to build; these boards show what it should look like. Where they differ, the spec wins. A later design change lands as a new commit here, recorded as a numbered decision below.

Nothing here is built into the site. Every number is a made-up sample shaped like the snapshot. Where the boards disagree, the second-pass decisions (10–21) win over the first pass (1–9). The boards to build from are `Main` and `MainDark` (phone), `Desktop`, `Zoom` and `ZoomDark` (temperature), `RainZoom` and `WindZoom` and `WindZoomDark`. `Raining` shows the rain feature and the battery line; its lower tiles are stubs.

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

15. **Matt's review (Nov 11): the flow is fine and dark looks OK.** A tile hidden for missing history shifts the later tiles; that is accepted. The dark tokens stay as they are; the #38444f inset stays on the board as a reference only.
16. **The band's label goes in the legend, not the plot.** Bars cover a label drawn inside the band, in both themes. Every chart with a band under bars gets a legend swatch, "usual range" (the band fill with its dashed edge), and no label inside the plot. The "today" label sits above the plot with a leader line down to today's bar.

17. **Ten tiles: Inside is the tenth** (Matt, Nov 11). It has one row for each indoor sensor: the reading, its humidity and a 24-hour line. Lookout Post has two sensors, Salmon Run one. The tile hides on a station with no indoor sensor. On the phone it pairs with Daylight; on desktop row 4 holds Daylight and Inside, with two cells empty.
18. **Battery health is a header warning, not a tile** (Matt, Nov 11). Ambient reports each sensor's battery only as OK or low. A line under the freshness note names the sensor, for example "Rain gauge battery low", and appears only while a flag reads low. It is drawn in ink, not orange, because orange means now. The Raining board shows it.

19. **The wind zoom leads with a wind rose** (Matt, Nov 11). It has 16 directions over a window of 24 h, 7 days or 30 days. In each direction a dark wedge reaches the average wind and a lighter, edged wedge behind it reaches the highest gust. Rings sit every 5 mph, and now is the orange dot with a leader to its label. Calm readings have no direction, so they are left out, and a line under the rose gives their share. Under the rose, a bar chart gives the share of windy readings from each of the 8 compass points. The wind tile links to this zoom. Boards: `WindZoom`, `WindZoomDark`. Radial charts for time, such as the year in a circle, are tabled.

20. **Matt's last calls (Nov 11).** He approves the Rain tab changes in decision 14. #252's selector is set aside: the rain zoom ships without it, and the board's period control is a placeholder. Wind stays in slate, with no new wind tokens.

21. **This branch is the implementation target** (Matt, Nov 11). It stays complete and unmerged for reference; see Status at the top.

## Third pass (Nov 12)

22. **Sources go in one Overview footer, not on each tile** (Matt, Nov 12). Mosaic tiles and zooms carry no "Source · …" fold and no method paragraph. One footer at the foot of the Overview lists, for each tile, what it reads and how its comparison is made. The archive time is not repeated there, because the header's freshness line already says it once. The footer also shows the site's release version (for example "Lookout 2026.11.12"); `debug.html` shows the same version, and an endpoint returns it. The Rain, Rain Events and Solar tabs keep their source notes for now, as they keep their accordions. No board draws the footer: it follows the `Main` board's type and spacing at the page foot. Release B2 (#472, #477, #478; version: its own ticket).

## Accessibility audit (Nov 11, `/impeccable audit`)

Axe-core 4.10 (WCAG 2.2 A/AA and best practice) on every board, light and dark, found no colour-contrast failures in text. The impeccable detector found no anti-patterns, only advisories that sizes and radii are off the DESIGN.md ramp (prototype drift; the build uses the tokens). Chart graphics were measured by hand, since axe does not check them.

Fixed on the boards:
- The "today" label overlapped the bars (1.3:1 against a warm bar); it now sits above the plot.
- The band label was lost under the bars; it is now a legend swatch (decision 16).
- Tile links carried "Temperature: open the temperature view" as their `aria-label`, which hides the finding from a screen reader. They now carry the finding: "Temperature 60°, high 62°, 7° above usual. Open the temperature view." This is decision 12 on the issue made concrete.

For the spec to carry:
- **P1. A tile is one link, and its name is its finding.** Every tile is an `<a href="#overview/<tile>">`, not only Temperature and the feature. Its accessible name is the tile's dropped sentence plus "Open the … view". The mini chart is `aria-hidden` inside the link so the sentence is not read twice.
- **P1. Zoom focus.** Phone: opening a zoom moves focus to its heading; Back returns focus to the tile it came from. Desktop: the tile has `aria-expanded` and `aria-controls`, focus moves into the zoom, and Close or Escape returns focus to the tile. Opening and closing respect `prefers-reduced-motion`.
- **P1. Headings and landmarks.** The Overview sits in `<main>` with one `<h1>` (the station name, visually hidden where the selector shows it); each tile's title and each zoom section's kicker is a heading. The boards draw them as plain text.
- **P2. Above-usual against at-or-below differs by hue only** (#b5602a against #5a7da1 is 1.05:1; dark 1.2:1). Position against the band carries the same fact, and the legend names both, so this passes WCAG 1.4.1. Keep the band edge visible wherever these bars are drawn.
- **P2. The hero's band has no edge.** Its fill is 1.3:1 (light) and 1.9:1 (dark) on the now surface, and `--temp-band-edge` reaches only 2.9:1 and 2.3:1 there. The median line (7.9:1, 8.1:1) and the today line carry the comparison, so the band is context. If the band must stand alone, it needs an edge colour that reaches 3:1 on the now surface.
- **P2. Segmented controls** (the range and period selectors) are about 35 px tall on the boards. The site's 44 px floor applies.
- **P3. The dry-spell grid's rain days** stand 2.7:1 off the dry days. The number and "usual is 3" carry the finding.

## Open questions

None left for the prototype. Next: `/to-spec` on the issue.

## Contract impact

New snapshot fields, so schema bumps shipped in two releases each (ADR-0008): the usual band by hour, the Full history window, each comparison's history depth and next-unlock date, and the feature choice. Release A also adds the indoor readings (`tempinf`/`humidityin`, `temp1f`/`humidity1`, and more by sensor) and each sensor's battery flag (`battout`, `battin`, `batt1`; 1 is OK), which no snapshot field carries today. The ten-day list and the this-hour strips already compute the ±7-day usual, so the band reuses their code.

## Files

- `canvas/`: `canvas.json` plus the boards. First pass: `Main.dc.html` (dry day, the whole mosaic), `Raining.dc.html` (raining, first screen), `Zoom.dc.html` (the temperature view). Second pass: `Desktop.dc.html` (1440 px, temperature zoom open in place), `MainDark.dc.html` and `ZoomDark.dc.html` (dark), `RainZoom.dc.html` (the rain zoom). Design-canvas components; opened as plain HTML they render close to the canvas without the `support.js` runtime, so links do nothing.
- `images/proposed/`: each phone board at 390 px, 2×; `Desktop.png` at 1440 px, 1×. Rendered with the sticky tab bar unpinned so it does not cover tiles.
- `images/before/`: the current Overview on the raining golden at 390 px (2×) and 1440 px, and the current Rain tab's top section on the dry golden at 390 px (`Rain-top-390.png`).

## Known gaps

- No widget board.
- Charts are SVGs scaled to their box, so on the desktop board their text grows with the tile (zoom labels near 20 px, tile labels near 12 px). The site redraws at width (`drawSized`), so this is a drawing artefact, not a proposal.
- Only the temperature, rain and wind zooms are drawn; the rain zoom is light only.
- The Raining board's tiles below the fold are stubs, and it has no humidity tile.
- The dark boards are the light boards with the DESIGN.md -dark values swapped in; no dark-specific layout was tried.
