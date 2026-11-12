Lookout is one weather station's own record. Every block says one thing with the number in it and shows its evidence, sized to the space it has. The Overview is visual first: the now tile (today against the usual band), one feature tile, then a mosaic of small tiles (a number, a short delta, a mini chart) that each zoom to a full view, with method and provenance once in a sources footer. The posture is the Financial Times data desk; the presentation is a Dark Sky style current reading. It is a dashboard for the owner of a few Ambient Weather stations, read mostly on a phone in short glances, and by friends through a shared link.

Start from `tokens.json` for values and from the components for markup. Dark is the primary design and the first theme; light is its twin, held to the same contrast rules.

## Content fundamentals

- **Findings first.** A finding states what the data shows, with the number in it: "It has rained for 3 hours, and 0.42 in has fallen." It never describes the chart ("A chart of rain"). Set it in `finding` (`finding-lead` for the feature tile and a tab's top finding). A mosaic tile carries no sentence: its number and short delta are its finding, and its zoom carries the headline.
- **Headlines speak, evidence stays exact.** A finding says a duration the way a person does: `45 minutes`, `about 26 hours`, `about 3.5 days`, `about 12 days`; `about` marks a rounded value. Stat rows, chart labels, event rows and source lines keep the exact form: times `9:36 am`, dates with times `Oct 27, 11:00 pm`, durations `1 d 10 h`, `5 h 12 m`, `45 m`.
- **Units and zone.** Imperial: °F, mph, inches of rain, inHg, W/m², kWh/m², the UV index. Time is the station's own zone, never the viewer's.
- **Sentence case, left-aligned, no italics, no all-caps runs, no emoji.** Emphasis is weight. Kickers and tile titles name the subject in sentence case ("Rain", "Rain since Oct 1").
- **The tone is plain and measured.** Real copy: "Tue Nov 3, 7:40 am · 3 m ago", "No rain events in <water year> yet.", "Couldn't load this event's curve.", "Loading the curve…", "the bold line", "the band", "the rule marked now".
- **Name marks by label or position, never by colour.** "the bold line", not "the blue line".
- **Name the years, newest first.** A finding that compares years says which: "less than the same days of 2025 and 2024".
- **No forecasts and no fabricated values.** Lookout shows measured data only and says so. A weather app does forecasts.
- **Use the project's terms exactly.** station, station selector, snapshot, archive, gap status (valid, prorated, invalid), feature, usual band, Event Catalog, water year. Say "usual", never "normal". Say "station", not "device".
- **Archive lag is said once, in the header's freshness line**, and on the Rain, Rain events and Solar tabs in the source note; never in a finding or on an Overview tile. A live value beats an archived one.
- **Python decides.** Findings, short forms, the feature, which tiles and windows the record supports, gap status and every comparison are decided before the page and arrive in the snapshot. The site draws what the snapshot says; do not recompute a number in the UI. Each fact is computed once and shows the same number everywhere (the One Number Rule).

## Visual foundations

**Colour.** The chrome is near-greyscale slate; colour is kept for data. Rain is blue (`rain-*`), sun is gold (`sun-*`), temperature runs cool blue to warm orange (`temp-*`), and one orange, `now`, marks the subject of the finding. Every saturated colour on screen is a measurement, a usual band or `now`.
- Page behind everything: `ground`. Tiles and event rows: `tile` on `ground`. The one tinted tile, the current reading: `now-surface`. There are no shadows; depth is tile on ground.
- Text: `ink` for headlines and values, `ink-2` for deltas and labels, `ink-3` for kickers, axis ticks and source lines. Each passes 4.5:1 on `ground`, `tile` and `now-surface` in both themes. On `now-surface` secondary text may use `ink-on-now`.
- `rule` is for dividers and gridlines only (it is decorative and under 3:1 on purpose). Anything a reader must see, such as a control border, a chart baseline, a chip or a chevron ring, uses `rule-strong`.
- `now` is one orange subject per tile: the now rule, now dot and label, the dot over today's bar, the current month or hour outline, the latest chip, the now dot ending a mini chart's line. It never touches a data fill: it sits on `tile`, `now-surface` or `ground`, behind a gap in the surface colour (the hero's now label sits on a 4px `now-surface` halo; an outline round a ramp cell sits outside it with a 1px gap). Do not use `now` for chrome or for anything else.
- A usual band (`rain-band`, `temp-band`, `sun-band`) covers the middle half of the comparison set, with a dashed `median`. It passes 3:1 by its fill in dark and by its `*-band-edge` in light.
- Ramps (`rain-0` to `rain-4`, `sun-0` to `sun-4`): in light, rain runs pale to deep; in dark, deeper means lighter, so more rain is always the more visible cell. Text in a ramp cell uses the matching `on-rain-N` or `on-sun-N`.
- Colours that must be told apart also differ in lightness. Rain intensity on the hero's rain track is bar height, not shade. Today's bar keeps its data colour and is marked by the `now` dot and the word "today".
- Follow the OS colour scheme. `?theme=light` or `?theme=dark` in the address overrides it for that load; nothing is stored and there is no theme control.

**Type.** One family, Lexend (self-hosted on the site, with `system-ui, sans-serif` behind it), upright and left-aligned, tracking `0.01em`. Hierarchy comes from size and weight only: 300 for the display temperature and tile numbers, 400 for reading text and stats, 500 for findings and kickers, 600 for annotations, the station name and the current tab. Use tabular digits (`font-variant-numeric: tabular-nums`) where numbers line up: stats, tables, axes, ramp cells and the temperature. Findings and prose use proportional digits. No text is under 12px, chart text included. Phone and desktop sizes are separate styles (`finding`, `finding-phone`); desktop sizes apply from 760px.
- `display` is the current temperature and the number at the head of a zoom. `finding-lead` is the feature tile's finding and each tab's top finding (max 52ch); `finding` is a tab tile's finding (max 44ch), the delta under the temperature and under a zoom's number. `standfirst` is the small reading size in `ink-2`: a mosaic tile's delta, the Close label, and the reading guide the Solar and Rain events tabs keep. `kicker` is a tile's title, a zoom's heading and its panel titles, in `ink-3`. `fact` is the number on a mosaic tile and the feature tile. `stat` is a value over its label, and a word unit after a `fact` number. `source` is the folded source line and the sources footer. `condition` has no current use. `axis`, `label` and `annotation` are chart text; the now label is `annotation` in `now` on a 4px halo in the surface colour.

**Spacing and shape.** Spacing is the `s-1` to `s-8` scale (4, 8, 12, 16, 24, 32, 48, 72px); the tile grid gap is `s-4`. Tiles are `r-tile` (22px, `r-tile-phone` 20px on a phone) with `tile-pad` (22px, 16px on a phone). Event rows and tile links `r-row`, ramp cells `r-cell`; tabs, segments, chips and buttons are `r-pill`. Data dots are circles of radius 4 to 5.5. Strokes: axes and baselines 1px, thin marks 1.5px, the bold now line 3.5px with round caps and joins.

**Layout.** One column on a phone (16px gutter), two columns from `bp-two` (760px), four from `bp-four` (1180px), in a container at most `page-max` (1360px) with a 24px gutter. A tile spans one to four columns. A chart is drawn at its real pixel width and picks its phone layout from its own width under `bp-chart-narrow` (520px), never from the viewport; never shrink a desktop chart to fit a phone. On a phone each tab opens with a short fixed top. The Overview follows it with the feature tile and the mosaic, two tiles to a row (rows of four from 1180px, in the same order), and a tap opens a tile's zoom: full screen on a phone, a full-width row under the tapped tile's row on the grids, at `#overview/<tile>`. The Rain and Solar tabs follow their fixed top with finding rows. The section tabs become a sticky bottom bar padded by `env(safe-area-inset-bottom)`.

**Hatching.** 135 degree stripes in `hatch` (`ink-3` on the now tile) over the surface colour: `hatch-stripe` 2px on a `hatch-repeat` of 12px, and 2px on a 6px repeat in a 12px chip swatch. It is the one visual word for "not valid": an invalid month or window is hatched with a short reason in place and never computed. Text never sits on stripes; a reason sits on a 4px halo in the surface colour, and a hatched chip carries its hatch in a swatch beside its text.

**Borders, shadows, motion, states.** Flat: no shadows, no gradients except the hatch, no blurs. A `box-shadow` appears only as a ring or halo. Focus is a 2px `ink` outline at a 2px offset, on keyboard focus only, following the control's corners; one ring serves every surface. A pressed control or the current tab is an `ink` fill with `ground` text. The bold now line draws in once over 1.1s, skipped under reduced motion. Hover shifts text from `ink-2` to `ink`.

**Graphics and chart grammar.** Every graphic a reader needs (a line, bar, band, dot, hatch, outline, control border) meets 3:1 against what it sits on, in both themes. Charts are SVG with no chart box and no vertical gridlines; horizontal gridlines are `rule`, the baseline `rule-strong`, ticks 12px `ink-3`; labels sit on the marks with a halo in the surface colour; a key appears only where a ramp, the hatch or a band under bars needs decoding; a band under bars is named by a "usual range" legend swatch, never by a label inside the plot. Imagery: none. There are no photos or illustrations.

## Iconography

There is no icon system, no icon font, no emoji and no logo file. Marks are drawn in CSS or SVG from the tokens: chevrons are a rotated 1.5px border (a 6px square) in `ink-2`, in a 24px ring of `rule-strong` on a finding row; dots are circles in `now` (the now dot ending a line) or `rain-bar` (the live rain dot on the feature kicker). The feature tile's chevron is a rotated border in a 32px `rule-strong` ring. The product name is set in plain type, "Lookout", in Lexend at display weight 300; do not draw or approximate a logo. No favicon or app icon is in the source.

## Components

Each component has a live preview in both themes. Hero, FeatureTile, MosaicTile, Zoom and SourcesFooter are the Overview, top to foot; Tile and FindingRow are the Rain, Rain events and Solar tabs' surfaces; AlmanacGrid is a chart; DaysSwitch, Tabs and StationSelector are controls and navigation; GapHatch is the hatching pattern.

- `Hero`: the now tile, the only tile on `now-surface`: the temperature, its delta on usual for this hour, and today against the usual band.
- `FeatureTile`: one full-width link, the Feature Rule's choice: a value and its finding.
- `MosaicTile`: a title, a number, a short delta and an `aria-hidden` mini chart, one link to its zoom; the "usual range" legend swatch.
- `Zoom`: a tile's full view at `#overview/<tile>`, full screen on a phone and a full-width row on the grids, with its heading levels and Close or Back.
- `SourcesFooter`: the Overview's method and provenance, once, ending in the release line.
- `Tile`: a tab tile: kicker, finding, chart, then a folded source.
- `FindingRow`: a closed row on a phone on the Rain and Solar tabs, 56px or more; opening one closes the others.
- `AlmanacGrid`: years by months in the rain ramp, with `≈` prorated, hatched invalid, dashed before the archive and the current month outlined in `now`.
- `DaysSwitch`: the 30 / 90 day segmented control.
- `Tabs`: Overview, Rain, Rain Events and Solar as desktop pills or a phone bottom bar.
- `StationSelector`: the header's station name, a native select when there are two or more stations, and the live line.
- `GapHatch`: the hatch, its reason pill, the hatched chip and swatch.

## Rules that never bend

- All text meets 4.5:1 on whatever it sits on, in light and dark. Text in a ramp cell is chosen per cell.
- Hatch every invalid comparison with its reason in place. Do not smooth over a gap.
- Do not show a forecast or a fabricated value. Do not overprint rain on a band. Do not put the archive lag in a finding. Do not shrink a desktop chart to fit a phone. Do not add shadows, gradients, italics or all-caps. Do not build gauge walls or tile grids of raw numbers: every tile says something.
- Do show a finding's evidence at every size: number, delta and mini chart on a tile; the headline over its full charts in a zoom. Keep the Overview's method in its sources footer, and a tab tile's in its source note. Do mark the current value in `now` with a rule or dot and a label.

## Known contrast exceptions

The light sun ramp steps `sun-2`, `sun-3` and `sun-4` sit under 3:1 against `tile` (2.87, 1.88 and 1.53); the owner accepted this on 2026-11-04 because each cell carries its value as text and its ramp position. The pairs `rule-strong` and `hatch` on `now-surface` are under 3:1 (2.91 in light, 2.27 and 2.05 in dark), which is why the now tile draws its hatch and strong lines in `ink-3`; do not use `rule-strong` or `hatch` there. `ribbon-rain` on `ribbon-night` is under 3:1 in light, but the hero's rain track sits on `now-surface`, and no mark draws on `ribbon-night` now that the 48-hour ribbon is gone.
