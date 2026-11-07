Lookout is one weather station's own record. Every block says one thing in plain words with the number in it, then shows the evidence: a finding, a one-line standfirst on how to read the chart, the chart, and where the numbers came from. The posture is the Financial Times data desk; the presentation is a Dark Sky style current reading and 48-hour ribbon. It is a dashboard for the owner of a few Ambient Weather stations, read mostly on a phone in short glances, and by friends through a shared link.

Start from `tokens.json` for values and from the components for markup. Dark is the primary design and the first theme; light is its twin, held to the same contrast rules.

## Content fundamentals

- **Findings first.** A tile opens with a sentence that states what the data shows, with the number in it: "It has rained for 3 hours, and 0.42 in has fallen." It never describes the chart ("A chart of rain"). Set it in `finding` (`finding-lead` for the lead).
- **Headlines speak, evidence stays exact.** A finding says a duration the way a person does: `45 minutes`, `about 26 hours`, `about 3.5 days`, `about 12 days`; `about` marks a rounded value. Stat rows, chart labels, the ribbon, event rows and source lines keep the exact form: times `9:36 am`, dates with times `Oct 27, 11:00 pm`, durations `1 d 10 h`, `5 h 12 m`, `45 m`.
- **Units and zone.** Imperial: °F, mph, inches of rain, inHg, W/m², kWh/m², the UV index. Time is the station's own zone, never the viewer's.
- **Sentence case, left-aligned, no italics, no all-caps runs, no emoji.** Emphasis is weight. Kickers name the subject in sentence case ("Rain", "Sun and moon").
- **The tone is plain and measured.** Real copy: "Tue Nov 3, 7:40 am · 3 m ago", "No rain events in <water year> yet.", "Couldn't load this event's curve.", "Loading the curve…", "the bold line", "the band", "the rule marked now".
- **Name marks by label or position, never by colour.** "the bold line", not "the blue line".
- **Name the years, newest first.** A finding that compares years says which: "less than the same days of 2025 and 2024".
- **No forecasts and no fabricated values.** Lookout shows measured data only and says so. A weather app does forecasts.
- **Use the project's terms exactly.** station, station selector, snapshot, archive, gap status (valid, prorated, invalid), lead, Event Catalog, water year. Say "station", not "device".
- **Archive lag lives in the source note, never in a finding.** A live value beats an archived one.
- **Python decides.** Findings, short forms, the lead, gap status and every comparison are decided before the page and arrive in the snapshot. The site draws what the snapshot says; do not recompute a number in the UI. Each fact is computed once and shows the same number everywhere (the One Number Rule).

## Visual foundations

**Colour.** The chrome is near-greyscale slate; colour is kept for data. Rain is blue (`rain-*`), sun is gold (`sun-*`), temperature runs cool blue to warm orange (`temp-*`), and one orange, `now`, marks the subject of the finding. Every saturated colour on screen is a measurement, a usual band or `now`.
- Page behind everything: `ground`. Tiles and event rows: `tile` on `ground`. The one tinted tile, the current reading: `now-surface`. There are no shadows; depth is tile on ground.
- Text: `ink` for headlines and values, `ink-2` for standfirsts and labels, `ink-3` for kickers, axis ticks and source lines. Each passes 4.5:1 on `ground`, `tile` and `now-surface` in both themes. On `now-surface` secondary text may use `ink-on-now`.
- `rule` is for dividers and gridlines only (it is decorative and under 3:1 on purpose). Anything a reader must see, such as a control border, a chart baseline, a chip or a chevron ring, uses `rule-strong`.
- `now` is one orange subject per tile: the now rule, now dot and label, the dot over today's bar, the current month or hour outline, the latest chip, the leading tile's dot. It never touches a data fill: it sits on `tile`, `now-surface` or `ground`, behind a gap in the surface colour (the ribbon's now dot has a 2px `now-surface` ring; an outline round a ramp cell sits outside it with a 1px gap). Do not use `now` for chrome or for anything else.
- A usual band (`rain-band`, `temp-band`, `sun-band`) covers the middle half of the comparison set, with a dashed `median`. It passes 3:1 by its fill in dark and by its `*-band-edge` in light.
- Ramps (`rain-0` to `rain-4`, `sun-0` to `sun-4`): in light, rain runs pale to deep; in dark, deeper means lighter, so more rain is always the more visible cell. Text in a ramp cell uses the matching `on-rain-N` or `on-sun-N`.
- Colours that must be told apart also differ in lightness. Rain intensity on the ribbon is bar height, not shade. Today's bar keeps its data colour and is marked by the `now` dot and the word "today".
- Follow the OS colour scheme. `?theme=light` or `?theme=dark` in the address overrides it for that load; nothing is stored and there is no theme control.

**Type.** One family, Lexend (self-hosted on the site, with `system-ui, sans-serif` behind it), upright and left-aligned, tracking `0.01em`. Hierarchy comes from size and weight only: 300 for the display temperature and facts, 400 for reading text and stats, 500 for findings and kickers, 600 for annotations, the station name and the current tab. Use tabular digits (`font-variant-numeric: tabular-nums`) where numbers line up: stats, tables, axes, ramp cells, the ribbon and the temperature. Findings and prose use proportional digits. No text is under 12px, chart text included. Phone and desktop sizes are separate styles (`finding`, `finding-phone`); desktop sizes apply from 760px.
- `display` is the current temperature only. `finding-lead` is the leading tile's finding and each tab's top finding (max 52ch); `finding` is a tile's finding (max 44ch). `standfirst` is one line on how to read the chart, in `ink-2` (max 62ch). `kicker` sits above the finding in `ink-3`. `fact` is the one large value on a quick-fact tile. `stat` is a value over its label. `source` is the folded source line. `axis`, `label` and `annotation` are chart text; the now label is `annotation` in `now` on a 4px halo in the surface colour.

**Spacing and shape.** Spacing is the `s-1` to `s-8` scale (4, 8, 12, 16, 24, 32, 48, 72px); the tile grid gap is `s-4`. Tiles are `r-tile` (22px, `r-tile-phone` 20px on a phone) with `tile-pad` (22px, 16px on a phone). Event rows `r-row`, ramp cells `r-cell`, the ribbon `r-ribbon`; tabs, segments, chips and buttons are `r-pill`. Data dots are circles of radius 4 to 5.5. Strokes: axes and baselines 1px, thin marks 1.5px, the bold now line 3.5px with round caps and joins.

**Layout.** One column on a phone (16px gutter), two columns from `bp-two` (760px), four from `bp-four` (1180px), in a container at most `page-max` (1360px) with a 24px gutter. A tile spans one to four columns. A chart is drawn at its real pixel width and picks its phone layout from its own width under `bp-chart-narrow` (520px), never from the viewport; never shrink a desktop chart to fit a phone. On a phone each tab opens with a short fixed top and then an accordion of finding rows; the section tabs become a sticky bottom bar padded by `env(safe-area-inset-bottom)`.

**Hatching.** 135 degree stripes in `hatch` (`ink-3` on the now tile) over the surface colour: `hatch-stripe` 2px on a `hatch-repeat` of 12px, and 2px on a 6px repeat in a 12px chip swatch. It is the one visual word for "not valid": an invalid month or window is hatched with a short reason in place and never computed. Text never sits on stripes; a reason sits on a 4px halo in the surface colour, and a hatched chip carries its hatch in a swatch beside its text.

**Borders, shadows, motion, states.** Flat: no shadows, no gradients except the hatch, no blurs. A `box-shadow` appears only as a ring or halo. Focus is a 2px `ink` outline at a 2px offset, on keyboard focus only, following the control's corners; one ring serves every surface. A pressed control or the current tab is an `ink` fill with `ground` text. The bold now line draws in once over 1.1s, skipped under reduced motion. Hover shifts text from `ink-2` to `ink`.

**Graphics and chart grammar.** Every graphic a reader needs (a line, bar, band, dot, hatch, outline, control border) meets 3:1 against what it sits on, in both themes. Charts are SVG with no chart box and no vertical gridlines; horizontal gridlines are `rule`, the baseline `rule-strong`, ticks 12px `ink-3`; labels sit on the marks with a halo in the surface colour; a key appears only where a ramp or the hatch needs decoding. Imagery: none. There are no photos or illustrations.

## Iconography

There is no icon system, no icon font, no emoji and no logo file. Marks are drawn in CSS or SVG from the tokens: chevrons are a rotated 1.5px border (a 6px square) in `ink-2`, in a 24px ring of `rule-strong` on a finding row; dots are circles in `now` (8px leading dot, 9px ribbon dot) or `rain-bar` (the live dot). The product name is set in plain type, "Lookout", in Lexend at display weight 300; do not draw or approximate a logo. No favicon or app icon is in the source.

## Components

Each component has a live preview in both themes. Tile and FindingRow are the surfaces; QuickFacts, Hero, Ribbon and AlmanacGrid are the overview's blocks; DaysSwitch, Tabs and StationSelector are controls and navigation; GapHatch is the hatching pattern.

- `Tile`: kicker, finding, standfirst, chart, then a folded source; and the now tile, the only tile on `now-surface`.
- `FindingRow`: a closed accordion row on a phone, 56px or more; opening one closes the others.
- `QuickFacts`: wind, pressure, air, sun and moon, each a kicker, one `fact` value, one sentence.
- `Hero`: the leading tile with its stat strip, its bold now line over a usual band, and "Why this leads".
- `Ribbon`: the 48-hour night and day band, the temperature trace, and a separate rain track.
- `AlmanacGrid`: years by months in the rain ramp, with `≈` prorated, hatched invalid, dashed before the archive and the current month outlined in `now`.
- `DaysSwitch`: the 30 / 90 day segmented control.
- `Tabs`: Overview, Rain, Rain Events and Solar as desktop pills or a phone bottom bar.
- `StationSelector`: the header's station name, a native select when there are two or more stations, and the live line.
- `GapHatch`: the hatch, its reason pill, the hatched chip and swatch.

## Rules that never bend

- All text meets 4.5:1 on whatever it sits on, in light and dark. Text in a ramp cell is chosen per cell.
- Hatch every invalid comparison with its reason in place. Do not smooth over a gap.
- Do not show a forecast or a fabricated value. Do not overprint rain on the ribbon's day and night bands. Do not put the archive lag in a finding. Do not shrink a desktop chart to fit a phone. Do not add shadows, gradients, italics or all-caps. Do not build gauge walls or tile grids of raw numbers: every tile says something.
- Do start every tile with its finding and fold method and provenance into a one-line source note. Do mark the current value in `now` with a rule or dot and a label.

## Known contrast exceptions

The light sun ramp steps `sun-2`, `sun-3` and `sun-4` sit under 3:1 against `tile` (2.87, 1.88 and 1.53); the owner accepted this on 2026-11-04 because each cell carries its value as text and its ramp position. The pairs `rule-strong` and `hatch` on `now-surface` are under 3:1 (2.91 in light, 2.27 and 2.05 in dark), which is why the now tile draws its hatch and strong lines in `ink-3`; do not use `rule-strong` or `hatch` there. `ribbon-rain` on `ribbon-night` is under 3:1 in light, but the rain track sits on `now-surface`, not on the band.
