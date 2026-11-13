---
name: Lookout
description: A personal weather record for one station. It reads like Dark Sky and argues like the FT. Findings first, on quiet slate tiles, in both themes.
colors:
  # Light values are the base keys; each -dark key is the same role in dark mode.
  # Dark is the primary design; light is its twin, held to the same contrast rules.
  ground: "#f3f5f7"
  tile: "#ffffff"
  now-surface: "#e2e9f0"
  ink: "#11161c"
  ink-2: "#353e48"
  ink-3: "#4d5763"
  ink-on-now: "#2c353f"
  rule: "#e1e5e9"
  rule-strong: "#7d8995"
  hatch: "#7d8995"
  now: "#a74b11"
  median: "#11161c"
  # The two inks for text in a ramp cell, the same in both themes.
  on-ramp-pale: "#ffffff"
  on-ramp-deep: "#11161c"
  rain-line: "#1d4a7a"
  rain-bar: "#3b6fa6"
  rain-band: "#dce7f3"
  rain-band-edge: "#6f8fb3"
  rain-0: "#6a8fb7"
  rain-1: "#52789f"
  rain-2: "#3d6289"
  rain-3: "#2a4b6f"
  rain-4: "#1a3352"
  temp-band: "#e8ecf0"
  temp-band-edge: "#7d8995"
  temp-cool: "#3f6890"
  temp-mild: "#5e6e66"
  temp-warm: "#b5602a"
  temp-iqr: "#5a7da1"
  sun-line: "#8a6413"
  sun-bar: "#a87a24"
  sun-band: "#f5e9c8"
  sun-band-edge: "#a1802f"
  sun-0: "#6e5a2e"
  sun-1: "#8a6f2e"
  sun-2: "#b8942c"
  sun-3: "#e3b83a"
  sun-4: "#f5cd4f"
  ribbon-night: "#2c3640"
  ribbon-day: "#8797a8"
  ribbon-rain: "#2f5f94"
  ribbon-edge: "#5d6976"
  ground-dark: "#11161c"
  tile-dark: "#1a2129"
  now-surface-dark: "#354251"
  ink-dark: "#eef2f5"
  ink-2-dark: "#c3ccd5"
  ink-3-dark: "#aab5c0"
  ink-on-now-dark: "#dfe6ec"
  rule-dark: "#2a333d"
  rule-strong-dark: "#6b7886"
  hatch-dark: "#64717e"
  now-dark: "#f2a65e"
  median-dark: "#11161c"
  on-ramp-pale-dark: "#ffffff"
  on-ramp-deep-dark: "#11161c"
  rain-line-dark: "#d6e9fb"
  rain-bar-dark: "#7fb0e0"
  rain-band-dark: "#55779c"
  rain-band-edge-dark: "#55779c"
  rain-0-dark: "#4f7299"
  rain-1-dark: "#5f86b0"
  rain-2-dark: "#7aa3cc"
  rain-3-dark: "#9fc2e4"
  rain-4-dark: "#cfe3f6"
  temp-band-dark: "#606d7a"
  temp-band-edge-dark: "#606d7a"
  temp-cool-dark: "#b4cde6"
  temp-mild-dark: "#d3dfd8"
  temp-warm-dark: "#f5bd8a"
  temp-iqr-dark: "#9fb9d3"
  sun-line-dark: "#f6cf74"
  sun-bar-dark: "#c99d45"
  sun-band-dark: "#7d6b40"
  sun-band-edge-dark: "#7d6b40"
  sun-0-dark: "#7d6b40"
  sun-1-dark: "#a88b45"
  sun-2-dark: "#c99d45"
  sun-3-dark: "#e3b75a"
  sun-4-dark: "#f7d98f"
  ribbon-night-dark: "#1c242d"
  ribbon-day-dark: "#8a9aab"
  ribbon-rain-dark: "#a9d0f5"
  ribbon-edge-dark: "#8a9aab"
typography:
  # Base steps are the phone sizes; each -desktop step is the same role from 760px.
  display:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "clamp(5.25rem, 22vw, 7.5rem)"
    fontWeight: 300
    lineHeight: 0.88
    letterSpacing: "-0.05em"
    fontFeature: "\"tnum\" 1"
  finding-lead:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 500
    lineHeight: 1.33
    letterSpacing: "0.01em"
  finding-lead-desktop:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 500
    lineHeight: 1.33
    letterSpacing: "0.01em"
  finding:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0.01em"
  finding-desktop:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "1.1875rem"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0.01em"
  condition:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "0.01em"
  condition-desktop:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "0.01em"
  fact:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "2rem"
    fontWeight: 300
    lineHeight: 1.1
    fontFeature: "\"tnum\" 1"
  fact-desktop:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "2.5rem"
    fontWeight: 300
    lineHeight: 1.1
    fontFeature: "\"tnum\" 1"
  standfirst:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: "0.01em"
  kicker:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 500
    lineHeight: 1.3
    letterSpacing: "0.01em"
  stat:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 400
    lineHeight: 1.25
    fontFeature: "\"tnum\" 1"
  stat-desktop:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "1.375rem"
    fontWeight: 400
    lineHeight: 1.25
    fontFeature: "\"tnum\" 1"
  source:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.55
  axis:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    fontFeature: "\"tnum\" 1"
  annotation:
    fontFamily: "Lexend, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 600
rounded:
  cell: "3px"
  ribbon: "6px"
  row: "16px"
  tile-phone: "20px"
  tile: "22px"
  pill: "999px"
spacing:
  s-1: "4px"
  s-2: "8px"
  s-3: "12px"
  s-4: "16px"
  s-5: "24px"
  s-6: "32px"
  s-7: "48px"
  s-8: "72px"
components:
  tile:
    backgroundColor: "{colors.tile}"
    rounded: "{rounded.tile}"
    padding: "22px"
  tile-phone:
    backgroundColor: "{colors.tile}"
    rounded: "{rounded.tile-phone}"
    padding: "16px"
  now-tile:
    backgroundColor: "{colors.now-surface}"
    textColor: "{colors.ink}"
  tab-pill:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.pill}"
    height: "40px"
    padding: "0 16px"
  tab-pill-current:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
  tab-bar-phone:
    backgroundColor: "{colors.ground}"
    height: "48px"
    rounded: "14px"
  segment:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.pill}"
    height: "44px"
    padding: "0 14px"
  segment-pressed:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
  finding-row:
    height: "56px"
  event-row:
    backgroundColor: "{colors.tile}"
    rounded: "{rounded.row}"
    height: "64px"
  quality-chip:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.pill}"
    padding: "0 7px"
  hatched-chip:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.pill}"
    padding: "0 7px"
    swatch: "12px"
  station-select:
    textColor: "{colors.ink}"
    maxWidth: "28ch"
  focus-ring:
    outlineColor: "{colors.ink}"
    outlineWidth: "2px"
    outlineOffset: "2px"
  size-bar:
    backgroundColor: "{colors.rain-bar}"
    trackColor: "{colors.rule}"
    height: "4px"
    rounded: "{rounded.pill}"
# The Contrast table, machine-readable. The token contrast test reads this list.
# Each entry: fg (a token, or a list meaning "the better of these"), bg (a token),
# min (4.5 for text, 3 for graphics), and themes. A token resolves to its base key
# in light and its -dark key in dark.
contrast:
  # Text, 4.5:1. Chrome text on every surface it sits on.
  - {fg: ink, bg: ground, min: 4.5, themes: [light, dark]}
  - {fg: ink, bg: tile, min: 4.5, themes: [light, dark]}
  - {fg: ink, bg: now-surface, min: 4.5, themes: [light, dark]}
  - {fg: ink-2, bg: ground, min: 4.5, themes: [light, dark]}
  - {fg: ink-2, bg: tile, min: 4.5, themes: [light, dark]}
  - {fg: ink-2, bg: now-surface, min: 4.5, themes: [light, dark]}
  - {fg: ink-3, bg: ground, min: 4.5, themes: [light, dark]}
  - {fg: ink-3, bg: tile, min: 4.5, themes: [light, dark]}
  - {fg: ink-3, bg: now-surface, min: 4.5, themes: [light, dark]}
  - {fg: ink-on-now, bg: now-surface, min: 4.5, themes: [light, dark]}
  # A pressed control: ground text on an ink fill.
  - {fg: ground, bg: ink, min: 4.5, themes: [light, dark]}
  # Now orange. Its label is text, on each surface a now mark may sit on.
  - {fg: now, bg: ground, min: 4.5, themes: [light, dark]}
  - {fg: now, bg: tile, min: 4.5, themes: [light, dark]}
  - {fg: now, bg: now-surface, min: 4.5, themes: [light, dark]}
  # Text in ramp cells: the better of the two cell inks, per cell.
  - {fg: [on-ramp-pale, on-ramp-deep], bg: rain-0, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: rain-1, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: rain-2, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: rain-3, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: rain-4, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: sun-0, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: sun-1, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: sun-2, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: sun-3, min: 4.5, themes: [light, dark]}
  - {fg: [on-ramp-pale, on-ramp-deep], bg: sun-4, min: 4.5, themes: [light, dark]}
  # Graphics, 3:1. Lines and hatching a reader must see, on each surface.
  - {fg: rule-strong, bg: ground, min: 3, themes: [light, dark]}
  - {fg: rule-strong, bg: tile, min: 3, themes: [light, dark]}
  - {fg: hatch, bg: ground, min: 3, themes: [light, dark]}
  - {fg: hatch, bg: tile, min: 3, themes: [light, dark]}
  # On the now tile, hatching and strong lines draw in ink-3.
  - {fg: ink-3, bg: now-surface, min: 3, themes: [light, dark]}
  # The event size bar on its track.
  - {fg: rain-bar, bg: rule, min: 3, themes: [light, dark]}
  # Usual bands: by fill in dark, by edge in light (each -dark edge equals its fill).
  - {fg: rain-band-edge, bg: tile, min: 3, themes: [light, dark]}
  - {fg: temp-band-edge, bg: tile, min: 3, themes: [light, dark]}
  - {fg: sun-band-edge, bg: tile, min: 3, themes: [light, dark]}
  - {fg: median, bg: rain-band, min: 3, themes: [light, dark]}
  - {fg: median, bg: temp-band, min: 3, themes: [light, dark]}
  - {fg: median, bg: sun-band, min: 3, themes: [light, dark]}
  # Data marks, on the tile and across their band.
  - {fg: rain-line, bg: tile, min: 3, themes: [light, dark]}
  - {fg: rain-line, bg: rain-band, min: 3, themes: [light, dark]}
  - {fg: rain-bar, bg: tile, min: 3, themes: [light, dark]}
  - {fg: temp-cool, bg: temp-band, min: 3, themes: [light, dark]}
  - {fg: temp-mild, bg: temp-band, min: 3, themes: [light, dark]}
  - {fg: temp-warm, bg: temp-band, min: 3, themes: [light, dark]}
  - {fg: temp-iqr, bg: tile, min: 3, themes: [light, dark]}
  - {fg: sun-line, bg: tile, min: 3, themes: [light, dark]}
  - {fg: sun-line, bg: sun-band, min: 3, themes: [light, dark]}
  - {fg: sun-bar, bg: tile, min: 3, themes: [light, dark]}
  # Ramp step 0 on the tile, and on the rain-by-hour grid's ground background.
  - {fg: rain-0, bg: tile, min: 3, themes: [light, dark]}
  - {fg: rain-0, bg: ground, min: 3, themes: [light, dark]}
  - {fg: sun-0, bg: tile, min: 3, themes: [light, dark]}
  # The ribbon tokens. The 48-hour ribbon left the now tile with the hero (#468);
  # its pairs stay while the tokens do, and ribbon-rain draws the hero's rain bars.
  - {fg: ribbon-night, bg: ribbon-day, min: 3, themes: [light, dark]}
  - {fg: ribbon-edge, bg: now-surface, min: 3, themes: [light, dark]}
  - {fg: ribbon-rain, bg: now-surface, min: 3, themes: [light, dark]}
  # The usual band hero, on the now tile (#468): the median in ink-on-now and
  # today's line in ink, each across the band; its rain bars are ribbon-rain.
  - {fg: ink-on-now, bg: temp-band, min: 3, themes: [light, dark]}
  - {fg: ink, bg: temp-band, min: 3, themes: [light, dark]}
---

# Design System: Lookout

## Overview

**Creative North Star: "Tiles That Talk"**

Lookout is one station's own record. Every block of the page says one thing, with the number in it, and shows the evidence, sized to the space it has. The posture is the Financial Times data desk: an open view leads with a number-led finding over its chart, and says where the numbers came from. The presentation is Dark Sky: a large current temperature, today drawn against its usual, and colour that comes from the weather.

The page is a grid of quiet slate tiles on a darker ground (or white tiles on a pale ground in light mode). The Overview is visual first: the now tile (today against the usual band), one feature tile, then a mosaic of small tiles, each a number, a short delta and a mini chart, each one link to its zoom. Method and provenance for the whole Overview sit once, in the sources footer at its foot. The Rain, Rain events and Solar tabs keep their tiles of finding, chart and folded source note, and on a phone their finding rows.

Colour is kept for data. Rain is blue, sun is gold, temperature runs cool blue to warm orange, and one orange marks "now". The chrome is near-greyscale slate. One readable family, Lexend, sets every word and number. The system is built for dyslexic readers: text is left-aligned, nothing is italic, there are no all-caps runs, line height is generous, measures are short, and no text is under 12px.

The record is honest about itself. Gaps are hatched, never smoothed over. A live value beats an archived one. Each fact is computed once and agrees everywhere it appears. There are no forecasts: a weather app does that, and Lookout says so.

**Key Characteristics:**
- An Overview tile is a number, a short delta and a mini chart; its zoom carries the finding headline and the full charts.
- A tab tile (Rain, Rain events, Solar) runs kicker, finding, chart, then folded source.
- Slate chrome; data colour carries all the hue; one orange "now".
- One family, Lexend, for text and numerals; tabular digits where numbers line up.
- Hairline axes, direct labels, charts drawn at their real width.
- Every graphic that carries meaning meets 3:1; all text meets 4.5:1, in both themes.
- Hatching is the one visual word for "not valid".
- Dark is the primary design; light is its twin. The OS chooses; an explicit `?theme=` override is honoured for that page load.

## Colors

Light values are the frontmatter base keys and each `-dark` key is the same role in dark mode. Dark is the reference design; every light value was derived to pass the same checks (see Contrast).

### Chrome
- **Ground** (`ground`): the page behind the tiles.
- **Tile** (`tile`): every tile and event row. Depth comes from tile on ground, never from shadow.
- **Now surface** (`now-surface`): the one tinted tile, the current reading. Text on it uses `ink` and `ink-on-now`. Hatching and strong lines on it draw in `ink-3`, because `hatch` and `rule-strong` fall under 3:1 there.
- **Ink** (`ink`), **Soft Ink** (`ink-2`), **Faint Ink** (`ink-3`): headlines and values; deltas, labels and in-chart labels; kickers, tile titles, axis ticks and source lines. All three meet 4.5:1 on `tile` and `ground`.
- **Hairline** (`rule`): dividers and gridlines only. It is decorative and carries no meaning, so it sits under 3:1 on purpose.
- **Hairline Strong** (`rule-strong`): every border or line a reader must see: control borders, chart baselines, dashed "not yet" outlines, chip borders, chevron rings. 3:1 or better.

### Data
- **Rain** (`rain-line`, `rain-bar`, `rain-band`, `rain-0` to `rain-4`): the bold now-curve of a storm or season, daily bars, the usual band, and the five-step ramp for the almanac and the rain-by-hour grid. The ramp gets deeper with more rain; in dark mode deeper means lighter.
- **Temperature** (`temp-cool`, `temp-mild`, `temp-warm`, `temp-iqr`, `temp-band`): the ten-day bars (cool when the high stayed under usual, warm when over, mild when level), the middle-half bar in the periods chart, and the usual band behind each day.
- **Sun** (`sun-line`, `sun-bar`, `sun-band`, `sun-0` to `sun-4`): the solar trace, daily and quarter-hour bars, the usual band, and the solar almanac ramp. The ramp runs dim to bright with the sun in both themes: the brightest day is the brightest cell. Named exception: the owner accepted on 2026-11-04 that the light ramp's steps 2 to 4 (`sun-2` 2.87, `sun-3` 1.88, `sun-4` 1.53 on `tile`) sit under 3:1, because the grid reads as a spectrum in context; the cells carry no edge.
- **Ribbon** (`ribbon-night`, `ribbon-day`, `ribbon-rain`, `ribbon-edge`): `ribbon-rain` draws the rain track along the foot of the hero. The 48-hour ribbon that used the other three is gone from the site; they stay in the token set.
- **Median** (`median`): the dashed median drawn over a usual band, in both themes.

### The finding
- **Now Orange** (`now`): the now rule, the now dot and its label, the dot over today's bar, the current month's and current hour's outlines, the "latest" chip, and the now dot ending a line on a mini chart. It marks the thing the headline is about and nothing else. Orange replaces the earlier claret.

### Named Rules
**The Data Owns the Hue Rule.** Chrome is slate neutral. Every saturated colour on screen is a measurement, a usual band, or the one orange "now".

**The One Now Rule.** `now` marks the current value or the current item that the finding is about. A tile has one orange subject.

**The AA Both Ways Rule.** All text meets 4.5:1 on whatever it sits on, in light and dark. Text inside a ramp cell is chosen per cell, `on-ramp-pale` or `on-ramp-deep`, whichever passes. A pressed control is an `ink` fill with `ground` text.

**The Three-to-One Graphics Rule.** Any graphic a reader needs to understand the chart (a line, bar, band, dot, hatch, outline or control border) meets 3:1 against what it sits on (WCAG 1.4.11). A band meets it in one of two ways: by its fill (dark mode) or by a 3:1 edge around a pale fill (light mode, `*-band-edge`). A bold line also meets 3:1 against the band it crosses. Decorative hairlines are exempt and use `rule`.

**The Hue Is Never Alone Rule.** Colours that must be told apart also differ in lightness, and the copy names marks by label or position. Rain intensity on the hero's rain track is bar height, not shade. Today's bar keeps its data colour and is told apart by the `now` dot over it and the label "today", never by an orange fill.

**The Now Stands Off Rule.** A `now` mark never touches a data fill. It sits on `tile`, `now-surface` or `ground`, or behind a gap in the colour of the surface under it: the hero's now label sits on a 4px `now-surface` halo, and an outline around a ramp cell sits outside the cell with a 1px gap. So `now` is only ever checked against the three surfaces, where it passes as text.

## Typography

**Family:** Lexend, self-hosted (the CSP allows no third-party font origin), with `system-ui, sans-serif` behind it. Lexend replaces Atkinson Hyperlegible Next; it keeps the readability goal and gives the large light numerals their character.

**Character:** Hierarchy comes from size and weight only. Weights are 300 for the display temperature, 400 for reading text and stats, 500 for findings and kickers, and 600 for annotations, the station name and the current tab. Tracking is slightly open (0.01em).

### Hierarchy
The frontmatter carries each step: the base key is the phone size and the `-desktop` key is the size from 760px (`finding-lead-desktop`, `finding-desktop`, `condition-desktop`, `stat-desktop`, `fact-desktop`). A step with no `-desktop` key is the same on both.

- **Display** (300, 7.5rem desktop, 5.25rem phone, 0.88, -0.05em, tabular): the current temperature, and the number at the head of a zoom.
- **Finding, feature** (500, 1.5rem desktop, 1.25rem phone, 1.33, max 52ch; key `finding-lead`): the feature tile's finding and each tab's top finding.
- **Finding** (500, 1.1875rem desktop, 1rem phone, 1.4, max 44ch): a tab tile's finding, the label of a closed finding row, the delta on usual under the current temperature ("−1° on usual for 11 pm", tabular, `ink-2`), and the delta under a zoom's number.
- **Condition** (400, 1.25rem desktop, 1rem phone, 1.45): no current use. The condition sentence it set left the now tile with the hero; the step stays in the token set.
- **Small** (400, 0.875rem, 1.55, `ink-2`, max 62ch; key `standfirst`): a mosaic tile's delta, the Close label, and the reading lines the Solar and Rain events tabs keep.
- **Kicker** (500, 0.8125rem, `ink-3`): a tile's title or subject, sentence case; a zoom's heading and its panel titles.
- **Stat** (400, 1.375rem desktop, 1.25rem phone, tabular): values in a stat strip, over their label; a word unit after a Fact number ("mph", "inHg").
- **Fact** (300, 2.5rem desktop, 2rem phone, 1.1, tabular): the number on a mosaic tile and on the feature tile. Under 360px of viewport a mosaic number drops to Stat, so a value in words keeps to its half-width tile.
- **Source** (400, 0.8125rem, `ink-3`): the folded source line and its body on the Rain, Rain events and Solar tabs, and the Overview's sources footer.
- **Axis** (400, 12px, `ink-3`, tabular) and **in-chart label** (12px, `ink-2`): chart text.
- **Annotation** (600, 13px, `now`, with a 4px tile-coloured halo): the now label.

### Named Rules
**The No Italic, No Caps Rule.** Every word is Lexend, upright and left-aligned. Emphasis is weight. No italics, no centred paragraphs, no all-caps runs.

**The Twelve Pixel Floor.** No text renders under 12px, chart text included. A chart is drawn at its real pixel width (production already does), so a phone gets a phone layout of the chart, never a shrunken desktop one.

**The Chart Width Rule.** A chart picks its phone layout from its own width, not the viewport: under 520px of chart width it draws its phone layout. The page layout switches at 760px of viewport; a chart does not follow it, because a one-column desktop tile at 1180px is about 310px wide and needs the phone layout too.

**The Tabular Where It Lines Up Rule.** Tabular digits in stats, tables, axes, ramp cells, the hero's axis and the temperature; proportional digits in findings and prose.

## Layout

**Desktop and tablet (760px and up):** a grid of tiles in a container at most 1360px wide with a 24px gutter. Four columns from 1180px, two from 760px. Tiles span one, two, three or four columns. Grid gap 16px. Tile padding 22px, corners 22px. The header is the station name, the live line and the section tabs as pills.

**Phone (under 760px):** one column, 16px gutter, tile padding 16px, corners 20px. The section tabs move to a bottom tab bar. Each tab opens with a short fixed top, the Phone First-Screen Rule below.

The layout switches on the root's `data-layout` (`phone`, `two`, `four`), set from the viewport at 760px and 1180px, so the grid and the zoom read one answer.

**The Phone First-Screen Rule.** On a phone, each tab opens with a short fixed top, then its context:
- **Overview:** the now tile (the current temperature, its delta on usual for this hour, and today against the usual band), then the feature tile, full width, then the mosaic, two tiles to a row. Nothing on the Overview folds: the context is visible without a tap, and a tap zooms a tile. The sources footer ends the page.
- **Rain, Solar:** the tab's top finding with its stats and its main chart, always open; the tab's other sections as finding rows.
- **Rain events:** see Event list.

**The Overview grid.** The now tile and the feature tile each take a full row; the mosaic tiles follow in a fixed order, one column each: Temperature, Rain since Oct 1, Last 10 days, Dry spell, Wind, Pressure, Humidity, Sun today, Daylight, Inside. A phone sets them two to a row; two columns from 760px; four from 1180px, where they run in rows of four in the phone's order (Temperature to Dry spell, Wind to Sun today, Daylight and Inside with two cells empty). Nothing reorders between phone and desktop. On the Overview and in its zooms a hatch only marks a gap inside a period the record covers, so its reason reads without "n/a" ("gap in record", "gap" in a cell); a period the record does not cover yet is not drawn (#470). A tile the snapshot says is not shown (`record_depth.tiles`: the record does not cover it yet; Inside with no indoor sensor; Daylight with no location) is not drawn, and the later tiles move up.

**The Finding Row (Rain and Solar, phone).** A closed row is its kicker over its finding sentence, with a chevron in a `rule-strong` ring, at least 56px tall. Closed, the rows read as a list of findings, like the FT's key points. Opening a row closes any other row in the same group; the opened row shows its chart and folded source. The Overview has no finding rows.

**The Feature Rule.** One full-width feature tile sits under the now tile. The refresh run chooses it (ADR-0009) and the snapshot carries the choice (`feature`); the site only words, formats and links it. The rule, checked in order: a rain event live or ended in the last 12 hours; then a reading below the 10th or above the 90th percentile of its readings at this hour within a week of this date in earlier years, the furthest out winning (skipped on a station with no earlier year); then a dry spell longer than the usual dry spell; otherwise "a usual day", with today's temperature. A snapshot with no `feature` draws no feature tile. The sources footer states the rule in words.

## Elevation & Depth

Flat. No shadows. Depth is tile on ground, and the one tinted now tile. Inside charts the layering is: usual band at the back, hatching over invalid spans, the bold now mark on top, labels haloed in the tile colour.

## Shapes

Tiles 22px (20px on a phone). Event rows 16px. Pills for tabs, segments, chips and buttons. Ramp cells 3px. The `ribbon` radius (6px) has no current use; it stays in the set with the ribbon tokens. Data dots are circles (r 4 to 5.5). Strokes: axes and baselines 1px, thin marks 1.5px, the bold now line 3.5px with round caps and joins. Hatching is always 135° stripes, `hatch` over the tile colour (`ink-3` over the now surface), 2px on a 12px repeat (2px on a 6px repeat in a chip swatch, which is 12px square). Text never sits on the stripes: a reason over a hatched span sits on a 4px halo in the surface colour, and a hatched chip carries its hatch in a swatch beside its text.

## Components

### Header and tabs
- **Desktop:** station name (600), live dot and line ("Tue Nov 3, 7:40 am · 3 m ago", `ink-3`), and the tabs as pills on the right: 40px tall, `ink-2`; the current tab is an `ink` pill with `ground` text.
- **Phone:** a 52px header (the station, and the live line), and a bottom tab bar, sticky, four equal 48px targets; the current tab is an `ink` fill with `ground` text. The bar is padded below by `env(safe-area-inset-bottom)`, so its targets clear the home indicator; the page sets `viewport-fit=cover`. The bottom bar names the tab, so the header always names the station.
- **Station selector:** with two or more stations in the station index, the station name is the selector, in the same slot on desktop and phone. It is a native `select` styled as the 600-weight name with a chevron after it, so the header looks the same with one station or many. With one station the name is plain text and has no chevron. A long name stays on one line and ends in an ellipsis: at most 28ch on desktop; on a phone it takes the header width the live line leaves, and the live age never truncates. The open list shows every name in full, and an unnamed station shows the end of its MAC.
- Staleness: past 15 minutes, the live dot goes hollow and the age sits in a hatched chip.
- **Battery line:** under the freshness line, one line per sensor whose battery reads low ("Rain gauge battery low"), in ink, never orange. Python decides low and names the sensor; with every flag OK there is no line.

### Focus
Every control shows `:focus-visible` as a 2px `ink` outline at a 2px offset, following the control's corners. `ink` passes on `ground`, `tile` and `now-surface`, so one ring serves every surface. Focus from a pointer shows no ring.

### Theme choice
The OS colour scheme picks the theme. An explicit `?theme=light` or `?theme=dark` in the page address overrides it for that page load and stores nothing, so a shared link can fix the theme. There is no theme control on the page.

### Now tile
The Overview's fixed top, labelled "Right now", the only tile on `now-surface`. The live temperature in Display with its "°F", the one-line delta on usual for this hour under it, and the hero chart: today hour by hour against the usual band for this date (see Chart grammar). On a phone the temperature sits above the chart; on the grids, beside it.

### Feature tile
One link filling a full-width tile: the kicker with a chevron in a 32px `rule-strong` ring on top, then the value in Fact type beside the finding in Finding, feature type. A live rain event's kicker carries an 8px dot in `rain-bar`, never `now`. The kicker and value are `aria-hidden`; the link's name is the finding plus where it goes. It links to the zoom of its subject ("Open the temperature view"), or to the Rain events tab for a rain event ("Open the Rain Events tab"). It draws no chart and carries no source note.

### Mosaic tile
A tile is one link, `#overview/<tile>`, to its zoom. Inside the link, in order:
- the title, an `h2` in Kicker type ("Rain since Oct 1");
- the number in Fact type, a word unit after it in Stat type;
- the short delta in Small type, `ink-2` ("−1° on usual", "steady over 3 hours");
- the mini chart at the tile's foot, 64px tall on every tile so a row lines up, `aria-hidden`, a crop of its zoom's main chart; with its "usual range" legend swatch above it when it draws a band under bars, so the chart stays the tile's last part and keeps the row's baseline;
- "Open the … view", for screen readers only.
The link's name is its own text, the tile's finding as number and delta, then "Open the … view"; no `aria-label` repeats it and the mini chart is never read. A tile has no sentence, no method and no source note. Where a chart needs words the number and delta do not carry (sunrise and sunset), they are spoken text, hidden on screen. Inside draws one row per indoor sensor in place of one number: the name, the reading with its humidity aside, and an `aria-hidden` 24-hour line. Focus rings the whole link; hover shifts the title to `ink-2`.

### Zoom
A tap on a mosaic tile (or the feature) opens its zoom. The route is the hash `#overview/<tile>`: opening pushes it, leaving it closes the zoom, and a shared link opens the zoom directly. An unknown tile falls back to `#overview`.
- **Phone:** full screen. The zoom takes the Overview's place; the mosaic, the sources footer and the bottom tab bar step aside, and the header stays. It leads with a "‹ Overview" back button ("Back to the Overview" to a screen reader), then the head, then its panels one under the other, split by `rule` hairlines.
- **Two and four columns:** in place, a full-width tile inserted in the grid under the tapped tile's row, with a Close pill (44px) at its top right. On four columns its panels sit two by two (a wide panel takes the row); on two columns, one under the other. The tapped tile carries `aria-expanded` and `aria-controls`; tapping it again closes the zoom.
- **Head and headings:** the zoom's heading is an `h2` in Kicker type (the tile's title), then its number in Display and its delta in Finding type. Each panel's title is an `h3` in Kicker type; a finding inside a panel is an `h4`. A panel carries no source note and no method; those are in the sources footer.
- **Close:** Close, Back (the button or the browser's) and Escape all close the zoom, stepping back through the history the zoom pushed, so the browser's Back and the zoom's agree. Opening moves focus to the zoom heading; closing returns it to the tile's link, and on a phone restores the scroll.
- **Motion:** the one animation, a 180ms fade and 8px rise on open, runs only under `prefers-reduced-motion: no-preference`; under reduce the zoom opens and closes at once, and scrolls without smoothing.
- **Contents:** the temperature zoom leads with the window range chart (a Usual row on top, Full history at the foot) and then the now-against-this-date strips; the rain zoom draws the Rain tab's water-year section and the dry spell zoom its dry-spell section; the sun zoom draws the Solar tab's top section; the wind zoom leads with the wind rose. Rain, Dry spell and Sun end in a "More in Rain" or "More in Solar" panel, one link to the tab. Last 10 days, Pressure, Humidity, Daylight and Inside show their readings at full size.

### Sources footer
The Overview's method and provenance, once, at the page foot under the grid, on `ground` with a `rule` top. Its heading is "Sources and method" (an `h2` in Kicker type). Under it a definition list, one entry per drawn section in page order (Right now, The feature, then each mosaic tile): the section's name (500, `ink-2`), what it reads, and how its comparison is made, in Source type and `ink-3`, at most 62ch. A section the snapshot leaves out may say why (Daylight with no location). One column on a phone, two from 760px. The archive time is not repeated: the header's freshness line says it once. The footer ends with the release line, "Lookout <version> · <sha>" ("Lookout 2026.11.12 · 93e95b8"), the repo's `VERSION` and the short SHA of the build ("local" outside a Vercel build), baked in at build time. The Rain, Rain events and Solar tabs keep their own source notes.

### Tile (Rain, Rain events, Solar)
Kicker, finding, chart or stats, then the folded source note. Solar and Rain events keep a one-line reading guide under the finding, in Small type.

### Source note (Rain, Rain events, Solar)
A native `details`: one closed line, "Source · <what>, archived <time>", in Source type with a 32px target; open, the full method paragraph Python already writes. A zoom never carries one, even where it draws a tab's section. When the archive lags past 15 minutes or a gap hides part of the chart, the closed line carries a hatched chip with the short reason. The in-chart hatch and its short reason stay; the note explains them, it never replaces them.

### Stat strip
A definition list, value over label, between hairlines or `rule` tops. Two columns on a phone, up to four on desktop.

### Segmented control and select
Segments are pills in a `rule-strong` border, 44px tall on a phone (40px on desktop); the pressed one is `ink` with `ground` text. Where a choice has more than three options or would not fit a phone row, it is a native `select` styled as a pill (the water-year filter).

### Event list (Rain events)
The list is the navigator. A pinned filter bar (water-year select, and sort: Newest, Wettest, Longest) stays at the top while the list scrolls. Each event is a row on its own tile, one open at a time: date, length, peak and quality chip on the left, the total on the right, and a 4px size bar under it giving the total against the wettest event on record: a `rain-bar` pill on a full-width `rule` track, the wettest event filling the track. The live event, or else the newest, is open by default. An open row shows the event's finding, a Total / Rate switch, the curve against the median event, its quality and its folded source. Opening another row closes it, so the curve always sits under the event it belongs to. "Show N more" ends the list.

- **Loading a curve:** the finding, stats and quality come from the snapshot and show at once. The chart area keeps the curve's height and reads "Loading the curve…" in Source type, `ink-3`.
- **A curve that fails to load:** the chart area reads "Couldn't load this event's curve." with a Try again pill. The rest of the row stays.
- **No events:** with a filter, one tile under the pinned filter bar reads "No rain events in <water year> yet." With none on record, the tile reads "No rain events on record yet." The filter bar stays.

### Chips and buttons
Quality chip: a pill in `rule-strong`, 12px; a poor rating is dashed with `ink-3` text. Hatched chip: a pill in `rule-strong` that starts with a 12px hatched swatch, then its reason in `ink-2` on the surface colour; the text never sits on stripes. The "latest" chip is outlined in `now`. Buttons are pills in `rule-strong`, 44px minimum.

### Quick facts
Gone from the Overview. Each became a mosaic tile with a zoom: Wind, Pressure, Humidity (with dew point; it took the Air card), and Sun and moon split into Sun today and Daylight (sunrise, sunset, day length, moon). A station with no coordinates has no Daylight tile; the sources footer says why.

### macOS widgets
Small (one value and one sentence), medium (the reading, a sentence and a 24-hour ribbon, or rain since Oct 1 against past years) and large (the reading, three stats and three range bars with the now rule). Widgets use the same tokens and the same contrast rules; their sentences are short forms Python writes.

### Chart grammar
Every chart is SVG drawn at its real pixel width, with no chart box and no vertical gridlines. Horizontal gridlines are `rule`; the baseline is `rule-strong`. Ticks are 12px `ink-3`. Labels sit on the marks with a tile-coloured halo. A key appears only where a ramp, the hatch or a band under bars needs decoding. The bold now line draws in once over 1.1s, skipped under reduced motion.

- **Today against the usual band (the hero, on the now tile):** midnight to midnight in station time. The band is the middle half of each hour's readings for this date, with its median as a dashed line in `ink-on-now`; it runs to midnight as measured history, never a forecast. Its basis (earlier years, the station's recent days, or the median only) is Python's, and a label under the axis words it ("median, your last day"). Today's line runs in `ink` from the archive's 15-minute slots and ends at the `now` dot; where the live reading is ahead of the archive, a dotted line joins them. Today's rain stands as quarter-hour bars on a track along the foot, in `ribbon-rain`, heavier rain a taller bar, never over the band; a slot the archive cannot speak for is hatched in `ink-3`. A snapshot with no band draws today's line and rain alone.
- **Mini charts (mosaic tiles):** each a crop of its zoom's main chart, 64px tall, with no axis title: the temperature tile's usual and today rows with the now rule, the rain tile's water-year curves (earlier years thin, this one bold to the `now` dot), the ten-day bars, the dry spell's square a day (dry in `rule`, a rain day in `rain-bar`, a day not in the record hatched, today outlined in `now`), the wind's compass arrow, the solar trace, the daylight arc, and Inside's 24-hour lines.
- **The "usual range" legend swatch:** every chart with a band under bars names the band in a legend, never by a label inside the plot, where bars would cover it: a 14×10px swatch of `temp-band` with a 1px dashed `temp-band-edge` and 2px corners, then "usual range" in 12px `ink-3`.
- **Range bars with a middle-half bar and a now rule (periods):** one row per window (today, yesterday, 7, 30, 90, 365 days). A 1.5px whisker min to max, a 12px `temp-iqr` bar over the middle half, one `now` rule across all rows with "now 47°" at the top. In the temperature zoom a Usual row leads and a Full history row (the archive's first day to now) ends the list. A window the record does not cover yet is not drawn, and one line under the chart names the next unlock ("Next to unlock: 7 days, on Nov 17, 2026."); an invalid window inside the record is a hatched row with its reason. On a phone each row's label sits above its bar.
- **Bold now over a usual band (events, solar):** the usual band (`*-band`, edged in light mode) covers the middle half of the comparison set with a dashed `median`. The current event, day or trace is a 3.5px bold line in its data hue, ending in an r5.5 `now` dot with its value.
- **Strips of now against this date:** each past day within a week of this date in earlier years is a dot; today is a 2.5px `now` rule labelled with its value; a claim above names the share.
- **Ten-day bars:** each day's low-to-high bar over the usual-range band for that date, coloured cool, mild or warm against the usual high.
- **Daily bars:** one bar per day in its data colour. Today's bar is marked by an r4 `now` dot 4px above its top (on the baseline when today is zero) and the label "today". The selected day is outlined in `ink`; missing days are hatched. On a phone the default is the last 30 days with a 30 / 90 switch; desktop shows 90.
- **Rain-by-hour grid:** days as columns, hours as rows, filled from the rain ramp on a `ground` background; the current hour outlined in `now` outside the cell, with a 1px `ground` gap. On a phone, the last 30 days.
- **Almanac grid:** years as rows, months as columns, cells filled from the rain or sun ramp, with the value in the cell. ≈ marks a prorated month; an invalid month is hatched with "n/a" ("gap" on the Overview); a month before the archive, one the record does not cover yet, or one still to come is a dashed `rule-strong` outline; the current month is filled from the ramp by its total so far, like the season cell, and outlined in `now` outside the cell with a 1px `tile` gap; it is hatched only when it has no total so far.
- **Water-year lines:** one line per water year, past years in `ink-3`-weight slate, this year bold in `rain-line` with a `now` dot.

### Named Rules (data and copy)
**The Finding First Rule.** A finding states what the data shows, with the number in it. It does not describe what the chart is.

**The One Number Rule.** Each fact is computed once, from one comparison set, and every view that shows it shows that number, the widgets included.

**The Live Beats Archive Rule.** A live value replaces the archived value for the same quantity. Archive lag is said once, in the header's freshness line, and on the Rain, Rain events and Solar tabs in the source note (or its hatched chip); never in a finding or on an Overview tile.

**The One Gap Status Rule.** Each month and window has one gap status: valid, prorated (≈) or invalid. Every chart, table and finding follows it. Invalid comparisons are hatched and never computed.

**The Name the Years Rule.** A finding that compares years names them, newest first.

**The One Format Rule.** Times `9:36 am`, dates with times `Oct 27, 11:00 pm`, durations `1 d 10 h`, `5 h 12 m`, `45 m`. Station time zone (`station.tz`), imperial units. That is the evidence form: stat rows, chart labels, event rows and source lines keep it exactly. A finding (its headline) says a duration the way a person does, with the units spelled out: minutes under an hour (`45 minutes`), whole hours to under 3 days (`about 26 hours`), the nearest half day to under 7 days (`about 3.5 days`), whole days after (`about 12 days`). `about` marks a rounded value; an exact one has none. Headlines speak, evidence stays exact.

**The Label Not Colour Rule.** Copy names marks by label or position ("the bold line", "the band", "the rule marked now"), never by colour.

**The Python Decides Rule.** Findings, short forms, the feature, which tiles and windows the record supports, the next unlock, gap status and every comparison are decided in Python and carried in the snapshot. The site draws what the snapshot says.

## Contrast

The pair list is the `contrast:` list in the frontmatter, and it is the only copy: the token contrast test reads it, and a new mark adds its pairs there. Each entry names a foreground token (or a list, meaning the better of them, for text in ramp cells), the token it sits on, the minimum (4.5 for text, 3 for graphics) and the themes. Every mark is listed against each surface it sits on: `tile`, `ground`, `now-surface`, or its band. Both themes pass every entry.

Two rules keep the list short. A `now` mark stands off data fills (the Now Stands Off Rule), so `now` is checked only on the three surfaces. On the now tile, hatching and strong lines draw in `ink-3`, so one entry covers both.

The tightest pairs, measured Nov 2026:

| Pair | Theme | Ratio | Needs |
|---|---|---|---|
| `temp-band-edge` (the dark band's fill) on `tile` | dark | 3.07 | 3 |
| `rain-0` on `ground` (the rain-by-hour grid) | light | 3.09 | 3 |
| `sun-0` and `sun-band-edge` on `tile` | dark | 3.13 | 3 |
| `temp-warm` on `temp-band` | dark | 3.16 | 3 |
| Text in the `rain-1` cell | light | 4.62 | 4.5 |
| `now` on `now-surface` | light | 4.69 | 4.5 |
| `ink-3` on `now-surface` | dark | 4.92 | 4.5 |
| `now` on `now-surface` | dark | 5.07 | 4.5 |

## Do's and Don'ts

### Do:
- **Do** show a finding's evidence at every size: a number, a short delta and a mini chart on a tile; the headline over its full charts in a zoom.
- **Do** keep the Overview's method and provenance in its one sources footer; on the Rain, Rain events and Solar tabs, fold them into the tile's source note.
- **Do** keep the phone page short: a fixed top, the feature, then the mosaic.
- **Do** hide what the record does not cover yet, and name the next unlock in one line.
- **Do** draw each chart at its real width, with 12px text.
- **Do** check every new graphic against the Three-to-One Graphics Rule in both themes.
- **Do** hatch every invalid comparison with a short reason in place.
- **Do** mark the current value in `now` orange, with a rule or dot and a label.
- **Do** follow the OS colour scheme and honour an explicit `?theme=` choice in the address.

### Don't:
- **Don't** show a forecast, or a fabricated value.
- **Don't** overprint rain on a band; rain stands on its own track.
- **Don't** use colour for chrome, or orange for anything but the finding.
- **Don't** shrink a desktop chart to fit a phone.
- **Don't** put the archive lag in a finding.
- **Don't** add shadows, gradients, italics or all-caps.
- **Don't** build gauge walls or tile grids of raw numbers: every tile says something.

## Sources

- Prototypes: the Lookout Prototypes canvas, rows E (desktop, widgets) and F (phone). https://claude.ai/artifact/3L81ndVLwwTKZP1jE4Ecmp
- Design system for decks and Design canvases: `design-system/` in this repo, built from this file and the shipped `web/src/styles/`. The repo copy is the source, and this file wins where they differ. Change `design-system/` first, then republish its mirror at https://claude.ai/artifact/Bmri2bVCvPqPumexcZkMXc.
- The Mosaic Overview: spec #460 with its amendments, and the boards and numbered decisions on branch `prototype/mosaic-overview` (decision 22, the sources footer; decision 23, the full-width feature row). Where the shipped site and a board differ, this file follows the site.
- The earlier Paper-world system this replaces is in git history (DESIGN.md before Nov 2026).
