---
name: Lookout
description: A personal weather record for the lookout post station. It reads like Dark Sky and argues like the FT.
colors:
  ground: "#f7f4ee"
  surface: "#fcfbf8"
  ink: "#26231f"
  ink-2: "#4b463f"
  ink-3: "#686259"
  rule: "#e3ded3"
  rule-strong: "#c2baab"
  rain-0: "#dde5ee"
  rain-1: "#a9bdd3"
  rain-2: "#6a8aac"
  rain-3: "#365f8b"
  rain-4: "#1f405f"
  temp-cold: "#4a6f94"
  temp-cool: "#86a3bd"
  temp-mild: "#b3aea4"
  temp-warm: "#cf9654"
  temp-hot: "#b0522b"
  sun: "#d9a23c"
  sun-soft: "#f1e2bd"
  night: "#ece7dc"
  note: "#94325a"
  gap-a: "#e6e1d6"
  gap-b: "#f7f4ee"
  ground-dark: "#1b1a18"
  surface-dark: "#242321"
  ink-dark: "#ece8e0"
  ink-2-dark: "#c3bdb2"
  ink-3-dark: "#9d968b"
  rule-dark: "#34322e"
  rule-strong-dark: "#4f4c46"
  rain-0-dark: "#263342"
  rain-1-dark: "#3a5573"
  rain-2-dark: "#5f87b2"
  rain-3-dark: "#93b3d6"
  rain-4-dark: "#d0e0f0"
  temp-cold-dark: "#6f95bd"
  temp-cool-dark: "#94b1cb"
  temp-mild-dark: "#8f8a80"
  temp-warm-dark: "#d8a463"
  temp-hot-dark: "#e07a4d"
  sun-dark: "#e2b04f"
  sun-soft-dark: "#3a3220"
  night-dark: "#151412"
  note-dark: "#e59ab8"
  gap-a-dark: "#2a2927"
  gap-b-dark: "#1b1a18"
typography:
  display:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "clamp(2.75rem, 12vw, 3.5rem)"
    fontWeight: 650
    lineHeight: 0.9
    letterSpacing: "-0.02em"
    fontFeature: "\"tnum\" 1"
  headline:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "1.3125rem"
    fontWeight: 600
    lineHeight: 1.28
    letterSpacing: "0.005em"
  title:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "1.1875rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.005em"
  subtitle:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: "0.005em"
  body:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: "0.012em"
  standfirst:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "0.012em"
  source:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: "0.012em"
  stat:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "1.1875rem"
    fontWeight: 650
    lineHeight: 1.3
    fontFeature: "\"tnum\" 1"
  label:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.3
    letterSpacing: "0.012em"
  axis:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    fontFeature: "\"tnum\" 1"
  annotation:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 650
rounded:
  xs: "2px"
  sm: "4px"
  md: "8px"
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
  tab:
    textColor: "{colors.ink-2}"
    typography: "{typography.body}"
    rounded: "0"
    padding: "0 14px"
    height: "46px"
  tab-selected:
    textColor: "{colors.ink}"
  segment:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-2}"
    typography: "{typography.standfirst}"
    rounded: "{rounded.md}"
    padding: "0 14px"
    height: "40px"
  segment-pressed:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
  button-more:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "44px"
  quality-chip:
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  almanac-cell:
    rounded: "{rounded.sm}"
    typography: "{typography.label}"
    height: "44px"
  condition-ribbon:
    backgroundColor: "{colors.night}"
    rounded: "{rounded.sm}"
    height: "52px"
---

# Design System: Lookout

## Overview

**Creative North Star: "The Annotated Almanac"**

Lookout is one station's own record, set out the way the Financial Times sets out a chart: a short finding in plain words, a one-line standfirst, the evidence, then where the numbers came from. The Dark Sky half supplies the presentation: a large current temperature, a 48-hour condition ribbon, and colour that comes from the weather rather than from the chrome. The headlines say what the data shows and then step back. The chart is the most contrasty thing in any section.

The Paper world sits on a warm off-white ground by day and a warm off-black ground at night. The chrome is close to greyscale and warm: ink, two softer inks, two hairline rules. Colour is kept for data. Rain is a desaturated blue ramp from drizzle to downpour, temperature runs from cold blue to terracotta, solar is amber, and one claret marks the finding on a chart. One readable family, Atkinson Hyperlegible Next, sets every word and number. The system is built for dyslexic readers: text is left-aligned, nothing is italic, there are no long all-caps runs, line height is generous, and measures are short.

The record is honest about itself. Gaps are hatched, never smoothed over. A live value beats an archived one. Each fact is computed once and agrees everywhere it appears. The system has no dark-pattern urgency, no gauge walls and no tile grids of raw numbers.

**Key Characteristics:**
- Every section runs finding headline, standfirst, chart, source line.
- Warm neutral chrome; data colour carries all the hue.
- One sans family for text and numerals; tabular digits only where numbers line up.
- Hairline axes, direct labels, no chart boxes.
- Hatching is the one visual word for "not valid".
- Light and dark follow the OS. An explicit override is honoured.

## Colors

The palette is warm paper and warm ink, with four restrained data hues and one claret annotation. Light values are the frontmatter keys. Each `-dark` key is the same role in dark mode.

### Primary
- **Ledger Blue** (`rain-3`): the strongest data hue and the main interactive accent. It draws the bold "now" curve on rain charts, the selected-tab underline, the focus ring (3px), the met-rule dot in "Why this leads", and links. In dark mode it lifts to `rain-3-dark` so it stays above 4.5:1 on the ground.
- **Precipitation ramp** (`rain-0` to `rain-4`): drizzle to downpour. Ribbon rain intensity, almanac rain cells and the dry-spell strip use this ramp. `rain-0` is also the **usual band**, the soft fill behind every "now against normal" chart. `rain-2` draws the dashed median.

### Secondary
- **Temperature ramp** (`temp-cold`, `temp-cool`, `temp-mild`, `temp-warm`, `temp-hot`): cold blue through stone to terracotta. It fills the ten-day low-to-high bars (each bar shows the slice of the ramp it covers), the almanac mean-temperature cells, and the warm or cold "now" line on the temperature lead. `temp-cool` is the IQR bar in the periods chart.
- **Sun Amber** (`sun`, `sun-soft`): solar. `sun-soft` is daylight on the ribbon and the usual band on solar charts. The solar "now" mark is drawn in sun-ink, a mix of 62% `sun` with 38% `ink`, so the line keeps its contrast.

### Tertiary
- **Claret Note** (`note`; dark `note-dark`): the finding. It colours the now rule, the now dot and the "now 46.9°" label on charts and strips, the current-month outline in the almanac, and the live-event dot. Use it only for what the headline is about. Never use it for chrome, warnings or decoration.

### Neutral
- **Paper Ground** (`ground`): the page, an off-white. It also draws the 4px halo stroke behind chart labels so they read over marks.
- **Card Paper** (`surface`): unpressed buttons, segmented controls and hover labels. There is no tinted panel layer.
- **Warm Ink** (`ink`): headlines, values and the ribbon's temperature trace. It also fills the pressed segment.
- **Soft Ink** (`ink-2`): standfirsts, source lines, `dt` labels, table headers and chart labels.
- **Faint Ink** (`ink-3`): axis tick labels and range whiskers only. It is short and large enough to stay legible.
- **Hairline** (`rule`): axes, gridlines, table row dividers and stat-strip borders.
- **Hairline Strong** (`rule-strong`): control borders, table header rule and strip baselines.
- **Night** (`night`): the night band on the condition ribbon.
- **Gap Hatch** (`gap-a` over `gap-b`): the 135° hatch for invalid comparisons.

### Named Rules
**The Data Owns the Hue Rule.** Chrome is warm neutral. Every saturated colour on screen is a measurement, the usual band, or the claret finding.

**The One Claret Rule.** `note` marks the "now" or the current item that the headline is about, and nothing else. A screen section has one claret subject.

**The AA Both Ways Rule.** All text meets WCAG AA (4.5:1, or 3:1 for 18px+ or bold 14px+) on the ground in light and dark. A selected control meets 4.5:1: the pressed segment is `ink` fill with `ground` text (about 14:1). Text on a ramp-coloured cell is chosen per cell, near-white or near-black, whichever has more contrast with the computed fill.

## Typography

**Display Font:** Atkinson Hyperlegible Next (with Atkinson Hyperlegible, system-ui, sans-serif)
**Body Font:** Atkinson Hyperlegible Next
**Label/Mono Font:** Atkinson Hyperlegible Next, using `tnum` for tabular figures

**Character:** One humanist, highly distinguishable family for everything. Letterforms are chosen so no two can be mistaken for each other. The tracking is slightly open (0.012em letter, 0.08em word) and the leading is generous (1.55). Hierarchy comes from weight and size only.

### Hierarchy
- **Display** (650, clamp(2.75rem, 12vw, 3.5rem), 0.9, -0.02em): the current temperature only, with a superscript °F at 0.4em.
- **Headline** (600, 1.3125rem; 1.625rem at 1024px and up; 1.28; max 30ch): the hero lead's finding.
- **Title** (600, 1.1875rem; 1.3125rem at 1024px and up; 1.3; max 36ch): section findings (`h2`).
- **Subtitle** (600, 1rem, 1.35): sub-blocks and table captions.
- **Body** (400, 1rem, 1.55, max 60ch): the condition summary beside the temperature, and narrative.
- **Standfirst** (400, 0.875rem, 1.6, ink-2, max 62ch): the one line under a headline that says how to read the chart.
- **Source** (400, 0.8125rem, 1.55, ink-2, max 62ch): the source line under each chart, which carries method, gaps and archive lag.
- **Stat** (650, 1.1875rem, tabular): Right-now values and hero facts. The value sits above its `dt` label (0.8125–0.875rem, ink-2).
- **Label** (400, 0.75rem): keys, almanac headers, ribbon ticks and chips. 12px is the floor for any text.
- **Axis** (12px, ink-3, tabular): chart ticks. In-chart labels are 12.5px in ink-2. Annotations are 650 at 13px in `note`.

Type steps: 0.875, 1, 1.1875, 1.3125, 1.625 and 4rem. That is a gentle ramp so the chart, not the headline, is read first.

### Named Rules
**The No Serif, No Italic Rule.** Every word is Atkinson Hyperlegible Next, upright and left-aligned. Emphasis is weight 600–650. Text never uses italics, centred paragraphs or all-caps runs longer than a short label.

**The Tabular Where It Lines Up Rule.** Tabular digits (`tnum`) are used only in tables, stat strips, axes, almanac cells, the ribbon and the temperature. Headlines, standfirsts, source lines and body text use proportional digits.

**The Quiet Headline Rule.** A headline is short, 600 weight, and never larger than step 3. If it competes with the chart for contrast, the headline gives way.

## Layout

There is one column on a phone. From 1024px up the page uses 5/7 or 7/5 two-column grids. The container is at most 1240px wide, with 16px side padding, rising to 32px at 1024px and up, and 96px bottom padding. Grid gaps are 48px vertical by 56px horizontal. Major sections are separated by 48px (`s-7`). Inside a section the rhythm is: headline, 4px, standfirst, 12px, chart, 12px, source line. Breakpoints are 480px (compact events table), 600px (stat values step up), 520px of chart width (narrow chart variants, almanac values hidden), and 1024px (two columns, larger headlines).

The tabs (Overview, Rain, Rain Events, Solar) are sticky at the top, on the ground colour, with a hairline below.

**The Phone First-Screen Rule.** On a phone, the first screen holds, in order: the station name and live timestamp, the current temperature with the condition sentence beside it, the Right-now stat strip (feels like, wind, rain today), the 48-hour condition ribbon, and then the hero lead.

**The Adaptive Lead Rule.** The hero is chosen by rules, checked in order. Rain leads if an event is live or ended in the last 12 h. Temperature leads if it is at least 10° from the month median. Solar leads if it is at least 400 W/m². Otherwise the lead is calm (days since last rain against the usual dry spell). A "Why this leads" disclosure lists each rule with its live value and marks which were met.

## Elevation & Depth

The system is flat. No surface has a shadow, and depth comes from tone and hairlines. The ground and surface differ by a shade. Stat strips and tables are held by 1px rules, not by cards. The only layering is in charts: a soft usual band at the back, hatching over invalid spans, the bold now mark on top, and labels haloed in the ground colour. Two ring effects are state marks, not elevation: the almanac's selected cell gets a 2px `ink` ring, and the ribbon's now dot gets a 2px `ground` ring.

### Named Rules
**The Flat Paper Rule.** Nothing floats. If a group needs separating, use a hairline or whitespace, never a shadow or a tinted card.

## Shapes

Corners are small and functional. Swatches, solar bars and strip cells use 2px. Almanac cells, the condition ribbon, gap rows and invalid-window bars use 4px. Range bands and IQR bars use 3px. Segmented controls and buttons use 8px. The quality chip is a pill. Data dots are circles (r 2.6–5.5). Tabs are square, with a 3px inset underline. Strokes: axes are 1px hairlines, thin marks are 1.5px, and the bold now line is 3.5px with round caps and joins. Hatching is always 135° stripes, 3–4px on an 8px repeat, in `gap-a` over `gap-b`.

## Components

### Tabs
- **Style:** ground background, 46px minimum height, 0 14px padding, 1rem in ink-2.
- **Selected:** ink, weight 650, 3px `rain-3` inset underline. Hover moves the text to ink.
- **Behaviour:** roving tabindex; arrow keys, Home and End move between tabs; the hash tracks the open tab.

### Segmented control
- **Style:** 1px `rule-strong` border, 8px corners, `surface` buttons 40px high, 0.875rem in ink-2, hairline dividers between buttons.
- **Pressed:** `ink` fill, `ground` text, weight 650 (meets 4.5:1 in light and dark).

### Buttons
- **More / secondary:** `surface` fill, 1px `rule-strong` border, 8px corners, 44px minimum height, 0 16px padding, weight 600.
- **Focus:** a 3px `rain-3` outline, 2px offset, on every focusable element.

### Chips
- **Quality chip:** pill, 1px `rule-strong` border, 0.75rem at 600. A low-quality rating uses a dashed border and ink-2 text.

### Stat strip
- **Style:** a definition list in three equal columns between top and bottom hairlines, 8px vertical padding. The value (Stat type, tabular) sits above its label (ink-2). Used for Right now (max 520px) and the hero facts.

### Disclosures
- **Why this leads / Show the numbers:** native details elements, 0.875rem ink-2 summary with a 32px target. The rules list shows an 8px ring per rule, filled `rain-3` when met, with the leading rule in ink at 600.

### Tables
- **Style:** full width, collapsed borders, 0.875rem, tabular. Headers are 0.75rem at 600 in ink-2 over a `rule-strong` line, and rows are divided by `rule`. Numbers are right-aligned and the first column left-aligned. An invalid cell is hatched, centred, and reads "n/a" with the reason in a title and in screen-reader text. The current event row is tinted `rain-0`, and the pinned live event sits above a 2px rule.

### Chart grammar
Every chart is drawn as SVG at its real pixel width. There are no chart boxes and no gridlines on x. Horizontal gridlines are hairlines in `rule`, and ticks are 12px ink-3 tabular. Labels sit directly on the marks, haloed in `ground`. A compact inline key appears only where a colour ramp or the hatch needs decoding (the almanac and the ribbon). The bold now line draws in once over 1.1s (cubic-bezier(.16,1,.3,1)), and the animation is skipped under reduced motion.

- **Range bars with an IQR band and a now rule (periods):** one row per window (today, yesterday, 7, 30, 90, 365 days), each labelled with its name and dates. A 1.5px ink-3 whisker runs min to max, with r 2.6 end dots and the values at each end. A 12px `temp-cool` bar covers the 25th to 75th percentile. One vertical 2.25px `note` rule crosses all rows at the current value, with a dot and a "now 46.9°" label at the top.
- **Strips of now against normal:** a 104px band with a baseline and ticks. Each past day within ±7 days of this date in earlier years is a jittered translucent dot (`temp-cool` for temperature, `rain-2` for rain on a square-root scale). Today is a 2.5px `note` rule labelled "today … so far". A claim line above names the percentage and the sample.
- **Almanac grid:** years as rows, months as columns, 44px cells with 4px corners and 3px gaps. A valid month is filled from the rain ramp (square-root scale) or the temperature ramp, with its value in the cell when there is room; contrasting text is chosen per cell. A prorated month shows ≈ before its value. An invalid month is hatched with no value. A month before the archive began has a dashed outline. The current month is outlined in `note` and lightly hatched, because it is not yet comparable. Tap, hover or focus writes the month's full sentence to a live readout below the grid.
- **Hatched gaps:** wherever a comparison is not valid (an archive gap, a low-coverage day, a window missing more than 10% of its days, the span after the archive ends), the mark is replaced by the 135° hatch. There is no line, bar or value under it. A short label in place says why ("n/a: gap in record", "not archived yet").
- **Bold now over a soft usual range (hero, events, solar):** the usual range is a soft fill (`rain-0`; `sun-soft` for solar) covering the middle half of the comparison set, with a dashed 1.5px median line. The current event, day or trace is a 3.5px bold line in the data hue, ending in an r 5 dot ringed in `ground` with its value in the 650 annotation. Where the live value is ahead of the archive, the unarchived span is hatched, the known live total is joined by a dotted line, and an in-chart note says "only the live total is known". The ten-day list uses the same grammar in miniature: a `rule`-coloured usual band behind each day's low-to-high bar, painted from the temperature ramp.
- **48-hour condition ribbon:** a 52px band with 4px corners. Each 15-minute slot is `night` or `sun-soft`, overprinted by the rain ramp in four steps. The temperature is traced over it in a 2px `ink` line on a 4.5px `ground` halo, with a `note` dot at the right edge and a "now" label. Unarchived spans are hatched. Ticks every 6 hours read "6 am", "noon" and weekday names. A one-line claim above gives the range and the rain hours.

### Named Rules (data and copy)
**The One Number Rule.** Each fact is computed once, from one comparison set, and every view that shows it shows that number.

**The Live Beats Archive Rule.** A live value always replaces the archived value for the same quantity. The archive lag goes in the source line, never in a headline.

**The One Gap Status Rule.** Each month and each window has one gap status: valid, prorated (≈) or invalid. Every chart, table and headline follows it. A ≈ window is never called "full". Invalid comparisons are hatched and never computed.

**The Name the Years Rule.** A headline that compares years names them, newest first: "against 3.5 in (2025) and 4.5 in (2024)".

**The One Format Rule.** Times are written `9:36 am`, and dates with times are written `Oct 27, 11:00 pm`. Durations are written `1 d 10 h`, `5 h 12 m` or `45 m`. Use station-local Pacific time and imperial units.

**The Label Not Colour Rule.** Copy refers to marks by label or position ("the bold line", "the line marked now", "the shaded band", "the outlined month"), never by colour.

**The Finding First Rule.** A headline states what the data shows, with the number in it. It does not describe what the chart is.

## Do's and Don'ts

### Do:
- **Do** build every section as finding headline, one-line standfirst (max 62ch), chart, then source line.
- **Do** keep the chart the highest-contrast element in its section. The bold now line is 3.5px in the data hue, and the headline is a quiet 600 weight.
- **Do** hatch every invalid comparison at 135° in `gap-a` over `gap-b`, with a short reason in place.
- **Do** mark the current value in `note` claret, with a rule or dot and a "now" label.
- **Do** set all text in Atkinson Hyperlegible Next, left-aligned, at a line height of at least 1.5, with no text under 12px.
- **Do** use tabular digits in tables, stats, axes, almanac cells and the ribbon, and proportional digits everywhere else.
- **Do** follow the OS colour scheme by default and honour an explicit light or dark choice.
- **Do** give every chart an accessible name and every invalid cell a spoken reason.

### Don't:
- **Don't** use a serif, italics, centred paragraphs or all-caps runs.
- **Don't** use colour for chrome, or claret for anything but the finding.
- **Don't** draw a line, bar or value across a gap, or call a ≈ window "full".
- **Don't** put the archive lag in a headline, or show an archived value where a live one exists.
- **Don't** refer to a mark by its colour in copy.
- **Don't** add shadows, tinted cards, chart boxes or vertical gridlines.
- **Don't** build gauge walls or tile grids of raw numbers. Lead with the finding.
