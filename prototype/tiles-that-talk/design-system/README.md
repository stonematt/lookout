# Lookout

**The Annotated Almanac.** One weather station's own record, read like Dark Sky and argued like the Financial Times. A short finding in plain words, a one-line standfirst, the evidence, then where the numbers came from.

This system is built from `stonematt/lookout`: `web/src/styles/tokens.css`, `DESIGN.md` and `PRODUCT.md`. Values are copied exactly. The **Direction** section and the components in the *Proposed* group are the next step for the phone: they are not in the site yet.

## Who it is for

The station operator, on a phone, in short glances between other things. A few friends through a shared link. The question every screen answers: what is it doing outside right now, and how normal is that? There is no forecast; every number is something this station measured.

## Content fundamentals

- **Finding first.** A headline states what the data shows, with the number in it: "Raining now: 0.30 in so far, against 0.42 in for a usual event". Never "Rain event chart".
- **The standfirst says how to read the chart**, in one line, max 62ch: "The bold line is this event's rain so far; the shaded band is the usual event."
- **Refer to marks by label or position**, never colour: "the bold line", "the line marked now", "the shaded band".
- **Name the years**, newest first: "against 3.5 in (2025) and 4.5 in (2024)".
- **One format.** `9:36 am`. `Oct 27, 11:00 pm`. `1 d 10 h`, `5 h 12 m`, `45 m`. Station time zone, imperial units.
- **Live beats archive.** The live value replaces the archived one. Archive lag goes in the source note, never in a headline.
- **Honest gaps.** An invalid comparison is hatched and named ("n/a: gap in record"), never computed or smoothed.

## Visual foundations

**Colour belongs to the data.** Chrome is warm paper (`ground`, `surface`) and warm ink (`ink`, `ink-2`, `ink-3`) with two hairlines (`rule`, `rule-strong`). Every saturated colour on screen is a measurement:

- Rain: `rain-0` to `rain-4`, drizzle to downpour. `rain-0` is also the usual band; `rain-2` the dashed median; `rain-3` (Ledger Blue) the bold now-curve, the selected tab, the focus ring and links.
- Temperature: `temp-cold` → `temp-cool` → `temp-mild` → `temp-warm` → `temp-hot`.
- Solar: `sun`, with `sun-soft` for daylight and the solar usual band. `night` is the ribbon's night band.
- **One claret.** `note` marks the "now" the headline is about, and nothing else. One claret subject per section.
- **Hatching is the one word for "not valid"**: 135° stripes of `gap-a` over `gap-b`, 4px on an 8px repeat.

**Flat paper.** No shadows, no tinted cards, no chart boxes. Groups separate by hairline or whitespace. The chart is the highest-contrast thing in its section; the headline is a quiet 600.

**AA both ways.** All text meets 4.5:1 (3:1 at 18px+ or bold 14px+) on `ground` in light and dark. The pressed segment is an `ink` fill with `ground` text, about 14:1.

## Typography

One family, **Atkinson Hyperlegible Next**, for every word and number, built for dyslexic readers: upright, left-aligned, no italics, no all-caps runs beyond a short label, line height at least 1.5, 12px floor. Hierarchy is weight and size only, on a gentle ramp (0.875, 1, 1.1875, 1.3125, 1.625 rem) so the chart reads before the headline. Tabular digits only where numbers line up: tables, stat strips, axes, almanac cells, the ribbon, the current temperature.

## Layout

One column on a phone, 16px gutter. From 1024px, 5/7 or 7/5 two-column grids in a 1240px container with a 32px gutter. Section rhythm: headline, `s-1`, standfirst, `s-3`, chart, `s-3`, source note. Sections are `s-7` apart.

**Phone first screen**, in order: station and live age, the current temperature with its condition sentence, the Right-now strip, the 48-hour condition ribbon, then the lead.

**Adaptive lead**, checked in order: rain if an event is live or ended in the last 12 h; temperature if at least 10° from the month median; solar if at least 400 W/m²; otherwise calm (days since rain against the usual dry spell). "Why this leads" lists every rule and marks the ones met.

## Iconography

None. Lookout has no icon set and no logo mark: the name is set in type. Dots, rules and the hatch are the only symbols, and every one is drawn from tokens.

## Charts

SVG at real pixel width. Horizontal hairline gridlines in `rule`, never vertical. Ticks 12px `ink-3`. Labels sit on the marks with a 4px `ground` halo; a key appears only where a ramp or the hatch needs decoding. The bold now-line is 3.5px in the data hue, ending in an r5 `note` dot ringed in `ground`, and draws in once over 1.1s (skipped under reduced motion).

## Don't

Serif, italics, centred paragraphs. Colour on chrome, or claret on anything but the finding. A line across a gap. Lag in a headline. Gauge walls or tile grids of raw numbers.
