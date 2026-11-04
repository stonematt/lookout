# Direction: the phone edition

The Streamlit port kept the FT's grammar but laid every part of it out at full volume, so the Overview on a phone runs about five screens, and a third of the words on it are method. The FT puts the source in one small line under the chart because the reader trusts the masthead. Lookout's readers do too. Keep the posture; turn the volume down on everything that isn't the finding.

The proposals below keep every rule in the brand book. Each one has a preview in the *Proposed* component group.

## 1. Provenance becomes a footnote: SourceNote

Today each chart closes with a 3–5 line paragraph: method, archive lag, gap list. Collapse it to one line, closed by default: **Source · archived 10:30 pm**, with a chevron. Tap opens the full method. Two rules keep it honest:

- **Promote only what changes the reading.** When the archive lags more than 15 minutes, or a gap hides part of the chart, the summary carries a hatched flag ("30 m not archived") so the caveat is visible without opening.
- The hatch on the chart still says "not valid" in place. The note explains it; it never replaces it.

## 2. Fold the hatched rows: GapFold

Four of six period rows read "n/a: before the archive", each a full-width hatched bar. Honest, and a wall. Show the valid rows, then fold every invalid run into one hatched row: **90 d, 365 d · before the archive**. Same gap status, a quarter of the height. As the archive grows the fold shrinks and disappears on its own.

## 3. Keep "now" on screen: NowBar

Once the hero scrolls away the reader loses the one number they came for. Merge a compact reading into the sticky tab bar: **59.3° · raining 0.12 in/hr · 3 m ago**. It appears only after the hero leaves the viewport, and it is the place the stale state lives (hollow dot, hatched age).

## 4. An Overview that ends

Overview answers "now, and how normal". On a phone that is: the first screen, the adaptive lead, and one "against normal" chart chosen by the same rules as the lead. The periods chart, the ten-day list and the rain almanac move to their tabs (Temperature can become a tab of its own), each linked from the Overview with a one-line finding: "Rain this water year: 4.2 in, ahead of 2025 → Rain". Target: Overview in two and a half screens.

## 5. A kicker over every finding

A small sentence-case label above the headline ties a section to its subject: **● Rain event**, **Temperature**, **Solar**. The dot is `note` when that section is the live finding. It gives a scanning thumb something to land on without raising the headline's volume.

## 6. Scrub, don't hover

Phones have no hover. On the ribbon, the event curve and the solar trace, a press-and-drag moves a hairline cursor and writes the value into the standfirst line ("2:15 am · 0.18 in"), the way the almanac already writes to its readout. Release returns the standfirst.

## 7. Charts at phone width

Under 520px of chart width: drop to three y-ticks, put the now label above the dot instead of beside it, and widen the ribbon's ticks to every 12 h. The rule stays the same; the density follows the width.

## 8. Later: the glance surfaces

The iOS widget from PRODUCT.md is this system's smallest layout: the NowBar's three facts plus a 12-hour ribbon. Design it as a 2×2 of the same tokens before building the feed, so the site and widget agree on what "now" says.
