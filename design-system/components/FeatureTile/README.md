The feature tile is one full-width tile under the now tile, holding whatever matters most today. It is one link: the kicker with a chevron in a 32px `rule-strong` ring on top, then the value in `fact` (a word unit in `stat`) beside the finding in `finding-lead`.

**Consumer provides.** The snapshot's `feature`, chosen by the refresh run's Feature Rule (ADR-0009), checked in order: a rain event live or ended in the last 12 hours; a reading outside the 10th to 90th percentile of its readings at this hour within a week of this date in earlier years, the furthest out winning (skipped with no earlier year); a dry spell longer than the usual one; otherwise "a usual day". The site words, formats and links it; it never chooses. A snapshot with no `feature` draws no tile.

**Use.**
- The link goes to the zoom of its subject ("Open the temperature view"), or for a rain event to the Rain events tab ("Open the Rain Events tab"). Those words end the link's name, after the finding, for screen readers only.
- The kicker and value are `aria-hidden`, so the finding is read once.
- A live rain event's kicker carries an 8px dot in `rain-bar`, never `now`: the now tile holds the page's orange.
- No chart, no source note. The sources footer states the rule.
- The tile spans every column, on a phone and on both grids.

Static rendition of `web/src/overview/feature.ts` and `feature.css`.
