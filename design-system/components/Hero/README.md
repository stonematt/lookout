The hero is the leading tile: the finding in `finding-lead`, a standfirst, a three-stat strip, the bold now line over a usual band, and "Why this leads" and the source folded below. Rain leads if an event is live or ended in the last 12 hours; temperature leads if it is at least 10 degrees from the month median; solar leads if it is at least 400 W/m²; otherwise the lead is calm.

**Consumer provides.** The finding, the standfirst, the stats, the band, median and now series, the now value, and the list of lead rules with their live values and which were met, all from the snapshot.

**Use.**
- The usual band covers the middle half of the comparison set, edged in light mode and passing by its fill in dark, with a dashed `median`. The current value is a 3.5px bold line in its data hue, ending in an r5.5 `now` dot and its value in `annotation` on a halo.
- "Why this leads" lists each rule with its live value and marks the ones met with a filled ring; the lead rule is weight 600.
- Name marks by label ("the bold line", "the band"), not by colour.

Static rendition of `web/src/overview/hero.ts` and `hero.css`.
