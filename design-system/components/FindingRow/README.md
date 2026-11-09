A finding row is a closed accordion row on a phone: its kicker over its finding sentence, with a chevron in a `rule-strong` ring, at least 56px tall. Closed, the rows read as a list of findings; opening one closes any other row in the same group and shows its standfirst, chart and folded source.

**Consumer provides.** A kicker and a finding per row, and the opened row's chart and source. All rows of a tab sit in one tile, split by `rule` hairlines, with an `In context` heading in `kicker` type.

**Use.** Use native `details` elements sharing one group name per tab. The chevron points down when closed and up when open. Do not give a row a colour; the finding is the content.

Static rendition of `web/src/tiles.ts` (`.tile-row`) and `web/src/styles/tiles.css`.
