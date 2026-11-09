The tile is the block every tab's sections take: a kicker, a finding, a standfirst, a chart or stats, then a folded source note, on `tile` over `ground` with 22px corners and padding. The now tile is the same shape on `now-surface` and holds the temperature in `display`, a condition sentence and a four-stat strip.

**Consumer provides.** The kicker, the finding sentence, the standfirst and the chart or stat content, all decided in Python and carried in the snapshot; the source line and, when the archive lags more than 15 minutes or a gap hides part of the chart, a hatched chip with the short reason.

**Use.**
- Open every tile with its finding, with the number in it. Method, provenance and "why this leads" fold into one-line disclosures at the tile's foot, with a 32px target.
- A tile sets only `--surface` (`tile`, or `now-surface` on the now tile); every halo, hatch ground and ring inside follows it. On the now tile hatching and strong lines draw in `ink-3`.
- One orange subject per tile. The leading tile carries the 8px `now` dot before its kicker.
- Do not put the archive lag in the finding; the source note carries it.

Static rendition of `web/src/tiles.ts` and `web/src/styles/tiles.css`; the live tile is drawn by that module.
