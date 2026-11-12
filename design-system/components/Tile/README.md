The tile is the block the Rain, Rain events and Solar tabs build their sections from: a kicker, a finding, a chart or stats, then a folded source note, on `tile` over `ground` with 22px corners and padding (20px and 16px on a phone). The Overview's tiles are the Hero, FeatureTile and MosaicTile.

**Consumer provides.** The kicker, the finding sentence and the chart or stat content, all decided in Python and carried in the snapshot; the source line and, when the archive lags more than 15 minutes or a gap hides part of the chart, a hatched chip with the short reason. Solar and Rain events also keep a one-line reading guide under the finding, in `standfirst` size and `ink-2`.

**Use.**
- Open every tile with its finding, with the number in it. Method and provenance fold into the source note at the tile's foot, a native `details` with a 32px target.
- A tile sets only `--surface`; every halo, hatch ground and ring inside follows it.
- One orange subject per tile.
- Do not put the archive lag in the finding; the source note carries it.

Static rendition of `web/src/tiles.ts` and `web/src/styles/tiles.css`.
