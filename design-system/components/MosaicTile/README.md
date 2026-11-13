A mosaic tile is one small Overview tile: a title, a number, a short delta and a mini chart, all inside one link to its zoom at `#overview/<tile>`. The mosaic runs in a fixed order: Temperature, Rain since Oct 1, Last 10 days, Dry spell, Wind, Pressure, Humidity, Sun today, Daylight, Inside. A phone sets two to a row; from 760px two columns, from 1180px rows of four, in the same order.

**Consumer provides.** The number (with an optional word unit), the delta and the mini chart's data, all from the snapshot; and whether the tile is shown (`record_depth.tiles`, an indoor sensor for Inside, a location for Daylight). A tile not shown is not drawn and the later tiles move up.

**Use.**
- The title is an `h2` in `kicker` type, `ink-3`. The number is `fact`, weight 300, tabular; under 360px of viewport it drops to `stat`. The unit is `stat`. The delta is `standfirst` size in `ink-2`.
- The mini chart sits at the tile's foot, 64px tall on every tile so a row lines up, and is `aria-hidden`. It is a crop of the zoom's main chart. A chart with a band under bars carries the "usual range" legend swatch (a 14x10px `temp-band` fill with a 1px dashed `temp-band-edge`), never a label inside the plot.
- The link's name is its own text, then "Open the … view" for screen readers only. No `aria-label`, no sentence, no method and no source note on the tile.
- Inside draws one row per indoor sensor in place of one number: name, reading with humidity aside, and an `aria-hidden` 24-hour line.
- Focus rings the whole link; hover shifts the title to `ink-2`.

Static rendition of `web/src/overview/mosaic.ts`, `mosaic-tiles.ts`, `mosaic-conditions.ts` and `mosaic.css`.
