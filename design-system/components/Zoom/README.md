A zoom is a tile's full view. A tap on a mosaic tile, or on the feature, opens it at the hash route `#overview/<tile>`; leaving the route closes it, and a shared link opens it directly.

**Consumer provides.** The zoom's panels, each a title and its content: the finding headline and the full charts, drawn from the snapshot. The temperature zoom leads with the window range chart (a Usual row on top, Full history at the foot), then the now-against-this-date strips; the rain zoom adds the rain gauge ("Rain over each window": three panels of bars on shared axes, earlier years as thin grey bars, a year with no total hatched, `web/src/rain/gauge.ts`) in place of the window strip; Rain, Dry spell and Sun draw their tab's section and end in a "More in Rain" or "More in Solar" link; Wind leads with the wind rose.

**Use.**
- Phone: full screen. The mosaic, the sources footer and the bottom tab bar step aside; the header stays. A "‹ Overview" button leads ("Back to the Overview" to a screen reader), panels sit one under the other split by `rule` hairlines.
- Two and four columns: a full-width tile in the grid under the tapped tile's row, Close (a 44px pill) at its top right. Panels sit two by two on four columns, one under the other on two. The tapped tile carries `aria-expanded` and `aria-controls`.
- Headings: the zoom is an `h2` in `kicker` type, then its number in `display` and its delta in `finding` type; each panel title is an `h3` in `kicker` type; a finding in a panel is an `h4`.
- Close, Back and Escape close it, stepping back through the history it pushed. Focus moves to the zoom heading on open and back to the tile's link on close.
- The one animation, a 180ms fade and rise, runs only under `prefers-reduced-motion: no-preference`.
- No source note and no method in a zoom; they live in the sources footer.

Static rendition of `web/src/overview/zoom.ts`, `zoom.css` and `zooms/`.
