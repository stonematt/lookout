The sources footer holds the Overview's method and provenance, once, at the page foot under the grid. It sits on `ground`, not a tile, under a `rule` top: a "Sources and method" heading (`h2`, `kicker` type), then one entry per drawn section in page order, then the release line.

**Consumer provides.** For each section (Right now, The feature, each mosaic tile): its name, what it reads, and how its comparison is made, the words each section's own source gives. A section the snapshot leaves out may say why (Daylight with no location). The release line is the build's: "Lookout <version> · <sha>", the repo's `VERSION` and the short SHA ("local" outside a Vercel build).

**Use.**
- `source` type in `ink-3`; each section's name weight 500 in `ink-2`; each entry at most 62ch. One column on a phone, two from 760px.
- The archive time is not repeated here: the header's freshness line says it once.
- Overview tiles and zooms carry no source note; the Rain, Rain events and Solar tabs keep theirs.

Static rendition of `web/src/overview/sources.ts`, `sources.css` and `web/src/release.ts`.
