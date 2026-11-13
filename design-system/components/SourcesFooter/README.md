The sources footer holds the Overview's provenance, once, at the page foot under the grid. It sits on `ground`, not a tile, under a `rule` top: a "Sources" heading (`h2`, `kicker` type), then one short line per drawn section in page order, then one closed `details`, "How these are worked out", holding the method paragraphs, then the release line.

**Consumer provides.** For each section (Right now, The feature, each mosaic tile): its name; a short line naming the source and the comparison, at most about 12 words ("outdoor sensor, against ±7 days around this date in earlier years."); and its method paragraph for the disclosure. A section the snapshot leaves out may say why in its line (Daylight with no location), with no method. The release line is the build's: "Lookout <version> · <sha>", the repo's `VERSION` and the short SHA ("local" outside a Vercel build).

**Use.**
- `source` type in `ink-3`; each section's name weight 500 in `ink-2`, run in before its line ("Temperature: …"); each entry at most 62ch. One column on a phone, two from 760px.
- The disclosure uses the source-note fold form (chevron, 32px target) and is closed on load. The method text may name percentiles; the findings never do.
- At 320px the closed footer is shorter than one screen.
- The archive time is not repeated here: the header's freshness line says it once.
- Overview tiles and zooms carry no source note and none of the footer's text; the Rain, Rain events and Solar tabs keep theirs.

Static rendition of `web/src/overview/sources.ts`, `sources.css` and `web/src/release.ts`.
