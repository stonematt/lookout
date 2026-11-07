The gap hatch is the one visual word for "not valid": 135 degree stripes, `hatch-stripe` (2px) on a `hatch-repeat` (12px) of `hatch` over the surface colour, with a short reason in place.

**Consumer provides.** A reason string for each hatched span ("gap in record", "not archived yet", "n/a"), from the snapshot's gap status.

**Use.**
- Text never sits on the stripes. The reason sits on a pill in the surface colour with 4px of room round the glyphs, left-aligned, in `label` type and `ink-2`.
- On the now tile the stripes draw in `ink-3`, because `hatch` falls under 3:1 on `now-surface`.
- A hatched chip is a pill in `rule-strong` that starts with a 12px swatch (2px stripes on a 6px repeat) and then its reason in `ink-2` on the surface colour. Use it in a source note when the archive lags or a gap hides part of the chart.
- Hatch an invalid comparison and never compute it. An invalid month or window is shown as hatched, with its reason, in every chart, table and finding.

Static rendition of `web/src/gaps.ts`, `web/src/styles/gaps.css` and `base.css`.
