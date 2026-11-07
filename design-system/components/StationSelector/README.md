The station selector is the header's station name. With two or more stations in the station index it is a native `select` styled as the weight-600 name with a chevron after it; with one station the name is plain text with no chevron, so the header looks the same either way. Beside it is the live line: a dot, "Fri Nov 6, 7:40 am" and the age.

**Consumer provides.** The station names (an unnamed station shows the end of its MAC), the reading time and its age.

**Use.**
- A long name stays on one line and ends in an ellipsis: at most 28ch on desktop, and on a phone the header width the live line leaves. The open list shows every name in full. The live age never truncates. The select is at least 1rem so iOS does not zoom on focus.
- Staleness is past 15 minutes: the live dot goes hollow and the age sits in a hatched chip, with its stripes in a swatch beside the text.
- The 28ch cap is in the design but the shipped stylesheet only caps the select at its container, so this preview truncates by width.

Static rendition of `web/src/header.ts` and `web/src/styles/base.css`.
