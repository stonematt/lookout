The 48-hour condition ribbon is a 44px band of 15-minute slots, each night (`ribbon-night`) or day (`ribbon-day`), with the temperature traced over it and a `now` dot at the right edge for the live value, then a separate rain track under the band. It lives on the now tile.

**Consumer provides.** Per slot: daylight, rain, temperature; the live value; the one-line claim above it ("40° to 55° over the last 48 hours, with rain in 9 h of the 47 h on record.").

**Use.**
- Rain is its own track in `ribbon-rain`; its height gives intensity in four steps. Never overprint rain on the band: no blue passes 3:1 on both night and day and still tells light from heavy rain.
- The temperature is a 2px `ink` line on a 5px halo in the surface colour; the dot is `now` with a 2px `now-surface` ring, labelled "now 47°" in `annotation`.
- Spans the archive cannot speak for (a gap, or the span only the live reading covers) are hatched in the band and the track, joined to the trace by a dotted line.
- In light mode the band is outlined in `ribbon-edge`. Under 520px of band width the ticks name only the days; on a phone the ribbon runs edge to edge in its tile with a 0 radius.

Static rendition of `web/src/overview/ribbon.ts` and `first-screen.css`.
