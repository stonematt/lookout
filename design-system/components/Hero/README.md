The hero is the now tile, the Overview's fixed top and the only tile on `now-surface`: the live temperature in `display` with its "°F", the one-line delta on usual for this hour under it ("−1° on usual for 11 pm", `finding` type in `ink-2`, tabular), and today hour by hour against the usual band for this date. On a phone the temperature sits above the chart; on the grids, beside it.

**Consumer provides.** All from the snapshot: the live temperature, the delta on usual, the band (the middle half and median of each station-local hour's readings) with its basis (earlier years, the station's recent days, or the median only), today's 15-minute series, today's rain by quarter hour, and the slots the archive cannot speak for.

**Use.**
- The chart runs midnight to midnight. The band runs to midnight as measured history, never a forecast. A label under the axis words its basis.
- The band is context: `temp-band` with no edge. The dashed median draws in `ink-on-now` (`median` is too dark on the dark now surface), and today's line in `ink`, ending at the r5.5 `now` dot. Its label, "now 39°", is `annotation` in `now` on a 4px `now-surface` halo.
- Where the live reading is ahead of the archive, a dotted line joins them.
- Rain stands as quarter-hour bars in `ribbon-rain` on a track along the foot, heavier rain a taller bar, never over the band. A slot the archive cannot speak for is hatched in `ink-3`.
- A snapshot with no band draws today's line and rain alone.
- **Rain column**, only when rain today (the live total) or yesterday is above 0. On the grids, a third column opposite the temperature headed "Rain": two `ribbon-rain` bars, Today and Yesterday, on one scale, values with their unit on top, on the chart's x-axis line with names on its x-label line. On a phone, one line under the delta, "Rain today 0.30 in · yesterday 0.12 in". A total not known yet reads "n/a" with no bar. Dry, neither is drawn and the chart keeps the width.

Static rendition of `web/src/overview/first-screen.ts`, `today-hero.ts` and `first-screen.css`; the Rain column is `web/src/rain/hero-rain.ts` and `hero-rain.css`.
