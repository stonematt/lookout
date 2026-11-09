The almanac grid has years as rows and months as columns, cells filled from the rain ramp (or the sun ramp for solar) with the value in the cell. Cells are 44px, 3px corners, 3px gaps.

**Consumer provides.** For each year and month: the value, its ramp step (square-root scale, decided in Python), and its gap status.

**Use.**
- Cell text uses the matching `on-rain-N` or `on-sun-N`, never a fixed colour.
- The One Gap Status Rule: valid cells show their value; `≈` marks a prorated month; an invalid month is hatched with "n/a" on a halo; a month before the archive began or still to come is a dashed `rule-strong` outline with nothing inside.
- The current month is filled from the ramp by its total so far and outlined in `now` outside the cell with a 1px gap; it is hatched only when it has no total so far.
- The selected cell is ringed 2px in `ink`. A key shows the ramp, the hatch and the dashed outline.
- Finding copy names the years, newest first.

Static rendition of `web/src/overview/almanac.ts` and `almanac.css`.
