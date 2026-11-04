# ConditionRibbon
A 52px band of the last 48 hours in 15-minute slots: `night` or `sun-soft`, overprinted by the rain ramp, with the temperature traced over it.

- The trace is a 2px `ink` line on a 4.5px `ground` halo, ending in a `note` dot labelled "now".
- The unarchived span at the right edge is hatched. Ticks every 6 h ("6 am", "noon", weekday).
- A one-line claim above gives the range and the rain hours.
- Proposed for phone: ticks every 12 h under 520px; press-and-drag scrub writes the slot into the claim line.
- Source: `web/src/overview/ribbon.ts`.
