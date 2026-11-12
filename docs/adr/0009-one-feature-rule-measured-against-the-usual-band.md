# One Feature Rule picks what the Overview and the widgets feature, measured against the usual band

## Status

accepted (2026-11-11). Decided with Matt in the Mosaic Overview grill
(issue #426).

## Context

Two rules in the snapshot ranked "what matters now", and the Mosaic Overview
would have added a third. The **lead** (`lookout/snapshot/lead.py`, the
Adaptive Lead Rule) picked the hero's reading with fixed thresholds: 10°F
from this hour's month median, 400 W/m² of solar. The **widget view**
(`lookout/snapshot/widget.py`) picked a widget's picture with its own list.
The two already disagreed, and each fixed threshold measured a different
quantity on its own scale, so wind could never be ranked against pressure.
The mosaic replaces the hero with the usual band chart, which takes the
lead's job away.

## Decision

The refresh run picks one **feature**, by the Feature Rule, checked in
order, first that holds:

1. **Rain**: an event is live, or ended in the last 12 h.
2. **Unusual**: a reading is below the 10th or above the 90th percentile of
   its readings at this hour over ±7 days around this date in earlier years
   (the window behind the **usual band**). Temperature, humidity, wind gust,
   pressure and solar qualify; the one furthest out by percentile wins. A
   station with no earlier year skips this rule: its recent-days band is too
   thin to call anything unusual.
3. **Long dry spell**: the current dry spell is longer than the usual dry
   spell.
4. **Quiet**: otherwise temperature, as a usual day.

The Overview's feature tile shows it. The widget view is derived from it,
not decided apart: rain → `rain_event`, dry spell → `dry_spell`, anything
else → `temperature_ranges`.

## Considered options

- **Keep fixed thresholds per quantity** (the Adaptive Lead Rule). Rejected:
  no common scale, so the order of the rules decides the ranking, and a
  threshold fit for one station's climate is wrong for another's.
- **Always feature the seasonal temperature chart on a dry day.** Rejected:
  it hides the day that is unusual for a reason other than temperature.
- **A band edge at the middle half (25th/75th).** Rejected: half of all
  hours would be "unusual", so the feature would change constantly.

## Consequences

- Solar features only when it is unusual for this date, never just for
  being bright.
- The lead and the widget view stop being separate terms once the feature
  ships. The snapshot adds `feature` beside `hero_lead` (no version bump,
  ADR-0008 §4); removing `hero_lead` later is a bump and ships in two
  releases.
- A widget can only draw three views, so a feature it cannot draw (wind,
  pressure) maps to `temperature_ranges`.
