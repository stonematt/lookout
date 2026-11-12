# One Feature Rule picks what the Overview and the widgets feature, measured against the usual band

## Status

accepted (2026-11-11). Decided with Matt in the Mosaic Overview grill
(issue #426). Amended 2026-11-12 (#533): wind gust leaves the unusual test,
and the feature is held for 3 hours, after a replay showed it flipping
between reloads.

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
   (the window behind the **usual band**). Temperature, humidity, pressure
   and solar qualify; the one furthest out by percentile wins. Wind gust
   does not (amended, #533): it stays in the snapshot for the wind tile,
   but cannot win the feature until the wind method has had its own look.
   A station with no earlier year skips this rule: its recent-days band is
   too thin to call anything unusual.
3. **Long dry spell**: the current dry spell is longer than the usual dry
   spell.
4. **Quiet**: otherwise temperature, as a usual day.

**The hold** (amended, #533). The feature's subject is its kind, and for
unusual its reading; the feature carries when its subject was first
chosen. When the rule picks a different subject, a subject chosen under
3 hours ago is kept and marked held, unless the rule now picks rain, which
takes over at once. Rain is not held itself: its 12 hours after an event
already are its hold. A held feature's values are recomputed every build,
so the tile never shows an old number; a held unusual reading back inside
its band says so, and one that cannot be judged this build is not held.
The refresh run reads the last subject from the last published snapshot;
with none, nothing is held.

The Overview's feature tile shows it. The widget view is derived from it,
not decided apart, and follows a held feature: rain → `rain_event`, dry
spell → `dry_spell`, anything else → `temperature_ranges`.

## Considered options

- **Keep fixed thresholds per quantity** (the Adaptive Lead Rule). Rejected:
  no common scale, so the order of the rules decides the ranking, and a
  threshold fit for one station's climate is wrong for another's.
- **Always feature the seasonal temperature chart on a dry day.** Rejected:
  it hides the day that is unusual for a reason other than temperature.
- **A band edge at the middle half (25th/75th).** Rejected: half of all
  hours would be "unusual", so the feature would change constantly.
- **No hold, the 10th/90th edges alone** (as first accepted). Rejected by
  a read-only replay of the rule over Lookout Post's last 60 days at
  15-minute steps: the feature changed a median of 20 times a day (48 on
  the worst), with 499 A→B→A bounces inside 30 minutes, the flip Matt saw
  between reloads. Wind gust was 8% of builds but on one side of 58% of
  the changes and in 45% of the bounces, and wind at this station is too
  inconsistent for its usual band to mean much yet. Salmon Run changed 6
  times in about 8 days and never went unusual.
- **A 1-hour hold.** Rejected for 3 hours: the replay gave 1 hour a median
  of 11 changes a day (18 on the worst) and no bounces, about 7 minutes'
  extra delay before a new feature shows; 3 hours gave a median of 5 (8 on
  the worst) and no bounces, about 12 minutes. The replay's 15-minute
  steps understate changes slightly, and its delay figures are approximate.

## Consequences

- Solar features only when it is unusual for this date, never just for
  being bright.
- The lead and the widget view stop being separate terms once the feature
  ships. The snapshot adds `feature` beside `hero_lead` (no version bump,
  ADR-0008 §4); removing `hero_lead` later is a bump and ships in two
  releases.
- A widget can only draw three views, so a feature it cannot draw
  (pressure, humidity, solar) maps to `temperature_ranges`.
- The snapshot reads its own last published document: a failed read fails
  the station (ADR-0004), so the last snapshot stays up rather than a
  build that silently drops the hold. The hold adds fields only (no
  version bump, ADR-0008 §4); a snapshot published before it carries no
  choice time and holds nothing.
