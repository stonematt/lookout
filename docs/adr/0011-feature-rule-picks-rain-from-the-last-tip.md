# The Feature Rule picks rain from the last tip, not from the rain event

## Status

accepted (2026-11-23). Decided with Matt in the "rain should feel like
Oregon" grill (issue #647). Replaces clause 1 of
[ADR-0009](0009-one-feature-rule-measured-against-the-usual-band.md); the
rest of ADR-0009 stands.

## Context

ADR-0009 made rain the feature while an event is live, or for 12 hours after
it ends. A rain event's bounds are Ambient's: its event counter and its own
end rule. Ambient can stretch one event over a day and a half of on-and-off
rain. The 12-hour tail then puts a stale headline under the temperature,
such as "Rain ended 9 h ago · 0.17 in over 1 d 8 h, against 0.20 in for a
usual event". That line does not answer what a visitor opens the site for
after hearing rain: is it raining, how hard, and since when.

## Decision

Clause 1 of the Feature Rule becomes:

1. **Rain**: the gauge tipped in the last 3 hours (**last rain** within
   3 h).

**Last rain** is the time of the archive's last gauge tip, after carryover
corrections. It is not Ambient's `lastRain` field. It is **raining** when
last rain is within 30 minutes. The rain feature states the rate (the rain
in the last 60 minutes), and the event's start and total as context; the
event no longer decides whether rain leads. The 3-hour window matches the
feature's 3-hour hold, and rain still takes over a held feature at once.

The dry spell clause uses whole dry days (GLOSSARY **dry spell**) on every
reader, so a pause between showers never reads as a dry spell.

## Considered options

- **Keep the event as the trigger with a shorter tail.** Rejected: a live
  Ambient event can run for a day with hours of no rain, so "live" does not
  mean "raining".
- **Ambient's `lastRain` for last rain.** Rejected: wet hours and the wet
  share come from the archive, and the now story should come from the same
  readings. The refresh run brings the archive up to date before every
  build, so it is as fresh.
- **A 30-minute window for the feature.** Rejected: drizzle can go 30 to 60
  minutes between tips, and a feature that drops out between tips would
  flip. 92% of gaps between tips inside a rainy period are 30 minutes or
  less (back-test on the Lookout Post archive, 2026-11-23), which sets
  "raining", not the feature window.

## Consequences

- The snapshot gains last rain, and the feature's rain values change. This
  ships inside the schema 5 bump for #647 (ADR-0008: the site reads 5
  first, then the refresh run writes 5).
- `DrySpellNow.days_since_rain` stops reading `lastRain` and counts whole
  dry days.
- The widget view follows the feature, so widgets show rain on the same
  rule.
