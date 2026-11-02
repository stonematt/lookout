# The station's time zone comes from Ambient, per build

## Status

accepted (2026-11-01). Decided in the grilling of "Build the daily rain
series and coverage once, on a station clock" (#174). Amends the deferred
work in [ADR-0002](0002-station-local-time-as-analytical-frame.md).

## Context

ADR-0002 makes the station's local zone the frame for every day-granularity
reasoning, and hardcodes `DEFAULT_TZ = "America/Los_Angeles"` while there is
one station. It defers storing a `tz` column on every archive row, and it
rejects reading the zone from device metadata at runtime as fragile.

A second station, Salmon Run, is coming. It is in Pacific time too, so no
output changes. But the zone is copied as a constant into the Python
sections, the daily rain extraction and the site, and some sections take no
zone at all. Ambient sends the station's `tz` with every reading.

## Decision

1. **The zone is Ambient's metadata for the station.** The refresh run reads
   it from the device's latest reading on every build, and the station
   carries it through the build.
2. **The snapshot publishes it.** The station in the dashboard document and
   the context block names its zone. The site formats every time, day and
   month label in that zone, not in a constant. Adding the field keeps
   `SCHEMA_VERSION`.
3. **No zone, no build.** If Ambient sends no zone, the refresh run exits 1
   and publishes nothing; the last good snapshot stays up. `DEFAULT_TZ`
   remains only as a test default. This follows ADR-0004: a value we could
   not learn is never replaced in silence.
4. **Archive rows carry no zone.** An archive belongs to one station, so
   a stored zone matters only if a station moves. ADR-0002's per-row `tz`
   column and per-row helpers wait for a relocation.

## Considered options

- **Store `tz` on every archive row now** (ADR-0002's plan). Rejected: it
  changes the archive schema and backfills every row to record a value that
  is the same for the whole archive, while Ambient already sends it.
- **Pass a zone to the build and keep the site's constant.** Rejected: the
  site would keep its own copy of station metadata, against ADR-0003's
  "the site computes nothing itself".
- **Fall back to `DEFAULT_TZ` when Ambient sends no zone.** Rejected: a
  wrong zone shifts every day boundary in silence, which is worse than a
  frozen dashboard with a Healthchecks alert.
