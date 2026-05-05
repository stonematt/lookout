# Station-local time is the analytical frame; `dateutc` is sort machinery

## Status

accepted (2026-05-04)

## Context

Every weather-station row carries three timestamp-shaped fields from the
Ambient Weather API:

- `dateutc` — int64 epoch ms UTC
- `date` — ISO 8601 UTC instant string
- `tz` — IANA timezone name of the **lookout post** that emitted the row
  (e.g. `"America/Los_Angeles"`)

When an operator asks "what was yesterday's high?" or "how many days since
the last rain?", *yesterday* means the calendar day in the station's local
zone, not in UTC. UTC midnight is meaningless to a person looking at a
backyard weather station. The station's local zone is the natural unit for
display, grouping, and day-arithmetic.

`dateutc`, by contrast, exists for two narrow technical purposes: a stable
sort key and a deduplication key. It survives parquet round-trips losslessly
(see [ADR-0001](0001-dateutc-post-2020-floor.md)) and gives unambiguous row
ordering across DST transitions. It is **not** the unit operators reason in.

## Decision

1. The **station-local zone** is the analytical frame for all
   day-granularity logic: grouping, "days since X" arithmetic, slider
   boundaries, calendar plots.
2. `dateutc` is the canonical sort + dedup key. It is converted to local
   datetimes only at the boundary (display, grouping, slider input).
3. Conversion helpers in `lookout/utils/dateutc.py` accept a `tz`
   parameter that defaults to `DEFAULT_TZ = "America/Los_Angeles"`. Today
   there is a single lookout post and that zone is correct.

## Deferred work (not this PR)

Held back from this PR to keep the dateutc-normalizer scope tactical (no
archive schema change, no catalog touches). Listed in priority order — the
near-term consolidation work matters more than the multi-station shape.

**High priority — finish the consolidation:**

- **Solar pipeline migration.** `core/solar_energy_periods.py` and
  `core/solar_data_transformer.py` build their own tz-aware datetime
  upstream of where `dateutc.py` would help. Migrate them onto the new
  helpers and drop the inline `tz_localize` / `tz_convert` boilerplate.
- **Energy catalog migration.** The catalog parquet stores naive
  datetimes that get re-localized on read. Reshape so the catalog stores
  int64 ms `dateutc` (matching archive), then derive tz-aware columns
  through `dateutc.to_local`. Removes the duplicated re-localization
  pattern at lines 195–200 of `solar_energy_periods.py` and similar
  sites.

**Lower priority — multi-station / tz persistence:**

- Stop discarding `tz` at ingest; persist as archive column.
- Backfill existing rows with `"America/Los_Angeles"`.
- Add row-aware helpers (`to_local_per_row(df)`) that read `df["tz"]`
  instead of taking a `tz` parameter.

The Ambient API's `tz` field is consumed at ingest and discarded today.
That is a known limitation, not a permanent shape — but no concrete
station relocation or second-station purchase is on the horizon, so the
single-post default (`DEFAULT_TZ = "America/Los_Angeles"`) is safe for now.
Revisit when a real second post is acquired or a relocation is planned.

## Considered alternatives

- **Persist `tz` and adopt per-row helpers in this PR.** Rejected:
  collapses the "tactical, low-risk" framing into the strategic refactor
  that was deliberately deferred.
- **Treat UTC as the analytical frame and convert at display only.**
  Rejected: forces every grouping / day-arithmetic call site to repeat
  tz-conversion boilerplate and gets day boundaries wrong on either side
  of midnight.
- **Compute station-local zone from device metadata at runtime instead of
  storing it.** Rejected: same fragility as today and obscures the
  intent that the *post* owns its zone.
