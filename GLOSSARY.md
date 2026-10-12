# Lookout — Domain Context

Glossary of load-bearing terms. Add entries as they crystallize during design
sessions. Keep entries terse and meaningful to a domain expert (weather data
operator, not language nitpicker).

## Glossary

### dateutc

Canonical timestamp column on every weather-station row. Stored as **int64
epoch milliseconds, UTC**. This is the one true timestamp; alternative names
(`timestamp`, `ts`, `date`, `dt_utc`) are rejected — if seen on input, they
get renamed to `dateutc`.

Why int64 ms (not tz-aware datetime): parquet round-trip stability. April 2026
incident: pandas 3.0 silently changed parquet timestamp resolution and
collapsed real ms values to fractional ones — int64 ms persists losslessly.

Display conversions (to UTC datetime, to Pacific datetime) are derived at the
boundary; the stored representation is always int64 ms.

### normalize (dateutc normalizer)

The single function that validates and casts a DataFrame's `dateutc` column
to canonical form: int64 ms UTC, drops invalid rows (NaN, ≤ 0, below the
post-2020 floor), de-duplicates, sorts ascending. Lives at
`lookout/utils/dateutc.py`. All archive ingest and load paths go through it.

### lookout post

A physical Ambient Weather station. Each post owns its own IANA timezone
(the API's `tz` field, e.g. `"America/Los_Angeles"`). All operator-facing
day-granularity reasoning ("yesterday's high", "days since last rain") is
in the post's local zone — the post is the **timezone-aware unit**, not
the dataframe or the codebase. The Ambient account today owns a single
post; the API supports multiple. See
[ADR-0002](docs/adr/0002-station-local-time-as-analytical-frame.md).

**Known limitation:** the post's `tz` is currently consumed at ingest and
discarded. Existing archives do not persist a `tz` column. `DEFAULT_TZ =
"America/Los_Angeles"` is hardcoded as the single-post default. Persisting
`tz` is **lower priority** than finishing the dateutc consolidation in
the solar pipeline and energy catalog — see ADR-0002 deferred work.

### post-2020 floor

`POST_2020_MS = 1577836800000` (2020-01-01 UTC, ms). Minimum legal
`dateutc`. Rows below the floor are treated as corrupt and dropped at
normalize time. The floor exists because small positive integers (e.g.
1–28,800,000 ms) survive a naive `> 0` guard but map to 1969-12-31 PT after
tz conversion and collapse downstream date ranges. See
[ADR-0001](docs/adr/0001-dateutc-post-2020-floor.md) for why the floor
stays even after pinning `pandas < 3`.
