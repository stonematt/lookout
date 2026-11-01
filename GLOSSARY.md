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

Its ascending order is not the stored order. The archive in storage is
newest first, whichever command saved it: every writer saves through
`storj.save_archive`, which sorts it so. Readers that need the latest
reading take the greatest `dateutc` (`trend_utils.newest_row`), not the
first row.

### station

One physical Ambient Weather station, keyed by its MAC address. Each
station owns its own IANA timezone (the API's `tz` field, e.g.
`"America/Los_Angeles"`). All operator-facing day-granularity reasoning
("yesterday's high", "days since last rain") is in the station's local
zone — the station is the **timezone-aware unit**, not the dataframe or
the codebase. See
[ADR-0002](docs/adr/0002-station-local-time-as-analytical-frame.md).

Stations have names. **Lookout Post** is the one station today; a second,
**Salmon Run**, is coming at a new location. Lookout, the product, is the
dashboard, not a station.
_Avoid_: "lookout post" or "post" as the general term.

**Known limitation:** the station's `tz` is currently consumed at ingest
and discarded. Existing archives do not persist a `tz` column. `DEFAULT_TZ =
"America/Los_Angeles"` is hardcoded as the single-station default. Persisting
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

### water year

The rain season Lookout totals over: Oct 1 through Sep 30, in the station's
local zone, named by the year it ends in. Season totals,
year-to-date rain, the almanac's year rows and year-over-year accumulation
all use it, so a winter's rain is never split across two years.
_Avoid_: rain year, season year, calendar year (for rain totals).

### snapshot

The precomputed state of the dashboard for one station: what each tab
shows, built by the refresh run from the archive, the live reading and the
catalogs, and published to Storj as JSON under `snapshot/` (ADR-0003). The
Vercel site reads it and computes nothing itself. Its full form is the
**dashboard document** (`snapshot/<MAC>.dashboard.json`), versioned by
`schema_version`; the contract rules live in `lookout/snapshot/model.py`.
_Avoid_: cache, export, payload.

### context block

The small part of the snapshot that `/current` serves next to Ambient's live
`lastData`: the station, the build and observation times, current conditions
and the active rain event (`CONTEXT_FIELDS`). Published as
`snapshot/<MAC>.context.json`, so a live reading always arrives with the
context it is read against.
_Avoid_: context file, mini snapshot.

### gap status

The verdict on one month or window of the archive before any value is
computed over it: **valid** (every day counted), **prorated** (a few days
missing, at most 10%: a total is scaled over the missing days and shown with
`≈`) or **invalid** (too much missing: the value is null, never computed, and
the span is hatched). A day counts when at least 80% of its 5-minute
readings are archived. Every value over a month or window goes through it
(`lookout.snapshot.gaps.comparable`).
_Avoid_: coverage (the readings-per-hour grid), data quality, completeness.

### archive

A station's stored readings, one row for each `dateutc`. It holds only
what Ambient sends; Ambient's own computed fields (feels-like, dew point)
count as raw. Values Lookout derives, such as the hour or a wind bin, are
never stored: readers compute them. The archive is the one thing Lookout
cannot regenerate, so no routine write may drop a stored reading or blank
a stored value. Only an explicit repair may.
_Avoid_: history, the parquet, the data file.

### archive merge

How fetched readings join the archive. Ambient is the source of truth:
for each value Ambient sends, Ambient's value wins; where Ambient sends
nothing, the stored value stays. A merge reports readings **added**,
values **filled** (stored had none) and values **updated** (Ambient's
differs). A deliberate correction is a **replace**, never a merge.
_Avoid_: combine, overwrite, "fetched wins".

### Event Catalog

The stored list of rain events detected from a station's archive. It
is derived: detection can always rebuild it from the archive. A rebuild is
not free, though: history that detection no longer reproduces, such as
events shaped by earlier rain corrections, is lost unless a backup is kept.
So a rebuild is a human act, never a side effect of a failed read.
_Avoid_: rain catalog, events file.

### read outcome (found / missing / failed)

Every storage read ends in exactly one of three outcomes:

- **found**: the object exists and was read.
- **missing**: storage says the object does not exist.
- **failed**: we could not learn whether the object exists, or it exists
  but cannot be read (network error, timeout, corrupt file).

Missing and failed are never interchangeable: a failed read is never acted
on as "nothing stored". A missing archive is an error too: adding a station
is a human act, never inferred from an absent archive. See
[ADR-0004](docs/adr/0004-failed-storage-read-aborts.md).
_Avoid_: "empty" for either missing or failed.

### curve file

A completed rain event's readings, as its accumulation and rain rate at
each archived reading, written once to Storj as
`snapshot/<MAC>.events/<start_ms>.json` and never rewritten, since a
completed event's readings do not change. The ongoing event's curve rides in
the snapshot instead. Keyed by the event's start, not
its catalog `event_id`, which an ongoing event does not keep.
_Avoid_: event file, event chart data.

### lead

The one reading the Overview's hero puts first: rain, temperature, solar or
calm, picked by the **Adaptive Lead Rule** (DESIGN.md), whose rules are
checked in that order and the first that holds leads. Rain leads while an
event is live or ended in the last 12 h; temperature at 10°F or more from
this hour's month median; solar at 400 W/m² or more; otherwise calm, the
days since rain against the usual dry spell.
_Avoid_: headline, hero metric, featured stat.

### dry spell

A run of dry days with a rain day (at least 0.01 in, one tip of the gauge)
on each side. A day the archive cannot count breaks the run, so a gap never
reads as dry. The **usual dry spell** is the median completed dry spell over
the last 365 days. One definition serves the Rain tab and the calm lead
(`lookout.core.rainfall_analysis.dry_spells`).
_Avoid_: drought, days without rain (for the spell itself).
