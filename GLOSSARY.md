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

The station's zone is Ambient's metadata for that station, read on every
build and published with the snapshot. A station Ambient sends no zone
for does not build; its zone is never guessed. Archive rows carry no zone:
an archive belongs to one station, so a stored zone matters only if a
station moves. Both stations are in Pacific time. See
[ADR-0005](docs/adr/0005-station-time-zone-from-ambient-per-build.md).

### station inventory

The ordered list of stations Lookout archives. Its order is the priority
order. A person adds a station to it; nothing adds one automatically. See
[ADR-0006](docs/adr/0006-station-inventory-on-storj-written-by-hand.md).
It is stored at `station_inventory.json`, outside `snapshot/`, and only
`stations add` writes it, after a backup.
_Avoid_: "station list", "STATIONS".

### station index

The published list of stations the site can show, in inventory order,
with each station's MAC, name, zone, newest reading and archive start. The
refresh run publishes it last, at `snapshot/stations.json`, after every
station; a station whose run failed keeps its entry from the last good
index, and one that has never published is not listed. The site reads it,
never the station inventory, and serves `?station=` only for a station it
lists. It has its own `schema_version`, outside the snapshot's
(`lookout/snapshot/station_index.py`).

### default station

The first station in the station inventory, shown when no station is
chosen: Lookout Post. A job that works on one station and is not told which
works on the default station (`station_inventory.resolve_station`); there
is no pinned MAC.

### station selector

The dropdown on the dashboard that chooses which one station every tab
shows. It lists each station by name. It never mixes stations: a view
that combines stations is the blended dashboard, which comes later.
_Avoid_: "navigator", "station picker".

### blended dashboard

A later view that combines stations. Not built yet.

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
the active rain event and the widget view (`CONTEXT_FIELDS`). Published as
`snapshot/<MAC>.context.json`, so a live reading always arrives with the
context it is read against.
_Avoid_: context file, mini snapshot.

### status document

The JSON the refresh run writes at `snapshot/<MAC>.status.json`, one per
station in the station inventory, at the end of every run it does not skip,
success or failure; a run that cannot open the archive-writer lock file is a
failed run and writes them too. It names the code version and checkout SHA
the run started on, that station's outcome (`ok` or `failed`: one station's
failure leaves the others' documents `ok`, while the run exits 1) and, on
failure, the exception class; the site serves one at `/status`. A run
skipped for a held lock writes none. It is not part of the snapshot, so
`SCHEMA_VERSION` does not cover it (`lookout/snapshot/status.py`).
_Avoid_: heartbeat (that is the Healthchecks ping), run log.

### gap status

The verdict on one month or window of the archive before any value is
computed over it: **valid** (every day counted), **prorated** (a few days
missing, at most 10%: a total is scaled over the missing days and shown with
`≈`) or **invalid** (too much missing: the value is null, never computed, and
the span is hatched). The days it judges are **counted days**. Every value
over a month or window goes through it
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

### archive writer

A command that saves the archive or a catalog (the Event Catalog or the
energy catalog), of which at most one runs on a host at a time. The refresh
run, catchup, `diagnostics fill-gap`, `diagnostics energy-catalog --save`,
`diagnostics rain-catalog --save` and `repair_rain --write` are archive
writers; each takes the per-host archive-writer lock
(`lookout.utils.archive_lock`) before it does any work. A read-only command,
or a writer run without its save flag, is not one and takes no lock.
_Avoid_: saver, cron job (the refresh run is one writer among several).

### counted day

A station-local day the archive can speak for: at least 80% of its due
5-minute readings are archived. For rain, a counted day also needs a reading
of the daily rain counter; a counted day without one is a gap for rain, but
not for temperature or solar, so a dead rain gauge never hatches their
windows. A day that is not counted is unknown, never dry or calm. Rain
consumers judge it through the **rain day kind**.
_Avoid_: covered day, valid day (valid is a gap status, not a day).

### daily rain series

The maximum daily rain counter for each station-local day, extracted
once per snapshot build beside the **counted days**. Today's value comes
from the live reading's daily counter when the live reading is today. It
is the one place a day's rain amount is read, so the lead and the Rain tab
agree.

### rain day kind

The rain view over the **counted days** and the **daily rain series**: a
day is a rain gap when it is not counted, or counted with no counter value.
The Rain tab, the almanac, rain events and the calm lead judge their
windows through it; temperature, solar and the gaps section use the plain
counted day.
_Avoid_: rain coverage.

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

### widget view

Which view a widget shows now, picked in the snapshot so every widget shows
the same one (`lookout.snapshot.widget`). Checked in order: **rain_event**
while an event is live with more than 0.04 in in it; **dry_spell** in June
to September once it has been dry for 3 days or more; otherwise
**temperature_ranges**, now against today, 7, 30 and 365 days. It is not the
lead: the lead ranks the Overview's hero, the widget view picks a widget's
picture.
_Avoid_: widget mode, widget state.

### dry spell

A run of dry days with a rain day (at least 0.01 in, one tip of the gauge)
on each side. A day the archive cannot count breaks the run, so a gap never
reads as dry. The **usual dry spell** is the median completed dry spell over
the last 365 days. One definition serves the Rain tab and the calm lead
(`lookout.core.rainfall_analysis.dry_spells`).
_Avoid_: drought, days without rain (for the spell itself).
