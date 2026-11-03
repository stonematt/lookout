# The station inventory lives on Storj and only a human command writes it

## Status

accepted (2026-11-02). Decided in the grilling of "Serve a second station:
Salmon Run" (#179). Builds on
[ADR-0004](0004-failed-storage-read-aborts.md), which makes adding a station
a human act.

## Context

Lookout archives one station, pinned twice: `STATION_MAC` in the Python
settings and again in the Vercel env. Salmon Run, a second station on the
same Ambient account, installs on 2026-11-03. Stations are added almost
never, so adding one should take little ceremony, but the refresh run and
the site must agree on the list, its order, and which station is the
default.

## Decision

The station inventory is one small object on Storj: an ordered list of
station MACs. Its order is the priority order, and the first station is
the default station.

Only a manual command writes it. `stations scan` lists the account's
devices and marks which are in the inventory. `stations add <MAC>`
backs the inventory up, appends the station, and gives it an archive. If
an archive already exists for that MAC, the command adopts it and never
rebuilds it. If none exists, it builds the first archive from Ambient's
history.

The refresh run only reads the inventory. A missing inventory, or a
station in it with no archive, is an error, as ADR-0004 requires. A lost
archive is never silently rebuilt from the API, which would discard the
stored data.

The site never reads the inventory. The refresh run publishes a station
index to `snapshot/`, in inventory order, and the site reads that. The
`STATION_MAC` setting and the Vercel pin are removed.

## Considered options

- **A settings list (`STATIONS` in env or the secrets file).** Rejected: it
  must be kept in step on the VPS and Vercel, and an edit to it is not a
  logged, backed-up act.
- **A file in the repo.** The cron's `git pull` would deploy it, but each
  station added costs a commit and a release, and the site would still
  need its own copy.

## Consequences

The first `stations add` of the existing station seeds the inventory.
Until then the refresh run fails, so seeding happens in the same ops step
that deploys this change.
