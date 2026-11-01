# A failed storage read aborts; a missing object means something different for each kind

## Status

accepted (2026-11-01). Decided in the grilling of "Storage reads tell
found, missing and failed apart" (#172).

## Context

Every Storj read has three outcomes: found, missing and failed (see
**read outcome** in GLOSSARY.md). Several Python readers collapsed
"failed" into "empty". The worst case was the Event Catalog. A transient
read error made the refresh run re-detect every event and overwrite the
stored catalog. If the existence check failed too, it skipped the backup.
The run still exited 0.

## Decision

1. **Failed always aborts the refresh run.** The run exits 1, writes
   nothing, and the last good snapshot stays up. This holds for the
   archive, the rain corrections catalog, the Event Catalog and the
   energy catalog. The single exception is the curve listing, which runs
   after publish. It cannot stop the snapshot, but it still exits 1 so
   Healthchecks alerts.
2. **A corrupt object counts as failed.** The run never rebuilds it on its
   own. A human rebuilds the Event Catalog with a `diagnostics` command
   that backs up first and writes only with `--save`.
3. **Missing means something different for each kind of object.**
   - Archive: missing is an error. Adding a lookout post is a human act,
     never inferred from an absent archive. A wrong MAC or bucket prefix
     must not start a fresh archive in silence.
   - Derived catalogs: missing is legal. The run detects from the archive
     and saves, because storage has confirmed that nothing is there to
     lose.
4. **Shape.** A reader returns the value when found, `None` when missing,
   and raises when failed. This matches the existing `read_json_from_s3`
   and `web/server/storj.ts`. `object_exists` is removed: a separate
   existence check doubled the round trips and turned "failed" into
   "missing".

## Considered options

- **Degrade and mark the snapshot stale** on a failed catalog read.
  Rejected: when the read fails there is no stored catalog to fall back
  on, so "stale" would mean "no catalog".
- **Rebuild a corrupt catalog automatically.** Rejected: rebuilding loses
  history that detection no longer reproduces, and a frozen dashboard
  with a Healthchecks alert is the safer failure.
- **Sealed `Found | Missing | Failed` result types.** Rejected for now:
  they add boilerplate at every pandas caller, and they gain little over
  `None` plus an exception.
