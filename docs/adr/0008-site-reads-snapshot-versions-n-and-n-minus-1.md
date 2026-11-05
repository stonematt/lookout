# The site reads snapshot versions N and N−1, so a schema bump ships in two releases

## Status

accepted (2026-11-03). Decided with Matt in the uat-env grill on the
Tiles That Talk release (ledger #348, issue #352). Generalizes the one-off
fix #351 (PR #354). Fills in the "own rules" that ADR-0007 left to
`SCHEMA_VERSION`.

## Context

The snapshot contract is the JSON the refresh run on `itri` writes and the
site on Vercel reads. Its `SCHEMA_VERSION` used to be checked by strict
equality on both sides, and a breaking change bumped both in one change.
But the two deploy separately: the refresh run when `itri` pulls `main`
and the cron next fires, the site when Vercel builds `main`. A lockstep
bump means that for a few minutes one side writes a version the other
refuses, and production shows a contract error.

The v3 bump (#346, the sun and moon block) hit this. It also meant a
Vercel preview of a goal branch could not read the live snapshot until
production moved, so UAT on live data was blocked. #351 widened the
site's dashboard and context checks to read 2 and 3, and that shipped
alone as release 2026.11.5, before the refresh run wrote 3.

## Decision

1. **The site accepts `SCHEMA_VERSION` N and N−1** for the dashboard and
   context documents (`SNAPSHOT_SCHEMA_VERSIONS` in `web/src/snapshot.ts`).
   A field that N added and N−1 lacks reads as null.
2. **The refresh run moves to N only in a release after one whose site
   already reads N.** The reader must lead: an old site cannot know a
   version that does not exist yet. So a schema bump ships in two
   releases:
   - Release A: the site reads N−1 and N. The refresh run still writes
     N−1.
   - Release B: the refresh run writes N.
   Release A may carry other work. The two may ship the same day, but
   B's deploy waits until A's site is live in production.
3. **N−2 support drops once no N−1 document is live**, in any later
   release. Narrowing the list is a site-only change.
4. **What counts as a shape change** stays as `lookout/snapshot/model.py`
   states it. Adding a field keeps the version. Renaming, removing or
   changing the meaning of a field bumps it. Correcting a wrong value
   (now null as unknown, or once null now always sent) keeps it.
5. **Exceptions.**
   - Curve files are never rewritten, so the site reads every version in
     `CURVE_SCHEMA_VERSIONS`, not only N and N−1.
   - The `/current` envelope stays strict at N. The site's own server
     builds it from the same `SCHEMA_VERSION` and ships in the same
     Vercel deploy as the page that reads it. The context block inside it
     goes through the N and N−1 check.

## Considered options

- **Lockstep bump, deploy the refresh run first** (the old `CLAUDE.md`
  rule, "When `parseDashboard` tightens, the refresh run deploys before
  the site"). Rejected: it orders the race but does not remove it, and it
  still blocks preview UAT on live data until production moves.
- **Never bump; only add fields.** Rejected: a rename or a change of
  meaning would have to keep the old field forever, and the site could
  not tell an old document from a broken one.
- **The site reads every past version.** Rejected for the dashboard and
  context: unlike curve files, they are rewritten every 5 minutes, so an
  old version is gone from storage one run after the refresh run moves.

## Consequences

A breaking snapshot change costs one extra release, and the release
manager plans the pair. A goal branch can UAT on live data once release A
is in production. The parser test should read N−1 from one named
constant, so the next bump changes that constant and the test keeps
guarding the rule.
