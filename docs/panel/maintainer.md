# Maintainer

**Lens:** the maintainer opening the refresh pipeline a year from now, after the cron failed silently and the dashboard went stale.

**Reviews with:** code-review, improve-codebase-architecture, codebase-design

**Seat when:** tests and CI (#67–#71), the snapshot and refresh run (#72–#76), retiring Streamlit (#86), and every architecture review the drift check triggers.

## What I care about here

- `build_snapshot`, the refresh command and the snapshot JSON can be understood from the code and ADR-0003 alone, without the git history.
- A failed refresh or a silent station shows up in Healthchecks before anyone notices stale data.
- The test suite is hermetic: no network, no real Storj, no Streamlit import in the core path.
- Characterization tests pin the behaviour of the catchup run, the archive merge and the Event Catalog before the code they cover changes.
- Secrets load from `~/.config/lookout/secrets.toml` in one place; the Vercel side holds only a read-only Storj credential.
- No second copy of the domain logic between Python and the front end; the front end reads the snapshot, it does not recompute.
- A writer that rewrites the archive prints its rollback backup key on stdout when it succeeds; a key only in the log is not enough (Matt, 2026-11-11, #451).

## What I let pass

- Visual design and chart craft — the station owner and the FT graphics editor.
- Whether a rain total is correct — the skeptic, unless a test that should catch it is missing.

## Verdict log

Newest at the bottom. Each review checks the open items above it and says whether they got fixed.

<!--
### YYYY-MM-DD — <artifact: PR #n, spec #n, prototype/<name>>
Verdict: <ship | fix | rethink>
- [ ] <blocker or fix, one line> — <where>
- [x] <item from an earlier review, now fixed> — fixed in <sha or PR>
-->

### 2026-10-31 — goal PR #96 (spec #66, Move Lookout off Streamlit), QA
Verdict: ship
- [ ] note: streamlit still imported at `lookout/core/data_processing.py:9` and in legacy CLIs that bypass `get_secret`; clear with Retire Streamlit (#86)
- [ ] note: no socket guard in `tests/conftest.py`
- [ ] note: the Vercel Storj key's scope is not verifiable from code; confirm it is read-only on `snapshot/`

### 2026-11-01 — release PR #241 (dev -> main, caf4f9f), release review
Verdict: ship
- [ ] fix: `cron_run.sh` takes no lock of its own, and its `--sleep` default is 900 s — #244. Not a live risk: the itri crontab wraps the run in `flock -n` with `--sleep 30` (recorded on #76)
- [ ] note: a missing `HC_PING_KEY` only warns, so the cron can run without alerts — `lookout/utils/healthchecks.py`
- [ ] note: the dashboard and context documents publish one after the other; a failure between them leaves them out of step until the next run (the run exits 1) — `lookout/snapshot/publish.py`
- [ ] note: streamlit import at `lookout/core/data_processing.py:9` — not rechecked; the refresh path does not import it
- [ ] note: socket guard in `tests/conftest.py` — not rechecked
- [ ] fix: the Vercel Storj key covers the whole `lookout` bucket, not `snapshot/` (Matt, 2026-11-01) — #247

### 2026-11-04 — goal PR #358 at `0abc3f2`, pre-UAT
Verdict: ship
- checked: lint clean (`no-unsanitized`), 731 web tests pass, CSP check 0 violations over 8 runs, `web/vercel.json` unchanged; no test files deleted
- [ ] note: ten-day colour decided in the web (`web/src/overview/temperature.ts:302`); move it to the snapshot with a `SCHEMA_VERSION` bump after UAT (#374)
- [ ] note: event rows lose ctrl-click open in a new tab (#370); lag arithmetic sits in three places off one web constant `STALE_AFTER_MS` (`web/src/format.ts:45`)
- streamlit import, socket guard, #247 scope, `HC_PING_KEY`, publish order — not checkable in a web-only diff; still open

### 2026-11-11 — goal PR #440 at `a2c7ed9`, archive store, pre-UAT

Verdict: ship
- checked: every archive writer saves through `lookout/storage/archive_store.py`; no `save_archive` or `backup_archive` call outside it and `storj.py`; refresh still reaches the run-level handler, the failed status and the Healthchecks ping on a store error
- [ ] fix: `docs/agents/operations.md` has no `catchup --repair-archive` entry (repair backup, read-back, exit 1)
- [ ] note: `operations.md:8` and `CLAUDE.md:28` name only the daily backup folder, not `<stamp>-repair-<reason>/`; `catchup.update_archive` keeps the deleted refresh function's name
- [x] #244 lock: now also checked at save time (`held_by_this_process`)
- streamlit import, socket guard, `HC_PING_KEY`, publish order, #247 scope, #374, #370 — not in this diff; still open
