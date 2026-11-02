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
- [ ] fix: refresh runs can overlap; no lock in `cron_run.sh`, and its `--sleep` default is 900 s — #244
- [ ] note: a missing `HC_PING_KEY` only warns, so the cron can run without alerts — `lookout/utils/healthchecks.py`
- [ ] note: the dashboard and context documents publish one after the other; a failure between them leaves them out of step until the next run (the run exits 1) — `lookout/snapshot/publish.py`
- [ ] note: streamlit import at `lookout/core/data_processing.py:9` — not rechecked; the refresh path does not import it
- [ ] note: socket guard in `tests/conftest.py` — not rechecked
- [ ] note: the Vercel Storj key's scope is still not verifiable from code — still open
