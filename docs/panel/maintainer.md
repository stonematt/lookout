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
