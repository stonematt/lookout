# Contributing

## Setup

```bash
git clone https://github.com/stonematt/lookout.git && cd lookout
uv venv --python 3.12 .venv
uv pip install -r requirements.txt -r requirements-dev.txt
cd web && npm ci
```

Python 3.12 matches the VPS (`itri`) and CI. Tests need no secrets; commands that touch live services are in `docs/agents/operations.md`.

## Checks

Both run in CI on every PR into `dev`:

```bash
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
cd web && npm run format:check && npm run lint && npm run build && npm test
```

`npm run format` fixes web formatting. Python format and lint are local: `.venv/bin/black` and `.venv/bin/flake8 --max-line-length 88`. Older Python files still fail both; a file you create or edit passes.

## Tests

- **Test through the public entry points**: assert on what `build_snapshot`, the CLIs or the storage module return or write, not on internals. Fakes for Storj and Ambient (`tests/fixtures/`) are the boundary; don't patch Lookout's own functions.
  - *Sanctioned exception:* `tests/test_snapshot_rain_agreement.py::test_the_days_are_counted_once_per_build` patches `ArchiveCoverage.__init__` and `extract_daily_rainfall` to count calls. Spec #174's "counted once per build" leaves no trace in the snapshot output. A new exception needs the same reason, named here.
  - *Sanctioned exception:* `tests/test_archive_store.py::TestMerge::test_a_merge_that_would_drop_a_stored_reading_raises` patches `archive_merge.merge` with a stand-in that drops a reading. The archive merge never drops one, so the archive store's loss check (spec #423, "a lossy merge raises") cannot be reached through its interface otherwise. The catchup tests' loss-check patches (`TestArchiveSafety`) predate this rule.
- Name a test for the behaviour it pins, and cover the error paths as well as the success paths.

## Branches

The flow is: feature or lane branch → `goal/<slug>` → `uat/<slug>` (UAT only) → `dev` → `main`.

- Branch `feat/*`, `fix/*`, `docs/*` or `refactor/*` off `dev`, and open the PR into `dev`. A release built by a swarm uses `lane/<slug>` branches that merge into `goal/<slug>`.
- For UAT, the goal tip is pushed as `uat/<slug>`. Fixes land on the goal branch and are merged forward into it. On Matt's pass, the goal PR merges into `dev`. The steps are in [`.claude/swarm-profile.md`](.claude/swarm-profile.md) ("UAT branch per release").
- A release is a PR from `dev` into `main`. Vercel deploys production from `main`, and the VPS checkout tracks `main`.
- Only `main`, `dev` and `uat/**` deploy on Vercel. Every other branch creates no deployment. Why: [ADR-0010](docs/adr/0010-deploy-allowlist-and-uat-branch-per-release.md).
- Commit and PR titles follow Conventional Commits (`fix(web): ...`, `docs(ops): ...`). The body says why.

## Standards

Code standards, the repo layout and the rules that protect the live archive are in [`CLAUDE.md`](CLAUDE.md). Domain terms are in [`GLOSSARY.md`](GLOSSARY.md); decisions in [`docs/adr/`](docs/adr/).
