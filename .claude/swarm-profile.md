# Swarm profile

**Repo** `stonematt/lookout` · root `/Users/mstone/src/github.com/stonematt/lookout` · base branch `dev` (release branch `main`, deploy branch `live`). Spec #66 work: phase 0 (#67, #68, #69, #70, #71, #63) lands on `dev`; phases 1–4 land on the integration branch `goal/off-streamlit`, cut from `dev` after phase 0.

**Ready label** none used for this run; every sub-issue of #66 took the finish-alone test.

**Read before designing** `CLAUDE.md`, `AGENTS.md`, `CONTEXT.md`, `GLOSSARY.md`, `docs/adr/` (ADR-0003 is the architecture), `code_standards.md`, `TESTING_GUIDELINES.md`, `DESIGN.md` (UI tickets), spec #66.

**Dependency sync** Python: `source $(conda info --base)/etc/profile.d/conda.sh && conda activate lookout-new` (one shared conda env; it has no per-worktree copy). A lane that adds a Python dependency adds it to `requirements-dev.txt`/`requirements.txt` and installs it into `lookout-new` with `pip install -r requirements-dev.txt`. Web (after #77): `cd web && npm ci`.

**Lanes at once** 3.

**Gates**, in this order, all green before anything opens:

| gate | command | baseline at `7592c37` | time |
| --- | --- | --- | --- |
| tests | `python -m pytest tests/ -q -p no:cacheprovider` (in `lookout-new`, from the worktree root) | **671 passed, 1 skipped, 1 xfailed at `2e09d09`** (goal branch after every ticket; clean worktree, no secrets; ~80s). At the goal cut `bef4bb7`: 356/1/1. Before #67: clean worktree: **2 collection errors** (`StreamlitSecretNotFound`, from import-time `st.secrets` reads — the defect #67 fixes). Main checkout with real secrets: 141 passed, 1 skipped. After #67 lands, the clean-worktree number is the baseline. | 28s |
| format | `black --check <changed .py files>` | 14 files already fail on `dev`; gate is: every file you create or edit passes | 2s |
| lint | `flake8 --max-line-length 88 <changed .py files>` | 375 lines on `dev`; gate is: files you create are clean, files you edit gain no new lines | 2s |
| web | `cd web && npm ci && npm run build && npm test` (build runs `tsc --noEmit` then vite) | build ok, **238 passed** at `2e09d09` (56 at `3d1e93b`) | 20s |

**CI** `.github/workflows/tests.yml` (from #67): pytest on PRs into `dev` and `goal/**`, Python 3.12, 15-minute job timeout (a slow fixture once hit it). `.github/workflows/web.yml` (from #77): web build + test. Before #67 there was none. #67 added one workflow running pytest on PRs into `dev` and into `goal/off-streamlit`. Until #67 merges, the gates above are the only check.

**Blast radius** — probes planted at `$TMPDIR`, `~/.config/`, `~`, and the main checkout's live `data/` dir; test suite run at `7592c37` in a clean worktree: all markers survived. The real risk is secrets, not deletes: the main checkout has `.streamlit/secrets.toml` with live Ambient and Storj keys, and modules read it at import. **Never run gates in the main checkout. Never copy `.streamlit/secrets.toml` into a worktree.** A lane's new tests pin to: no network, no real secrets (conftest stubs from #67), moto for S3, synthetic or committed fixtures, `tmp_path` for files.

**Leave unrun** `streamlit run streamlit_app.py` — the owner's dashboard. `lookout/cli/catchup.py`, `lookout/bin/cron_run.sh`, the new refresh command against real secrets — they write the live Storj archive and Event Catalog. `lookout/cli/pull_archive.py`, `rain_corrections_cli.py`, `manage_day_periods.py`, `regenerate*`/catalog CLIs, `verify_energy_catalog.py`, `test_functional_energy.py` — they read or write live Storj. Anything that ssh's to the VPS (`itri`). Any Vercel deploy.

**Lane server** `cd web && npm ci && npm run dev -- --port <port>` (from #77): runs the real `/dashboard` and `/current` handlers over the golden snapshots via a Vite plugin, no secrets. Page query `?fixture=dry|raining&age=<min>&theme=light|dark`. Screenshots: `npm run shots -- http://localhost:<port> <out-dir> "" "fixture=raining&age=3"` after `npx playwright install chromium` once. Port = 5170 + lane letter index (G 5177, H 5178, I 5179, J 5180, K 5181, L 5182, M 5183). Stop by killing the vite child PID too (the npm wrapper PID alone leaves vite running). Before #77: a command that serves one worktree's `web/` on a chosen port against the golden snapshot fixtures with no secrets (e.g. `cd web && npm run dev -- --port <port>`), and record it in its PR body; later UI lanes use it with a port of 5170 + lane letter index. Stop it with Ctrl-C / `kill` of its PID.

**Visual critique** `impeccable` (critique) at 1280px and 390px widths. Browser: drive your own headless browser (`npx playwright screenshot` or a Playwright script), not the session's shared Playwright MCP.

**Verbs** — implement `tdd` · review `code-review` · commit `stone-commit` · merge `stone-merge`

**Landing policy** Lanes never run `gh pr merge`; the orchestrator does. Phase 0 PRs target `dev`; phase 1+ PRs target `goal/off-streamlit`. Only the owner lands `goal/off-streamlit` into `dev`. Never push to `main`, `live` or `dev` directly. No Claude/AI attribution in commits or PR bodies (no `Co-Authored-By: Claude`, no "Generated with" footer). Conventional Commits.

**Test layout** Shared fixtures in `tests/fixtures/` (#68: `build_archive`, `scenario_archive`, `load_real_slice`, `AmbientReplay`; conftest `fake_s3`, `ambient_replay`, `frozen_today`). `tests/test_fixture_real.py:87` asserts exactly 3 data files under `tests/fixtures/`, so golden/generated files go under `tests/golden/`, never `tests/fixtures/`. Catchup end-to-end setup: `catchup_station` fixture in `tests/test_catchup.py` (#71). Tests that use `FAKE_MAC` must `monkeypatch.setenv('STATION_MAC', FAKE_MAC)`. Load-path golden tests (`tests/test_load_path_golden.py`, `tests/golden/load_path_*.json`) are marked `temporary`; regenerate with `LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_load_path_golden.py -p no:cacheprovider`.

**Shared generated files (goal branch)** Golden snapshot JSON fixtures (under `tests/golden/`) are regenerated, never hand-merged: on a rebase conflict in one, take either side, rerun the regenerate command #72 defines, then rerun the gates. `web/package-lock.json` is fenced to #77; a later lane that needs a new npm dependency escalates.
