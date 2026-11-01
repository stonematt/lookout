# Swarm profile

**Repo** `stonematt/lookout` · root `/Users/mstone/src/github.com/stonematt/lookout` · base branch `dev` (release branch `main`, deploy branch `live`). Spec #66 work: phase 0 (#67, #68, #69, #70, #71, #63) lands on `dev`; phases 1–4 land on the integration branch `goal/off-streamlit`, cut from `dev` after phase 0.

**Ready label** none used for this run; every sub-issue of #66 took the finish-alone test.

**Read before designing** `CLAUDE.md`, `AGENTS.md`, `CONTEXT.md`, `GLOSSARY.md`, `docs/adr/` (ADR-0003 is the architecture), `code_standards.md`, `TESTING_GUIDELINES.md`, `DESIGN.md` (UI tickets), spec #66.

**Dependency sync** Python: `source $(conda info --base)/etc/profile.d/conda.sh && conda activate lookout-new` (one shared conda env; it has no per-worktree copy). A lane that adds a Python dependency adds it to `requirements-dev.txt`/`requirements.txt` and installs it into `lookout-new` with `pip install -r requirements-dev.txt`. Web (after #77): `cd web && npm ci`.

**Lanes at once** 3.

**Gates**, in this order, all green before anything opens:

| gate | command | baseline at `7592c37` | time |
| --- | --- | --- | --- |
| tests | `python -m pytest tests/ -q -p no:cacheprovider` (in `lookout-new`, from the worktree root) | **789 passed, 1 skipped at `974c644`** (after the 2026-10-31 QA/UAT fixes #147 #148 #149 #156; ~80s). 738 passed, 1 skipped at `ce6c582` (after the 2026-10-31 follow-ups; #114 deleted the one strict xfail). 701 passed, 1 skipped, 1 xfailed at `dfff58b` (goal branch end of the spec #66 run; clean worktree, no secrets; ~85s). At the goal cut `bef4bb7`: 356/1/1. Before #67: clean worktree: **2 collection errors** (`StreamlitSecretNotFound`, from import-time `st.secrets` reads — the defect #67 fixes). Main checkout with real secrets: 141 passed, 1 skipped. After #67 lands, the clean-worktree number is the baseline. | 28s |
| format | `black --check <changed .py files>` | 14 files already fail on `dev`; gate is: every file you create or edit passes | 2s |
| lint | `flake8 --max-line-length 88 <changed .py files>` | 375 lines on `dev`; gate is: files you create are clean, files you edit gain no new lines | 2s |
| web | `cd web && npm ci && npm run build && npm test` (build runs `tsc --noEmit` then vite) | build ok, **347 passed** at `974c644`; 301 at `ce6c582` (287 at `dfff58b`) | 20s |

**CI** `.github/workflows/tests.yml` (from #67): pytest on PRs into `dev` and `goal/**`, Python 3.12, 15-minute job timeout (a slow fixture once hit it). `.github/workflows/web.yml` (from #77): web build + test. Before #67 there was none. #67 added one workflow running pytest on PRs into `dev` and into `goal/off-streamlit`. Until #67 merges, the gates above are the only check.

**Blast radius** — probes planted at `$TMPDIR`, `~/.config/`, `~`, and the main checkout's live `data/` dir; test suite run at `7592c37` in a clean worktree: all markers survived. The real risk is secrets, not deletes: the main checkout has `.streamlit/secrets.toml` with live Ambient and Storj keys, and modules read it at import. **Never run gates in the main checkout. Never copy `.streamlit/secrets.toml` into a worktree.** A lane's new tests pin to: no network, no real secrets (conftest stubs from #67), moto for S3, synthetic or committed fixtures, `tmp_path` for files.

**Leave unrun** `streamlit run streamlit_app.py` — the owner's dashboard. `lookout/cli/catchup.py`, `lookout/bin/cron_run.sh`, the new refresh command against real secrets — they write the live Storj archive and Event Catalog. `lookout/cli/pull_archive.py`, `rain_corrections_cli.py`, `manage_day_periods.py`, `regenerate*`/catalog CLIs, `verify_energy_catalog.py`, `test_functional_energy.py` — they read or write live Storj. Anything that ssh's to the VPS (`itri`). Any Vercel deploy.

**Lane server** `cd web && npm ci && npm run dev -- --port <port>` (from #77): runs the real `/dashboard` and `/current` handlers over the golden snapshots via a Vite plugin, no secrets. Page query `?fixture=dry|raining&age=<min>&theme=light|dark`. Screenshots: `npm run shots -- http://localhost:<port> <out-dir> "" "fixture=raining&age=3"` after `npx playwright install chromium` once. Port = 5170 + lane letter index (G 5177, H 5178, I 5179, J 5180, K 5181, L 5182, M 5183). Stop by killing the vite child PID too (the npm wrapper PID alone leaves vite running). Before #77: a command that serves one worktree's `web/` on a chosen port against the golden snapshot fixtures with no secrets (e.g. `cd web && npm run dev -- --port <port>`), and record it in its PR body; later UI lanes use it with a port of 5170 + lane letter index. Stop it with Ctrl-C / `kill` of its PID.

**Visual critique** `impeccable` (critique) at 1280px and 390px widths. Browser: drive your own headless browser (`npx playwright screenshot` or a Playwright script), not the session's shared Playwright MCP.

**Verbs** — implement `tdd` · review `code-review` · commit `stone-commit` · merge `stone-merge`

**Landing policy** Lanes never run `gh pr merge`; the orchestrator does. Phase 0 PRs target `dev`; phase 1+ PRs target `goal/off-streamlit`. Only the owner lands `goal/off-streamlit` into `dev`. Never push to `main`, `live` or `dev` directly. No Claude/AI attribution in commits or PR bodies (no `Co-Authored-By: Claude`, no "Generated with" footer). Conventional Commits.

**Test layout** Shared fixtures in `tests/fixtures/` (#68: `build_archive`, `scenario_archive`, `load_real_slice`, `AmbientReplay`; conftest `fake_s3`, `ambient_replay`, `frozen_today`). Golden/generated files go under `tests/golden/`; the fixture leak scans (`tests/test_fixture_real.py`, `tests/test_load_path_golden.py`) name the files they expect, so a new fixture under `tests/fixtures/` must be added to that list. Shared snapshot test builders: `tests/snapshot_helpers.py`. Registered marks in `pytest.ini`. Catchup end-to-end setup: `catchup_station` fixture in `tests/test_catchup.py` (#71). Tests that use `FAKE_MAC` must `monkeypatch.setenv('STATION_MAC', FAKE_MAC)`. Load-path golden tests (`tests/test_load_path_golden.py`, `tests/golden/load_path_*.json`) are marked `temporary`; regenerate with `LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_load_path_golden.py -p no:cacheprovider`.

**Shared generated files (goal branch)** Golden snapshot JSON fixtures (under `tests/golden/`) are regenerated, never hand-merged: on a rebase conflict in one, take either side, rerun the regenerate command #72 defines, then rerun the gates. `web/package-lock.json` is fenced to #77; a later lane that needs a new npm dependency escalates.

**Repo facts learned in the spec #66 run (2026-10-30)**
- `gh pr merge --delete-branch` from the main checkout also removes the lane's sibling worktree and local branch; a lane's step g usually finds nothing left.
- Rebases that touch both `lookout/snapshot/model.py` and `web/src/snapshot.ts` are the common conflict: adjacent additions, keep both, regenerate goldens. diff3 has twice dropped a decorator or a closing brace — re-run `tsc` and the Python suite after resolving.
- `web/scripts/shots.mjs` waits for lazily loaded Plotly; screenshots taken another way show blank Rain Events charts.
- The snapshot contract is `SCHEMA_VERSION` 2 on both sides; a web test reads each golden's `schema_version`, so bump both together.
- The VPS catchup cron on itri was paused 2026-10-30 for #95 (`# paused for lookout#95`); #76 turns on the 5-minute refresh cron.

**Repo facts learned in the follow-up run (2026-10-31)**
- Lane-server ports can be held by a server left over from an earlier run (5190 was still serving `../lookout-baseline` at `dfff58b`). With `--strictPort`, the new server exits and the old one answers 200. Check the listening process's cwd (`lsof -p <pid> -a -d cwd`) before trusting a page.
- Triage posts an **Agent Brief** comment on each issue (`gh issue view <n> --comments`). It can add acceptance criteria beyond the issue body, and it records owner decisions. Read it before briefing a lane.
- `gh pr checks --watch` inside a lane can be moved to the background after about 180s. Poll `gh pr checks` in the foreground until each check reports.
- The storage existence check is one helper, `storj.object_exists(bucket, key)`. The catalog classes' `catalog_exists` methods call it. A new catalog uses it too, rather than an inline `head_object`.

**Repo facts learned in the QA/UAT fix run (2026-10-31, #147 #148 #149 #156)**
- `web/vercel.json` now sets a CSP (#148). `style-src` needs `'unsafe-inline'` because Plotly injects styles, and `script-src` is `'self'` only. A lane that adds an inline script, an external origin or a worker must update the CSP and its test (`web/tests/security-headers.test.ts`). Vite dev does not apply `vercel.json` headers; `web/scripts/csp-check.mjs` applies them by Playwright route interception on `vite preview`.
- HTML built from snapshot values goes through `escapeHtml` in `web/src/dom.ts` (imported as `esc` in `web/src/events/tab.ts`). Server logging goes through `describeError` in `web/server/errors.ts`, never the raw error.
- The 390px no-sideways-scroll check (`scrollWidth <= clientWidth` on every tab, both fixtures, light and dark) was a throwaway script in the run's scratch dir, not in the repo. A UI lane needs it again; either commit it under `web/scripts/` or rewrite it from that description.
- Contract fields the browser needs go in the snapshot as decided values. The whole-run review found the browser re-deciding a nullable field (`EventSeason.status`) and a time cut (`first_hour_to_come`) that Python already knew. A consumer lane that finds a null it must interpret escalates to the producer side rather than inventing a rule.
- `parseDashboard` now refuses a dashboard without `first_hour_to_come` or with a season missing `status` (#156). Deploy the refresh before the site, or the site errors until the next refresh writes the new fields.
- The harness refuses a foreground `sleep N; gh pr checks`. Poll with an until-loop instead.

**Repo facts learned in the release-blocker run (2026-11-01, #160 #161)**
- Baseline at `6aeba28`: pytest **829 passed, 1 skipped** (~92s); web build ok, **381 passed**.
- Archive writers go through `storj.load_archive` (strict, normalized, refuses empty) and `storj.save_archive(df, bucket, mac, loaded_rows=)`; `loaded_rows` is required, and the count is taken after normalize. `backup_and_save_history` requires it too. `get_df_from_s3`, `get_json_from_s3` and `rain_corrections.load_catalog` stay forgiving for read-only users (Streamlit). A new writer uses the strict pair; #167 tracks the callers still on the forgiving path.
- Catchup and refresh raise `StorageReadError`, `StorageWriteError`, `ArchiveShrinkError`, `BackupError` or `CatchupStalled`; `refresh.main` returns 1 and pings Healthchecks `/1`. A missing archive fails catchup on purpose.
- The deployed read functions refuse unknown or repeated query params (400) and 308 any other spelling to one URL, via `checkQuery` in `web/server/handlers.ts`. `fixture` and `age` work only on the lane server. The client sends them only when `LANE_SERVER` is true (`import.meta.env.DEV` or a build with `VITE_LANE_SERVER=1`), so `vite preview` runs for `csp-check.mjs` and `shots` need `VITE_LANE_SERVER=1 npm run build` first.
- `web/src/page.ts` owns the page boot and 5-minute refresh cycle (`startPage` returns `booted` and `stop`); tests must call `stop()`.
- A scratch-dir script that does `require('playwright')` resolves `~/.claude/node_modules/playwright` (old, no browser). Require `web/node_modules/playwright` by absolute path.

**Deepen run (2026-11-01, specs #172 #173 #174)**
- Goal branch `goal/deepen-reads-merge-rain`, cut off `goal/off-streamlit` at `a43d894` with `docs/station-tz` and `docs/glossary-archive` merged in (ADR-0004, ADR-0005, the archive and archive merge glossary entries). Its goal PR targets `dev`, is stacked on #96, and lands after it. Lane PRs target `goal/deepen-reads-merge-rain`; CI runs on them (`goal/**`).
- Baseline at `d236bf9`: pytest **829 passed, 1 skipped** (~85s); web build ok, **381 passed**. Probes at `$TMPDIR`, `~`, `~/.config/` and `~/.config/lookout/` all survived; `tests/conftest.py` points the secrets loader at a missing file, so the real `~/.config/lookout/secrets.toml` is never read.
- Owner fence for this run: `lookout/storage/storj.py`, the archive save path and `lookout/cli/refresh.py` change in build order only, never in two lanes at once.
