# CLAUDE.md

Lookout is a weather station dashboard, live at https://lookout.stonecover.com. A VPS cron on `itri` (the refresh run, every 5 minutes) catches the archive on Storj up from the Ambient Weather API, updates the catalogs, and publishes a dashboard snapshot. A Vite + TypeScript site in `web/`, on Vercel, reads only the snapshot (ADR-0003). Python owns every decision; the site draws what the snapshot says.

## Setup and checks

```bash
uv venv --python 3.12 .venv                     # same Python as itri and CI
uv pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider   # the CI Python gate
cd web && npm ci && npm run format:check && npm run lint && npm run build && npm test  # the CI web gate
cd web && npm run dev                                     # site on golden fixtures, no secrets
```

CI runs both gates on PRs into `dev` and `goal/**` only, never on PRs into `main`. `npm run format` fixes web formatting. Tests go through the public entry points, with Storj and Ambient faked at the boundary (`CONTRIBUTING.md` names the one exception). Python format and lint are local: `.venv/bin/black` and `.venv/bin/flake8 --max-line-length 88` on the `.py` files you change. Older files still fail both, so a file you create or edit passes and the rest stay as they are. Tests need no secrets: `tests/conftest.py` points the secrets loader at a missing file. Web scripts live in `web/package.json`; the dev server's `?fixture=`, `?age=` and `?theme=` switches, the API routes and the CSP check are in `web/README.md`.

## Layout

- `lookout/storage/storj.py` - every Storj read and write; the archive load and save
- `lookout/core/` - archive merge, rain and energy catalogs, rain and solar analysis
- `lookout/snapshot/` - `build_snapshot` (pure), the snapshot dataclass and JSON contract, the publish
- `lookout/cli/` - `refresh.py` (the cron's run), `catchup.py`, `diagnostics.py`, `repair_rain.py`
- `lookout/api/`, `lookout/utils/`, `lookout/config.py`, `lookout/secrets_loader.py`
- `web/src/` the site; `web/api/` thin Vercel functions; `web/server/` their logic, free of Vercel and env

Storj keys, all per station MAC: the archive `<MAC>.parquet` (stored newest first; `load_archive` returns it oldest first), `<MAC>.rain_events_catalog.parquet`, `<MAC>.energy_catalog.parquet`, `<MAC>.rain_corrections.json`, `snapshot/<MAC>.dashboard.json`, `snapshot/<MAC>.context.json`, event curves `snapshot/<MAC>.events/<start_ms>.json`, and backups under `backups/` and `.backups/<MAC>/<date>/`.

## Rules that bite

Each of these cost a fix run to learn. The ADR or test named holds the full reason.

- **Live writes.** Catchup, refresh, `fill-gap`, `repair_rain --write` and the catalog `--save` commands write the live archive. Run them only on `itri` or a checkout holding real secrets, and only while no cron run can save: one archive writer at a time. Worktrees and tests stay on fixtures and moto.
- **Storage reads.** A read ends found, missing or failed (ADR-0004). Read through `read_object`, `read_df_from_s3` or `read_json_from_s3`. A failed read raises `StorageReadError` and aborts the run; only a found-nothing read means missing.
- **Archive saves.** Load with `load_archive` and save with `save_archive(df, bucket, mac, stored=..., repair=False)`, merging through `lookout/core/archive_merge.py`. A save that would drop or blank a stored reading raises `ArchiveLossError`. Catchup, gap fill and rain repair all take this path; a new writer takes it too.
- **Raw archive.** Rain carryover corrections apply on read (`apply_corrections` after loading) and are never written back (#95).
- **Event Catalog.** The refresh run is its only automatic writer, saving after a backup and only when a completed event is new by start time (`save_if_events_completed`). Key events on `start_ms`: an ongoing event gets a new `event_id` every run. The manual rebuild is `diagnostics rain-catalog --save`.
- **Time.** Analysis runs in station-local time (ADR-0002), with the zone read from Ambient `lastData.tz` on every build (ADR-0005). Store and pass epoch-ms `dateutc`; cut days with `lookout.utils.dateutc` in the station zone, and in the snapshot take `now` and the zone from the one `StationClock` (`lookout/snapshot/clock.py`). `dateutc` before 2020 is invalid (ADR-0001).
- **Snapshot contract.** A shape change bumps `SCHEMA_VERSION` in `lookout/snapshot/model.py` and `web/src/snapshot.ts` together, then regenerates the goldens with `LOOKOUT_REGEN_GOLDEN=1 .venv/bin/python -m pytest tests/test_snapshot.py -p no:cacheprovider`. Goldens are regenerated, never hand-merged; vitest reads them, so a Python change shows up as a web test failure. Put decided values in the snapshot so the site never re-decides them. When `parseDashboard` tightens, the refresh run deploys before the site.
- **Web security.** `web/vercel.json` sets the CSP (`script-src 'self'`). A new inline script, origin or worker updates the CSP and `web/tests/security-headers.test.ts` in the same change. Snapshot text reaches HTML through `escapeHtml` (`web/src/dom.ts`), and ESLint `no-unsanitized` fails an unescaped value in `innerHTML`; server errors go through `describeError`; read functions answer an unknown query param with 400 (`checkQuery`).
- **Secrets.** `get_secret(name)` reads an env var, then `~/.config/lookout/secrets.toml`, at first use and never at import. Names are in `secrets.example.toml`. Print only named non-secret fields.

## Code

Python: type hints on every parameter and return; a module docstring stating purpose and the functions it offers; `:param:`-style function docstrings; 88-character lines; fully qualified imports (`lookout.storage.storj`) grouped stdlib, third-party, local; logging through `lookout.utils.log_util.app_logger(__name__)`. `DEV_MODE` (env var, or `[environment] dev = true` in the secrets file) turns on local archive caching in `awn_controller`.

## Where practice lives

- **Branching**: feature (`feat/*`, `fix/*`, `docs/*`) → `dev` → `main`. `dev` is the default branch; Vercel deploys production from `main`. Release flow: `docs/product-lead/brief.md`.
- **Domain terms**: `GLOSSARY.md`, using its terms exactly. Decisions: `docs/adr/`. See `docs/agents/domain.md`.
- **Live services** (Storj, `itri`, Vercel, Healthchecks, Ambient) and every live CLI command: `docs/agents/operations.md` first.
- **Issues**: GitHub `stonematt/lookout` via `gh`; read an issue's brief before working it: a ticket's body, or a triaged issue's Agent Brief comment. `docs/agents/issue-tracker.md`, labels in `docs/agents/triage-labels.md`.
- **Swarm runs**: `.claude/swarm-profile.md` owns the gates, landing policy and per-run repo facts.
- **Initiatives and lessons**: `docs/product-lead/`. Release reviews: `docs/panel/`.
- **UI work**: `DESIGN.md` and `PRODUCT.md`.
- **Energy catalog design**: `ENERGY_CATALOG.md`.
