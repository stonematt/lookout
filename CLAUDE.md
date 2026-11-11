# CLAUDE.md

Lookout is a weather station dashboard, live at https://lookout.stonecover.com. A VPS cron on `itri` (the refresh run, every 5 minutes) catches each archive on Storj up from the Ambient Weather API, updates the catalogs, and publishes a dashboard snapshot, for every station in the station inventory (ADR-0006). A Vite + TypeScript site in `web/`, on Vercel, reads only the snapshot (ADR-0003). Python owns every decision; the site draws what the snapshot says.

## Setup and checks

```bash
uv venv --python 3.12 .venv                     # same Python as itri and CI
uv pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider   # the CI Python gate
cd web && npm ci && npm run format:check && npm run lint && npm run build && npm test  # the CI web gate
cd web && npm run dev                                     # site on golden fixtures, no secrets
```

CI runs both gates on PRs into `dev`, `goal/**` and `main`. `npm run format` fixes web formatting. Tests go through the public entry points, with Storj and Ambient faked at the boundary (`CONTRIBUTING.md` names the one exception). Python format and lint are local: `.venv/bin/black` and `.venv/bin/flake8 --max-line-length 88` on the `.py` files you change. Older files still fail both, so a file you create or edit passes and the rest stay as they are. Tests need no secrets: `tests/conftest.py` points the secrets loader at a missing file. Web scripts live in `web/package.json`; the dev server's `?fixture=`, `?age=` and `?theme=` switches, the API routes and the CSP check are in `web/README.md`.

## Layout

- `lookout/storage/storj.py` - every Storj read and write; the archive load and save
- `lookout/storage/archive_readback.py` - the raw read-back and checks of a saved archive (the rain repair)
- `lookout/core/` - archive merge, rain and energy catalogs, rain and solar analysis
- `lookout/snapshot/` - `build_snapshot` (pure), the snapshot dataclass and JSON contract, the publish
- `lookout/cli/` - `refresh.py` (the cron's run), `catchup.py`, `diagnostics.py` and `repair_rain.py` (both take `--station <MAC>`, the default station without it; a MAC not in the inventory exits 1), `stations.py` (`stations scan`, `stations add <MAC>`)
- `lookout/api/`, `lookout/utils/`, `lookout/config.py`, `lookout/secrets_loader.py`
- `web/src/` the site; `web/api/` thin Vercel functions; `web/server/` their logic, free of Vercel and env

Storj keys: the station inventory `station_inventory.json` (one per bucket, backed up to `backups/station_inventory_<stamp>.json`; read it with `lookout.storage.station_inventory`), the station index `snapshot/stations.json` (one per bucket, published by the refresh run after every station; read it with `lookout.snapshot.station_index`), and per station MAC the archive `<MAC>.parquet` (stored newest first; `load_archive` returns it oldest first), `<MAC>.event_catalog.parquet`, `<MAC>.energy_catalog.parquet`, `<MAC>.rain_corrections.json`, `snapshot/<MAC>.dashboard.json`, `snapshot/<MAC>.context.json`, `snapshot/<MAC>.status.json`, event curves `snapshot/<MAC>.events/<start_ms>.json`, and backups under `backups/` and `.backups/<MAC>/<date>/`.

## Rules that bite

Each of these cost a fix run to learn. The ADR or test named holds the full reason.

- **Live writes.** Catchup, refresh, `fill-gap`, `repair_rain --write`, `stations add` and the catalog `--save` commands write the live archive. Run them only on `itri` or a checkout holding real secrets. One archive writer at a time is enforced by the archive-writer lock (`lookout/utils/archive_lock.py`, #244): each writer takes it at its entry point, a second writer exits 1 naming the lock file, and a refresh run that finds it held skips. A new writer takes it too, at its entry point, and saves through the archive store (see Archive saves). Worktrees and tests stay on fixtures and moto; `tests/conftest.py` points `LOOKOUT_LOCK_FILE` under `tmp_path`.
- **Storage reads.** A read ends found, missing or failed (ADR-0004). Read through `read_object`, `read_df_from_s3` or `read_json_from_s3`. A failed read raises `StorageReadError` and aborts the run; only a found-nothing read means missing.
- **Archive saves.** The archive store (`lookout/storage/archive_store.py`) is the only route to save the archive: `open(bucket, mac)` then `merge(fetched)` or `replace(df, reason, ...)`, or `create` for a first build. It refuses to save unless the process holds the archive-writer lock, takes the backup before it saves, and merges through `lookout/core/archive_merge.py`. A save that would drop or blank a stored reading raises `ArchiveLossError`; only `replace` lifts that check, with a repair backup and a read-back. A new writer takes the lock at its entry point and goes through the store; it never calls `save_archive` or the backup functions in `lookout/storage/storj.py`. `tests/test_archive_store.py` holds the rules.
- **Raw archive.** Rain carryover corrections apply on read (`apply_corrections` after loading) and are never written back (#95).
- **Event Catalog.** The refresh run is its only automatic writer, saving after a backup and only when a completed event is new by start time (`save_if_events_completed`). Key events on `start_ms`: an ongoing event gets a new `event_id` every run. The manual rebuild is `diagnostics event-catalog --save`.
- **Time.** Analysis runs in station-local time (ADR-0002), with the zone read from Ambient `lastData.tz` on every build (ADR-0005). Store and pass epoch-ms `dateutc`; cut days with `lookout.utils.dateutc` in the station zone, and in the snapshot take `now` and the zone from the one `StationClock` (`lookout/snapshot/clock.py`). `dateutc` before 2020 is invalid (ADR-0001).
- **Snapshot contract.** A shape change bumps `SCHEMA_VERSION` in `lookout/snapshot/model.py` and `web/src/snapshot.ts`, then regenerates the goldens with `LOOKOUT_REGEN_GOLDEN=1 .venv/bin/python -m pytest tests/test_snapshot.py -p no:cacheprovider`. Goldens are regenerated, never hand-merged; vitest reads them, so a Python change shows up as a web test failure. Put decided values in the snapshot so the site never re-decides them. The site reads versions N and N−1 (`SNAPSHOT_SCHEMA_VERSIONS`), so a bump ships in two releases: first the site reads N, then the refresh run writes N (ADR-0008).
- **Web security.** `web/vercel.json` sets the CSP (`script-src 'self'`). A new inline script, origin or worker updates the CSP and `web/tests/security-headers.test.ts` in the same change. Snapshot text reaches HTML through `escapeHtml` (`web/src/dom.ts`), and ESLint `no-unsanitized` fails an unescaped value in `innerHTML`; server errors go through `describeError`; read functions answer an unknown query param with 400 (`checkQuery`).
- **Secrets.** `get_secret(name)` reads an env var, then `~/.config/lookout/secrets.toml`, at first use and never at import. Names are in `secrets.example.toml`. Print only named non-secret fields.
- **Stations.** The station inventory is the only list of stations; there is no `STATION_MAC` setting on the Python side (#271). A job that works on one station resolves it with `station_inventory.resolve_station(bucket, mac=None)`, the one helper: it takes the default station (the inventory's first) when no MAC is given and refuses a MAC the inventory lacks (`StationNotInInventory`), before any Ambient call; the job then fetches that device with `ambient_client.get_station_device(mac)`. The refresh run handles every inventory station; a log line inside one station's work carries its name and MAC (`log_util.station_context`). Tests seed the inventory with `tests.fixtures.storage.seed_inventory` (`catchup_station` does it for `FAKE_MAC`).

## Code

Python: type hints on every parameter and return; a module docstring stating purpose and the functions it offers; `:param:`-style function docstrings; 88-character lines; fully qualified imports (`lookout.storage.storj`) grouped stdlib, third-party, local; logging through `lookout.utils.log_util.app_logger(__name__)`. `DEV_MODE` (env var, or `[environment] dev = true` in the secrets file) turns on local archive caching in `awn_controller`.

## Where practice lives

- **Branching**: feature (`feat/*`, `fix/*`, `docs/*`) → `dev` → `main`. `dev` is the default branch; Vercel deploys production from `main`. Release flow: `docs/product-lead/brief.md`.
- **Domain terms**: `GLOSSARY.md`, using its terms exactly. Decisions: `docs/adr/`. See `docs/agents/domain.md`.
- **Live services** (Storj, `itri`, Vercel, Healthchecks, Ambient) and every live CLI command: `docs/agents/operations.md` first.
- **Issues**: GitHub `stonematt/lookout` via `gh`; read an issue's brief before working it: a ticket's body, or a triaged issue's Agent Brief comment. `docs/agents/issue-tracker.md`, labels in `docs/agents/triage-labels.md`.
- **Swarm runs**: `.claude/swarm-profile.md` owns the gates, landing policy and per-run repo facts.
- **Initiatives and lessons**: `docs/product-lead/`. Release reviews: `docs/panel/`.
- **UI work**: `DESIGN.md` and `PRODUCT.md`. The Lookout design system is checked in at `design-system/`: `README.md` (the brand book), `tokens.json`, and `components/<Name>/` guidelines and previews. It is built from `DESIGN.md`, which wins where they differ. Use it for decks, Design canvases and other artifacts made in this repo instead of the account default. Its claude.ai copy (https://claude.ai/artifact/Bmri2bVCvPqPumexcZkMXc) is a mirror: change `design-system/` first, then republish. The previews read the site's CSS variables from `web/src/styles/tokens.css`, plus `--font-sans` (the site's `--sans`).
- **Energy catalog design**: `ENERGY_CATALOG.md`.
