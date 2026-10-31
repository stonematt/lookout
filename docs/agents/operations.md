# Operations: Storj, the VPS, Vercel and Healthchecks

How an agent reaches Lookout's live services, and the gotchas each one hides. Secret values live in `.streamlit/secrets.toml` (local) and `~/.config/lookout/secrets.toml` (VPS); never print them. When a command mints or prints credentials, send its output to a mode-600 file and show only named non-secret fields.

## Storj

- Bucket `lookout` on the US1 satellite, S3 gateway `https://gateway.storjshare.io`.
- Archive and catalogs at the bucket root (`<MAC>.parquet`, `<MAC>.*_catalog.parquet`, `<MAC>.rain_corrections.json`). Daily archive backups under `.backups/<MAC>/<YYYYMMDD>/`, catalog backups under `backups/`. The dashboard snapshot goes under `snapshot/` (ADR-0003).
- **Encryption passphrase matters.** Objects are visible only to an access built with the app's passphrase. The app's S3 keys in `[lookout_storage_options]` work, and so does the local `uplink` profile `lookout-app` (Read and List, built from Matt's seed). The `uplink` profile `default` uses a different passphrase: it lists the bucket name but no objects. Derive any new credential from `lookout-app`.
- Read-only credential limited to a prefix: `uplink share --access lookout-app --readonly --register sj://lookout/<prefix>/`. The Vercel credential is one of these, for `snapshot/`.
- `uplink access revoke` echoes the grant it revokes.

## VPS (`itri`)

- Non-interactive commands need `ssh -o RemoteCommand=none itri '…'` (add `-T` when piping). The host config runs a remote command, so plain `ssh itri <cmd>` fails with "Cannot execute command-line and remote command".
- Live checkout `~/src/stone/lookout` (branch `live` until #76 moves it), venv `.venv` on a uv-managed Python 3.12. `.streamlit/secrets.toml` there symlinks to `~/.config/lookout/secrets.toml`.
- The system `python3` is 3.10 with no `tomllib`; run Python through the venv or `uv run --no-project --python 3.12`.
- In-place edits of the remote secrets file get denied. Prepare the file locally (mode 600, validated with `tomllib`) and hand Matt an `scp -p -o RemoteCommand=none` command. Top-level keys go above the first `[table]`.
- Crontab: the daily 06:00 catchup through `~/bin/run_catchup.sh`, and a `tickle_streamlit` uptime job every 4 hours (removed by #86). Logs in `~/log/cron/`.

## Vercel

- Project `lookout`, team "Matt Stone's projects" (scope `matt-stones-projects-aa308b10`), account `mstone-3228`, Hobby plan. Link with `vercel link --project lookout --scope matt-stones-projects-aa308b10`.
- Env vars for Production and Preview: `STORJ_ACCESS_KEY_ID`, `STORJ_SECRET_ACCESS_KEY`, `STORJ_ENDPOINT_URL`, `STORJ_BUCKET`, `AMBIENT_API_KEY` (Vercel's own key, separate from the VPS key), `AMBIENT_APPLICATION_KEY`. `HC_PING_KEY` stays VPS-only.
- `vercel env add NAME preview` needs an explicit empty branch argument for all Preview branches: `vercel env add NAME preview "" --value … --yes`.

## Healthchecks.io

- Project `lookout`, email alerts to the owner.
- Cron check slug `lookout-refresh` (period 5 min, grace 10 min). Station check slug `station-<slugified Ambient station name>`, auto-created on first ping; set it to period 5 min, grace 30 min afterwards.
- Ping URL shape: `https://hc-ping.com/<HC_PING_KEY>/<slug>`.

## Ambient Weather

- `GET https://rt.ambientweather.net/v1/devices/<MAC>?endDate=…&limit=288` returns up to 288 five-minute records ending at `endDate`, about 1 request per second. Raw history reaches back to at least 2022-05; `lookout/cli/find_earliest_ambient.py` finds the earliest.
- Ambient serves raw rain counters, carryovers included.

## Lookout commands against live services

Each of these reads live secrets and Storj (and most call Ambient). Run them from the repo root with `PYTHONPATH=.`, on the VPS or a checkout with real secrets, never from a test worktree. Only one writer at a time: never run a writing command while a catchup or refresh cron run could save the archive.

- **Refresh run**, the VPS cron's job: catchup, catalogs, snapshot publish. `python lookout/cli/refresh.py [--bucket lookout] [--pages N]`. Pings Healthchecks `lookout-refresh` (start, then the exit status) and the station check.
- **Diagnostics**, plain text, exit 1 when refused or failed. Read-only: `python -m lookout.cli.diagnostics gaps [--threshold 30] [--days 84 | --since D --until D]`, `coverage [--days 14]`, `status`, `last-record`.
- **Diagnostics writes**, each backs up before it saves:
  - `python -m lookout.cli.diagnostics fill-gap (--index N | --start T --end T) --dry-run` says what it would fetch and writes nothing; drop `--dry-run` to fill and save. Exits 1 when Ambient has no data for the gap.
  - `python -m lookout.cli.diagnostics energy-catalog` reports only; `--save --dry-run` says what a save would do; `--save` backs up and replaces the stored energy catalog. `--dry-run` without `--save` is refused.
- **Rain repair** (#95), restores the raw rain counters in the damaged windows from Ambient and cross-checks them against the raw backup. Read `python -m lookout.cli.repair_rain --help` first.
  - Dry run, the default, writes nothing: `python -m lookout.cli.repair_rain`. Every day should read `rows N -> N` (or more), no `short` line, and `0 over tolerance`; it exits 1 if not.
  - Real run: `python -m lookout.cli.repair_rain --write`. Same checks; it writes nothing if any fail, else backs the live archive up to `.backups/<MAC>/<today>/` and saves the repair.
  - Other windows, as station-local days (repeatable): `--window 2025-10-14 2025-12-31`. The 2025 window is about 90 Ambient pages, a second each.
