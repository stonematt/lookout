# Operations: Storj, the VPS, Vercel and Healthchecks

How an agent reaches Lookout's live services, and the gotchas each one hides. Secret values live in `~/.config/lookout/secrets.toml` on both hosts; never print them. When a command mints or prints credentials, send its output to a mode-600 file and show only named non-secret fields.

## Storj

- Bucket `lookout` on the US1 satellite, S3 gateway `https://gateway.storjshare.io`.
- The station inventory `station_inventory.json` and the archive and catalogs at the bucket root (`<MAC>.parquet`, `<MAC>.*_catalog.parquet`, `<MAC>.rain_corrections.json`). Daily archive backups under `.backups/<MAC>/<YYYYMMDD>/`, catalog backups under `backups/`. The dashboard snapshot goes under `snapshot/` (ADR-0003).
- **Encryption passphrase matters.** Objects are visible only to an access built with the app's passphrase. The app's S3 keys in `[lookout_storage_options]` work, and so does the local `uplink` profile `lookout-app` (Read and List, built from Matt's seed). The `uplink` profile `default` uses a different passphrase: it lists the bucket name but no objects. Derive any new credential from `lookout-app`.
- Read-only credential limited to a prefix: `uplink share --access lookout-app --readonly --register sj://lookout/<prefix>/`. The Vercel credential is one of these, for `snapshot/`.
- `uplink access revoke` echoes the grant it revokes.

## VPS (`itri`)

- Agents use the host `s.itri.me`, which starts no tmux: `ssh -T -o RemoteCommand=none s.itri.me '…'`. Read-only checks this way worked on 2026-11-03 (the checkout HEAD, `logs/cron_lookout_cli_refresh.status`, `~/log/cron/refresh.log`).
- The `itri` host alias runs a remote command, so plain `ssh itri <cmd>` fails with "Cannot execute command-line and remote command"; it also needs `-o RemoteCommand=none`.
- Live checkout `~/src/stone/lookout`, tracking `origin/main`; venv `.venv` on a uv-managed Python 3.12.
- The system `python3` is 3.10 with no `tomllib`; run Python through the venv or `uv run --no-project --python 3.12`.
- In-place edits of the remote secrets file get denied. Prepare the file locally (mode 600, validated with `tomllib`) and hand Matt an `scp -p -o RemoteCommand=none` command. Top-level keys go above the first `[table]`.
- Crontab: the refresh run every 5 minutes under `flock`, and the daily 06:00 catchup through `~/bin/run_catchup.sh` (paused for #95). The `tickle_streamlit` keep-alive was removed by #86; its checkout sits at `~/src/stone/tickle_streamlit.retired-20261102`. Logs in `~/log/cron/`. A pause comments a line out as `# paused for lookout#<N> (<date>)`, with the full crontab saved to `~/crontab.bak-pre-<N>`. Read the live state with `ssh -T -o RemoteCommand=none itri 'crontab -l'` before stating it.
- Target refresh crontab line (#244), with no `flock` and no sleep, since the run takes the repo's archive-writer lock itself:

  ```
  */5 * * * * cd /home/mstone/src/stone/lookout && lookout/bin/cron_run.sh lookout/cli/refresh.py >> /home/mstone/log/cron/refresh.log 2>&1
  ```

  It replaces the current `/usr/bin/flock -n /home/mstone/log/cron/refresh.lock lookout/bin/cron_run.sh --sleep 30 lookout/cli/refresh.py` line. Swap order: #179's ops session installs it only after the repo lock is deployed to itri (the checkout's HEAD includes #244), never before. Until the swap both locks run, which is harmless.

## Vercel

- Project `lookout`, team "Matt Stone's projects" (scope `matt-stones-projects-aa308b10`), account `mstone-3228`, Hobby plan. Link at the repo root with `vercel link --project lookout --scope matt-stones-projects-aa308b10`.
- Env vars for Production and Preview: `STORJ_ACCESS_KEY_ID`, `STORJ_SECRET_ACCESS_KEY`, `STORJ_ENDPOINT_URL`, `STORJ_BUCKET`, `STATION_MAC` (the same value as on the VPS; every read function refuses to start without it), `AMBIENT_API_KEY` (Vercel's own key, separate from the VPS key), `AMBIENT_APPLICATION_KEY`, and optionally `AMBIENT_ENDPOINT`. `web/server/env.ts` is the list the functions check. `HC_PING_KEY` stays VPS-only.
- `vercel env add NAME preview` needs an explicit empty branch argument for all Preview branches: `vercel env add NAME preview "" --value … --yes`.
- `STORJ_ENDPOINT_URL` must include the scheme: `https://gateway.storjshare.io`. Without it, every Storj read fails with `TypeError: Invalid URL`.
- Git: connected to `stonematt/lookout` (2026-11-01). Root Directory `web`, Framework `vite`, production branch `main`. Every other branch and every PR gets a preview deployment. Note that GitHub's default branch is `dev`, and Vercel picks the default branch as the production branch on connect. Check the production branch again after any reconnect.
- Root Directory is `web`, so CLI deploys take the repo root as their path. Linking `web/` alone makes Vercel look for `web/web`.
- Production guard: until the first production deploy, the Ignored Build Step is `[ "$VERCEL_ENV" = production ]`. It skips every Git-triggered production build. It exists so that nothing deploys production before the `dev` → `main` release PR merges. To go live: remove it (`vercel api /v9/projects/lookout -X PATCH` with `{"commandForIgnoringBuildStep": null}`), then deploy `main` to production. Read the live setting with `vercel api /v9/projects/lookout` before stating it.
- The CLI prints a banner on stderr, which breaks a `jq` pipe. Run `vercel api … 2>/dev/null | jq`.
- Deployment protection: Vercel Authentication, `all_except_custom_domains`. Previews need a Vercel login. A custom domain on Production is public.
- The CLI sends a project's first deploy to Production. Pass `--target preview` for a preview.

## Healthchecks.io

- Project `lookout`, email alerts to the owner.
- Cron check slug `lookout-refresh` (period 5 min, grace 10 min). Station check slug `station-<slugified Ambient station name>`, auto-created on first ping; set it to period 5 min, grace 30 min afterwards.
- Ping URL shape: `https://hc-ping.com/<HC_PING_KEY>/<slug>`.
- Agents cannot read Healthchecks. To check the refresh run from outside, read its status document, which every run writes at its end, naming the code version and SHA it started on (#268): `curl -s https://lookout.stonecover.com/status | jq '{outcome, sha, version, ended_at_ms, built_at_ms, error}'`. After a release, the check is: `sha` equals the release head on `main` (`git rev-parse origin/main`), `outcome` is `ok`, and `ended_at_ms` is after the merge. A failed run shows as `outcome: "failed"` with `error` naming the exception class; the log on itri has the rest. `sha` is null when git cannot answer in the checkout the cron runs. A run skipped for a held lock writes no status, so the document stays the last run that ran until the holder's run (or the next one) writes its own. The response is CDN-cached for up to a minute. Matt checks the `lookout-refresh` check itself.

## Ambient Weather

- `GET https://rt.ambientweather.net/v1/devices/<MAC>?endDate=…&limit=288` returns up to 288 five-minute records ending at `endDate`, about 1 request per second. Raw history reaches back to at least 2022-05; `lookout/cli/find_earliest_ambient.py` finds the earliest.
- Ambient serves raw rain counters, carryovers included.

## Lookout commands against live services

Each of these reads live secrets and Storj (and most call Ambient). Run them from the repo root with `PYTHONPATH=.`, on the VPS or a checkout with real secrets, never from a test worktree. Only one archive writer runs on a host at a time, and the code enforces it: each writing command below takes the archive-writer lock (`~/.local/state/lookout/archive-writer.lock`, moved by `LOOKOUT_LOCK_FILE`) before any work (#244). A writer started while the lock is held exits 1 naming the lock file; a refresh run that finds it held logs one line and exits 0 with no Healthchecks ping. A refresh run that cannot open the lock file at all (an OSError) is a failed run: it pings exit 1 and writes a failed status. Read-only commands, and writers without their save flag, take no lock.

- **Refresh run**, the VPS cron's job: catchup, catalogs, snapshot publish. `python lookout/cli/refresh.py [--bucket lookout] [--pages N]`. Pings Healthchecks `lookout-refresh` (start, then the exit status) and the station check, then writes its status document `snapshot/<MAC>.status.json` (served at `/status`); a run skipped for a held lock writes none.
- **Diagnostics**, plain text, exit 1 when refused or failed. Read-only: `python -m lookout.cli.diagnostics gaps [--threshold 30] [--days 84 | --since D --until D]`, `coverage [--days 14]`, `status`, `last-record`.
- **Diagnostics writes**, each backs up before it saves:
  - `python -m lookout.cli.diagnostics fill-gap (--index N | --start T --end T) --dry-run` says what it would fetch and writes nothing; drop `--dry-run` to fill and save. Exits 1 when Ambient has no data for the gap.
  - `python -m lookout.cli.diagnostics energy-catalog` reports only; `--save --dry-run` says what a save would do; `--save` backs up and replaces the stored energy catalog. `--dry-run` without `--save` is refused.
  - `python -m lookout.cli.diagnostics rain-catalog` is a dry run: it reports the stored Event Catalog's read outcome (found, missing or failed) and event count, the detected event count, and the events added or dropped. `--save` backs up the stored object as it is (a corrupt catalog is kept for inspection under `backups/`), then replaces it; with none stored it writes with no backup; a failed backup writes nothing and exits 1. This is the way out when a corrupt Event Catalog stops the refresh run.
- **Stations** (#270), the station inventory's one writer, exit 1 when refused or failed.
  - `python -m lookout.cli.stations scan` lists every device on the Ambient account (MAC and Ambient name) and marks which are in the station inventory (`station_inventory.json`) and which is the default station; it also lists any inventory station the account no longer has. Writes nothing and takes no lock.
  - `python -m lookout.cli.stations add <MAC>` refuses a MAC not on the account or already in the inventory. Otherwise it backs the inventory up to `backups/station_inventory_<stamp>.json` (nothing to copy on the first add), then gives the station an archive: a stored `<MAC>.parquet` is adopted unchanged; with none stored, it pages Ambient's history back from now until there is nothing older (a second a page, 288 readings each) and saves the first archive. Last it appends the MAC to the inventory; the first add seeds it, and that station is the default station. A failed backup writes nothing; a failed backfill leaves the inventory as it was. If the inventory write fails after the archive is saved, run the same `add` again: it adopts that archive.
- **Rain repair** (#95), restores the raw rain counters in the damaged windows from Ambient and cross-checks them against the raw backup. Read `python -m lookout.cli.repair_rain --help` first.
  - Dry run, the default, writes nothing: `python -m lookout.cli.repair_rain`. Every day should read `rows N -> N` (or more), no `short` line, and `0 over tolerance`; it exits 1 if not.
  - Real run: `python -m lookout.cli.repair_rain --write`. Same checks; it writes nothing if any fail, else backs the live archive up to `.backups/<MAC>/<today>/` and saves the repair.
  - Other windows, as station-local days (repeatable): `--window 2025-10-14 2025-12-31`. The 2025 window is about 90 Ambient pages, a second each.
