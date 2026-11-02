# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lookout is a weather station dashboard, live at https://lookout.stonecover.com. A VPS cron (the refresh run) pulls the Ambient Weather API into an archive on S3-compatible storage (Storj) and publishes a dashboard snapshot; a Vite + TypeScript site in `web/`, on Vercel, reads only the snapshot (ADR-0003). CLI scripts handle maintenance.

## Common Development Commands

### Running the Site
```bash
cd web && npm ci && npm run dev     # local dev server
cd web && npm run build && npm test # the web checks CI runs
```

### Running CLI Scripts
```bash
# Run catchup script to sync new data from Ambient Weather API
PYTHONPATH=. python lookout/cli/catchup.py

# Run with cron (includes logging and jitter)
lookout/bin/cron_run.sh --sleep 300 lookout/cli/catchup.py

# One refresh run (lookout/cli/refresh.py): catchup, catalogs, then publish
# the dashboard snapshot to Storj (snapshot/<MAC>.dashboard.json, .context.json)
PYTHONPATH=. python lookout/cli/refresh.py [--bucket lookout] [--pages N]

# Diagnostics (lookout/cli/diagnostics.py; plain text).
# Read-only unless noted; exit 1 when refused or failed.
PYTHONPATH=. python -m lookout.cli.diagnostics gaps [--threshold 30] [--days 84 | --since D --until D]
PYTHONPATH=. python -m lookout.cli.diagnostics coverage [--days 14]
PYTHONPATH=. python -m lookout.cli.diagnostics status          # archive newest vs device, and the lag
PYTHONPATH=. python -m lookout.cli.diagnostics last-record     # the device's live reading
# Writes: back up first, then save; --dry-run fetches and writes nothing
PYTHONPATH=. python -m lookout.cli.diagnostics fill-gap (--index N | --start T --end T) [--dry-run]
PYTHONPATH=. python -m lookout.cli.diagnostics energy-catalog [--save [--dry-run]]
PYTHONPATH=. python -m lookout.cli.diagnostics rain-catalog [--save]   # dry run unless --save

# Rain repair (lookout/cli/repair_rain.py): restore the raw rain counters in
# the windows #95 damaged. Dry run unless --write; see --help before running
PYTHONPATH=. python -m lookout.cli.repair_rain [--window FIRST LAST] [--write]
```

### Environment Setup
```bash
# macOS dev: Use conda environment
conda activate lookout

# Alternative: Use venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Code Quality
```bash
pytest tests/
```

## Architecture Overview

### Package Structure
- `lookout/api/` - API clients for external services (Ambient Weather Network)
- `lookout/core/` - Data processing: archive merge, catalogs, rain and solar analysis
- `lookout/models/` - Data models and type definitions (dataclasses)
- `lookout/storage/` - S3/Storj integration for data archival
- `lookout/utils/` - Shared utilities (logging, date handling, dataframe helpers)
- `lookout/snapshot/` - The dashboard snapshot: `build_snapshot` (pure), its dataclass and JSON contract, and the publish to `snapshot/` on Storj
- `lookout/cli/` - Command-line scripts for maintenance tasks, and `refresh.py`, the VPS cron's run
- `lookout/bin/` - Shell scripts for cron and operational tasks
- `lookout/config.py` - Shared configuration and sensor mappings
- `web/` - The site on Vercel: Vite + TypeScript + Plotly.js, reading the snapshot from Storj

### Key Components

**Data Flow Architecture:**
1. Ambient Weather API provides real-time and historical weather data
2. The refresh run (VPS cron, every 5 min) catches the archive up in S3/Storj (parquet), updates the catalogs, and publishes the dashboard snapshot to `snapshot/`
3. The site on Vercel reads the snapshot and draws the four tabs: Overview, Rain, Rain Events, Solar

**Storage Pattern:**
- Historical data stored as parquet files in S3/Storj bucket
- Device data identified by MAC address (e.g., `98:CD:AC:22:0D:E5.parquet`)
- Archive repair functionality normalizes schema inconsistencies

**Event Catalog Management Strategy:**
- The refresh run (`lookout/cli/refresh.py`) is the only automatic writer of the Event Catalog. The manual way to rebuild it is `diagnostics rain-catalog --save`
- Archive: saved after a backup on every catchup, whether run alone or as the refresh run's first step. It is stored **raw**: rain carryover corrections are never written back; they apply once, on read (`apply_corrections` after loading), so every reader sees corrected counters (#95, PR #113)
- Event Catalog: brought up to the archive in memory on every refresh run, saved only when it holds a completed event the stored catalog lacks, matched by start time (`RainEventCatalog.save_if_events_completed`, `newly_completed`). The stored catalog is backed up first; if the backup fails, nothing is saved and the next run retries
- No stored catalog: the first completed event seeds one
- An ongoing event is re-detected under a new `event_id` every run; key events on their start (`start_ms`), never on `event_id`
- Rationale: the archive is critical raw data; the catalog is derived and regenerable, so it is written only when its completed events change

**Configuration Management:**
- Secrets read by `lookout.secrets_loader.get_secret(name)`: env var, then `~/.config/lookout/secrets.toml`; names listed in `secrets.example.toml`
- Sensor mappings in `lookout/config.py`
- Supports multiple sensor locations (outdoor, indoor, office)

### Import Conventions
- Uses fully qualified imports: `lookout.api.ambient_client`
- Centralized logging via `lookout.utils.log_util.app_logger`
- Secrets read at first use via `get_secret`, never at import

### Code Standards
- Type hints required for function parameters and returns
- Comprehensive docstrings with parameter descriptions
- Line length limit: 88 characters
- Error handling with logging instead of print statements

## Development Notes

### Testing
- Test files in `tests/` directory using **pytest**
- Install dev dependencies: `pip install -r requirements-dev.txt`
- Run tests: `pytest tests/`

### Data Processing
- Timestamps normalized to epoch milliseconds (UTC)
- Pandas DataFrames used for all data manipulation
- Schema validation and repair handled in catchup script

### Dev Mode
- The `DEV_MODE` setting (env var, or `[environment] dev = true` in the secrets file) turns on local archive caching in `awn_controller`

## Agent skills

### Issue tracker

GitHub issues at `stonematt/lookout` via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.

### Operations

Touching Storj, the VPS (`itri`), Vercel, Healthchecks or the Ambient API → `docs/agents/operations.md` first.
