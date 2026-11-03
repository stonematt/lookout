# 🌤️ Lookout

**Lookout** is a weather station dashboard, live at
[lookout.stonecover.com](https://lookout.stonecover.com). A cron on a VPS pulls
readings from the Ambient Weather API into an archive on S3-compatible storage
(Storj) and publishes a dashboard snapshot there. A Vite + TypeScript site on
Vercel reads the snapshot and draws it ([ADR-0003](docs/adr/0003-vps-cron-precomputes-vercel-only-reads.md)).
CLI scripts handle maintenance tasks.

---

## 📁 Project Structure

```
lookout/
├── api/            # API clients (e.g. Ambient Weather)
├── core/           # Data processing: archive merge, catalogs, rain and solar analysis
├── snapshot/       # The dashboard snapshot: build, JSON contract, publish
├── storage/        # S3/Storj integration
├── utils/          # Logging and shared utilities
├── cli/            # Command-line scripts (catchup, refresh, diagnostics)
├── bin/            # Shell scripts for cron
├── config.py       # Shared config
web/                # The site on Vercel (Vite + TypeScript + Plotly.js)
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/stonematt/lookout.git
cd lookout
```

### 2. Set up virtual environment

With [uv](https://docs.astral.sh/uv/), on the same Python 3.12 the VPS and CI run:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt
```

### 3. Configure secrets

Copy `secrets.example.toml` to `~/.config/lookout/secrets.toml`, fill it in and
`chmod 600` it. `lookout.secrets_loader.get_secret(name)` reads each secret on
first use, in this order:

1. an env var of the same flat name (`AMBIENT_API_KEY`, `STORJ_ENDPOINT_URL`, ...);
2. `~/.config/lookout/secrets.toml`.

The three Storj names (`STORJ_ACCESS_KEY_ID`, `STORJ_SECRET_ACCESS_KEY`,
`STORJ_ENDPOINT_URL`) map to the `[lookout_storage_options]` table.

Tests need no secrets: `python -m pytest tests/` runs on a clean checkout.

---

## 🚀 Usage

### Run one refresh

The VPS cron's job: catchup, catalogs, then the snapshot publish.

```bash
PYTHONPATH=. python lookout/cli/refresh.py
```

### Run the catchup CLI script

```bash
PYTHONPATH=. python lookout/cli/catchup.py
```

Use with cron (with logging; `--sleep N` adds up to N seconds of jitter,
default 0):

```bash
lookout/bin/cron_run.sh lookout/cli/refresh.py
```

The refresh run, catchup and the other archive writers take a per-host lock
themselves (`~/.local/state/lookout/archive-writer.lock`, moved by
`LOOKOUT_LOCK_FILE`), so at most one runs at a time and a crontab needs no
`flock`.

### Run the site locally

```bash
cd web
npm ci
npm run dev
```

---

## 🧪 Testing

```bash
pytest tests/
cd web && npm run format:check && npm run lint && npm run build && npm test
```

---

## 🛠 Developer Notes

- Code standards, the checks CI runs, and the rules for agents are in
  [`CLAUDE.md`](CLAUDE.md); contributing workflow in [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Domain glossary in [`GLOSSARY.md`](GLOSSARY.md); architectural decisions in
  [`docs/adr/`](docs/adr/) (e.g.
  [ADR-0001 — post-2020 dateutc floor](docs/adr/0001-dateutc-post-2020-floor.md),
  [ADR-0002 — station-local time as analytical frame](docs/adr/0002-station-local-time-as-analytical-frame.md))

---

## 📡 Dependencies

- Pandas / NumPy / PyArrow
- Requests
- Boto3 (for S3/Storj)
- Ambient Weather API
- The site: Vite, TypeScript, Plotly.js

---

## 🧠 Author & License

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Z8Z41G13PX)

Maintained by [@stonematt](https://github.com/stonematt)  
Licensed under the MIT License
