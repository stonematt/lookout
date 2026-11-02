# 🌤️ Lookout

**Lookout** is a modular weather station dashboard built with Streamlit, Ambient Weather API, and S3-compatible storage (Storj). It supports data collection, archiving, visualization, and CLI-based maintenance tasks.

---

## 📁 Project Structure

```
lookout/
├── api/            # API clients (e.g. Ambient Weather)
├── core/           # Data processing and visualization
├── storage/        # S3/Storj integration
├── utils/          # Logging and shared utilities
├── cli/            # Command-line scripts (e.g. catchup)
├── config.py       # Shared config
├── __init__.py     # Package root
streamlit_app.py    # Main dashboard entry point
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/lookout.git
cd lookout
```

### 2. Set up virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Optional: Use `env_lookout_frozen.yml` if you're managing with Conda.

### 3. Configure secrets

Copy `secrets.example.toml` to `~/.config/lookout/secrets.toml`, fill it in and
`chmod 600` it. `lookout.secrets_loader.get_secret(name)` reads each secret on
first use, in this order:

1. an env var of the same flat name (`AMBIENT_API_KEY`, `STORJ_ENDPOINT_URL`, ...);
2. `~/.config/lookout/secrets.toml`;
3. `st.secrets` (`.streamlit/secrets.toml`), while Streamlit is loaded.

The three Storj names (`STORJ_ACCESS_KEY_ID`, `STORJ_SECRET_ACCESS_KEY`,
`STORJ_ENDPOINT_URL`) map to the `[lookout_storage_options]` table, so an
existing `.streamlit/secrets.toml` can be copied over unchanged.

Tests need no secrets: `pytest tests/` runs on a clean checkout.

---

## 🚀 Usage

### Run the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

### Run the catchup CLI script

```bash
PYTHONPATH=. python lookout/cli/catchup.py
```

Use with cron (with logging + jitter):

```bash
lookout/bin/cron_run.sh --sleep 300 lookout/cli/catchup.py
```

---

## 🧪 Testing

Test files live in the `test/` directory and mirror the `lookout/` package structure.

---

## 🛠 Developer Notes

- Code formatted with **Black** and **Ruff**
- Imports are fully qualified (`lookout.api.ambient_client`)
- Logging is centralized in `log_util.py`
- Follows standards defined in [`code_standards.md`](code_standards.md)
- Domain glossary in [`GLOSSARY.md`](GLOSSARY.md); architectural decisions in
  [`docs/adr/`](docs/adr/) (e.g.
  [ADR-0001 — post-2020 dateutc floor](docs/adr/0001-dateutc-post-2020-floor.md),
  [ADR-0002 — station-local time as analytical frame](docs/adr/0002-station-local-time-as-analytical-frame.md))

---

## 📡 Dependencies

- Streamlit
- Pandas / NumPy
- Requests
- Boto3 (for S3/Storj)
- Plotly
- Ambient Weather API

---

## 🧠 Author & License

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Z8Z41G13PX)

Maintained by [@stonematt](https://github.com/stonematt)  
Licensed under the MIT License
