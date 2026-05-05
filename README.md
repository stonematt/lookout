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

Create a `.streamlit/secrets.toml` file or use Streamlit CLI to store:

```toml
AMBIENT_API_KEY = "..."
AMBIENT_APPLICATION_KEY = "..."
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."

# connnect to the lookout bucket on storj.io
[lookout_storage_options]
ACCESS_KEY_ID = "..."
SECRET_ACCESS_KEY = "..."
ENDPOINT_URL = "https://gateway.storjshare.io"

──
```

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
- Domain glossary in [`CONTEXT.md`](CONTEXT.md); architectural decisions in
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
