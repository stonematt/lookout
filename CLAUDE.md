# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lookout is a modular weather station dashboard built with Streamlit, Ambient Weather API, and S3-compatible storage (Storj). It supports data collection, archiving, visualization, and CLI-based maintenance tasks.

## Common Development Commands

### Running the Application
```bash
streamlit run streamlit_app.py
```

### Running CLI Scripts
```bash
# Run catchup script to sync new data from Ambient Weather API
PYTHONPATH=. python lookout/cli/catchup.py

# Run with cron (includes logging and jitter)
lookout/bin/cron_run.sh --sleep 300 lookout/cli/catchup.py
```

### Environment Setup
```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Alternative: Use conda environment
conda env create -f lookout_env.yml
conda activate lookout
```

### Code Quality
```bash
# Format code with Black
black .

# Lint with flake8
flake8 .
```

## Architecture Overview

### Package Structure
- `lookout/api/` - API clients for external services (Ambient Weather Network)
- `lookout/core/` - Data processing and visualization logic
- `lookout/storage/` - S3/Storj integration for data archival
- `lookout/utils/` - Shared utilities (logging, date handling, dataframe helpers)
- `lookout/cli/` - Command-line scripts for maintenance tasks
- `lookout/ui/` - Streamlit UI components (overview, diagnostics)
- `lookout/config.py` - Shared configuration and sensor mappings

### Key Components

**Data Flow Architecture:**
1. Ambient Weather API provides real-time and historical weather data
2. CLI catchup script synchronizes new data to S3/Storj storage in parquet format
3. Streamlit dashboard loads data from storage and displays visualizations
4. Auto-refresh mechanism keeps dashboard current

**Storage Pattern:**
- Historical data stored as parquet files in S3/Storj bucket
- Device data identified by MAC address (e.g., `98:CD:AC:22:0D:E5.parquet`)
- Archive repair functionality normalizes schema inconsistencies

**Configuration Management:**
- Secrets stored in `.streamlit/secrets.toml` (API keys, storage credentials)
- Sensor mappings and UI configurations in `lookout/config.py`
- Supports multiple sensor locations (outdoor, indoor, office)

### Import Conventions
- Uses fully qualified imports: `lookout.api.ambient_client`
- Centralized logging via `lookout.utils.log_util.app_logger`
- Streamlit secrets accessed via `st.secrets`

### Code Standards
- Follows Black formatting and flake8 compliance
- Type hints required for function parameters and returns
- Comprehensive docstrings with parameter descriptions
- Line length limit: 88 characters
- Error handling with logging instead of print statements

## Development Notes

### Testing
- Test files in `test/` directory mirror the `lookout/` package structure
- No specific test framework configured - check existing patterns

### Data Processing
- Timestamps normalized to epoch milliseconds (UTC)
- Pandas DataFrames used for all data manipulation
- Schema validation and repair handled in catchup script

### UI Components
- Modular Streamlit tabs: Overview and Diagnostics
- Gauge configurations defined in `config.py`
- Auto-refresh with configurable intervals (6 minutes to 3 days)