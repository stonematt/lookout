# Agent Guidelines for Lookout Weather Dashboard

## Build/Lint/Test Commands
- Run app: `streamlit run streamlit_app.py`
- Format: `black .`
- Lint: `flake8 .`
- Run tests: `pytest` (test directory exists but is empty - no tests configured yet)
- Run single test: `pytest path/to/test_file.py::test_function_name`
- CLI scripts: `PYTHONPATH=. python lookout/cli/script_name.py`

## Code Style
- **Formatting**: Black (88 char line limit), flake8 compliant
- **Imports**: Fully qualified (`lookout.api.ambient_client`), grouped: stdlib → third-party → local
- **Types**: Type hints required for all function parameters and returns
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Docstrings**: Required for all modules/functions with `:param:` and `:return:` format (see code_standards.md)
- **Error Handling**: Use `lookout.utils.log_util.app_logger(__name__)` instead of print statements
- **Comments**: Inline comments for context, keep concise within 88 char limit

## Key Patterns
- Secrets via `st.secrets` (`.streamlit/secrets.toml`)
- Timestamps as epoch milliseconds (UTC)
- Data storage as parquet in S3/Storj, keyed by MAC address
- Pandas DataFrames for all data manipulation
