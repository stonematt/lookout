# Agent Guidelines for Lookout Weather Dashboard

> **Note**: General development standards (code style, branching, commits) are documented in [CONTRIBUTING.md](CONTRIBUTING.md). This file contains project-specific patterns and agent behaviors.
> 
> **For all messaging standards including commits, pull requests, and merge commits, see [CONTRIBUTION_STYLE_GUIDE.md](CONTRIBUTION_STYLE_GUIDE.md).

## Build/Lint/Test Commands
- Run app: `streamlit run streamlit_app.py`
- Format: `black .`
- Lint: `flake8 .`
- Run tests: `pytest` (test directory exists but is empty - no tests configured yet)
- Run single test: `pytest path/to/test_file.py::test_function_name`
- CLI scripts: `PYTHONPATH=. python lookout/cli/script_name.py`

## Project-Specific Code Style
- **Imports**: Fully qualified (`lookout.api.ambient_client`), grouped: stdlib → third-party → local
- **Error Handling**: Use `lookout.utils.log_util.app_logger(__name__)` instead of print statements
- **Docstrings**: Follow format in `code_standards.md` for consistency with existing codebase

## Key Patterns
- Secrets via `st.secrets` (`.streamlit/secrets.toml`)
- Timestamps as epoch milliseconds (UTC)
- Data storage as parquet in S3/Storj, keyed by MAC address
- Pandas DataFrames for all data manipulation
- **Archive data is reverse sorted by date** (newest first, oldest last)
- **When examining data, use .tail() for oldest records, .head() for newest**
- **Archive updates every 1-2 days; app fills gap with live Ambient data**
- **Event catalog updates during app runtime with combined archive + live data**
- **Session data is always current, even if cloud archive is stale**

## Archive Data Schema (AI Reference)

### Column Definitions
| Column | Type | Purpose | AI Usage Pattern |
|---------|------|---------|------------------|
| `date` | `datetime64[ns, America/Los_Angeles]` | Primary time column (TZ-aware) | `df['date']` |
| `dateutc` | `int64` | UTC timestamp (milliseconds) | `df['dateutc']` |
| `solarradiation` | `float64` | Solar radiation (W/m²) | `df['solarradiation']` |
| `tempf` | `float64` | Temperature (°F) | `df['tempf']` |
| `dailyrainin` | `float64` | Daily rainfall (inches) | `df['dailyrainin']` |
| `baromrelin` | `float64` | Barometric pressure | `df['baromrelin']` |

### Critical AI Rules
- **NEVER use `df['datetime']`** - column doesn't exist in archive
- **ALWAYS use `df['date']`** for time operations (already TZ-aware)
- **NEVER convert `df['date']` timezone** - already correct
- **ALWAYS check column existence**: `if 'col' in df.columns`

### ✅ CORRECT AI Patterns
```python
# Time access
df['date']                    # ✅ Primary time column
df['dateutc']                 # ✅ UTC timestamp

# Data validation
if 'date' in df.columns:     # ✅ Always check existence
if not df.empty:               # ✅ Check DataFrame emptiness

# Timezone handling
df['date']                    # ✅ Already TZ-aware
```

### ❌ FORBIDDEN AI Patterns
```python
# NEVER use these
df['datetime']                 # ❌ Column doesn't exist
df['date'].dt.date            # ❌ Loses timezone info
pd.to_datetime(df['dateutc'])  # ❌ Redundant conversion
df['datetime'] = df['date']   # ❌ Creating wrong column
```

### AI Quick Reference
```python
# Get time column
time_col = df['date']  # NOT df['datetime']

# Check if solar data available
has_solar = 'solarradiation' in df.columns and df['solarradiation'].notna().any()

# Validate archive structure
required_cols = ['date', 'dateutc', 'solarradiation']
missing = [col for col in required_cols if col not in df.columns]
```

## Energy Catalog Management

Energy catalogs provide 15-minute solar energy period caching following the same pattern as rain_events:

- **Storage**: `{mac_address}.energy_catalog.parquet` in Storj bucket
- **Session caching**: `st.session_state["energy_catalog"]` for instant loads
- **Auto-save**: Catalogs >2 days old are automatically saved to storage
- **Management**: Regenerate/save buttons available in diagnostics tab
- **Performance**: 30x faster cached loads (<1s vs 30s) using vectorized algorithms
- **Incremental updates**: Only processes new data since last period

## Agent-Specific Behaviors

### Branch Strategy for Agents
- **Always create branches for multi-step tasks** - Use todo list to track complex work
- **Branch naming**: `fix/` for bug fixes, `feature/` for enhancements
- **Commit strategy**: Commit logical milestones, not every small change
- **PR creation**: Create PRs when work is complete and tested
- **Merge target**: Always merge to `main` branch first, then `main` → `live` for releases

### Commit Decision Protocol

**When to Commit:**
- User explicitly directs: "commit", "commit this", "commit it"
- User switches topics with uncommitted changes (remind user first)
- Completing a validated, working feature milestone
- Before starting risky or experimental work (create checkpoint)

**When NOT to Commit:**
- Iterating on fixes or refinements (wait for validation)
- During active debugging sessions
- Making incremental improvements to UI/metrics/quality
- User explicitly says "don't commit until verified"

### Topic Change Protocol

When user shifts topics with uncommitted changes, remind them:

**Example:**
```
User: "Let's work on intensity-duration curves now"

Agent: "You have uncommitted changes to rain_events.py and rain.py 
(event quality fixes). Would you like to commit these first?"
```

**Pattern:**
- Acknowledge uncommitted work
- Ask user preference
- Don't auto-commit without permission

### Commit Message Style for Agents

**Structure:**
- Summary line (imperative mood, 50-70 chars)
- Blank line
- Bullet points grouped by category
- Rationale section (2-3 lines explaining technical decisions)

**Guidelines:**
- Focus on actual functional changes, not QA debugging details
- Don't include tactical fixes found during QA in commit messages
- Keep commits clean and focused on delivered functionality

**Example:**
```
Fix event quality metrics and add Pacific timezone display

Core fixes:
- Ongoing flag: only chronologically last event (others have definitive ends)
- Max rate: derive from dailyrainin intervals using 10-min rolling method
- Completeness: interval-based calculation prevents >100% values

UI improvements:
- Convert event timestamps to America/Los_Angeles for display
- Add regeneration logging and immediate UI refresh

Rationale: hourlyrainin is 60-min rolling accumulation, not instantaneous rate.
Proper rate uses dailyrainin.diff() * 12 for 5-min intervals,
then rolling(2).sum() * 6 for 10-min rate matching console definition.
```

**Rationale Section Value:**
The brief "Rationale:" paragraph helps agents understand design decisions
and make consistent choices in related code. It encodes domain knowledge
that isn't obvious from code alone.

**Example rationale notes:**
```
Rationale: Session-first approach eliminates Storj timing issues with st.rerun().
```
```
Rationale: Rolling window comparison provides true statistical context vs calendar-date matching.
```
```
Rationale: 5-minute intervals are normal; only gaps >10min indicate data loss.
```

### UI Layer Architecture Patterns

**Separation of Concerns:**
- UI modules (`lookout/ui/*.py`) should only handle presentation logic
- Data processing belongs in core modules (`lookout/core/*.py`)
- Render functions focus on display, user interaction, and caching
- Business logic, calculations, and data transformations stay in core layer

**Import Organization:**
- ALL imports at top of file, grouped: stdlib → third-party → local
- NO imports inside functions unless absolutely necessary
- Use fully qualified imports for local modules (`lookout.core.module`)
- Follow existing code standards in CONTRIBUTING.md

**Code Structure:**
- UI functions call core modules for data processing
- Core modules return processed data ready for display
- UI layer handles Streamlit widgets, charts, and user interaction
- Avoid data processing, calculations, or business logic in UI layer

**Streamlit Deprecations (AVOID):**
- ❌ `st.plotly_chart(fig, use_container_width=True)` → ✅ `st.plotly_chart(fig, width='stretch')`
- Always use current, non-deprecated Streamlit parameters to avoid warnings

**Example Pattern:**
```python
# UI layer - only presentation
import lookout.core.data_processing as lo_dp
import lookout.core.visualization as lo_viz

def render():
    df = st.session_state["history_df"]
    
    # Call core for data processing
    processed_data = lo_dp.calculate_statistics(df)
    
    # UI only handles display
    st.metric("Total", processed_data["total"])
    fig = lo_viz.create_chart(processed_data)
    st.plotly_chart(fig)
```

**Rationale:** Clean architecture improves maintainability, testability, and 
follows separation of concerns principle. UI modules should only handle 
presentation, while core modules contain business logic.
```
