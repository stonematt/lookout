# Agent Guidelines for Lookout Weather Dashboard

> **Note**: General development standards (code style, branching, commits) are documented in [CONTRIBUTING.md](CONTRIBUTING.md). This file contains project-specific patterns and agent behaviors.

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
