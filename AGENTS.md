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

## Branching Strategy

### Workflow Pattern
1. **main** = Development/pre-release branch (always start new features from here)
2. **feature/branch-name** = New features and enhancements
3. **fix/branch-name** = Bug fixes and patches
4. **live** = Production branch (only receives merges from main)

### Development Flow
- Create new branches: `git checkout -b feature/feature-name main`
- Create fix branches: `git checkout -b fix/issue-description main`
- Pull Requests: `feature/` or `fix/` branches → `main` (development integration)
- Releases: `main` → `live` (production deployment)

### Branch Guidelines
- **Prefer working on feature/fix branches** - create appropriate branches for development
- **Resist working directly on main or live** - only for emergency fixes or release merges
- **Delete feature/fix branches after merge** - keep repository clean
- **Use descriptive branch names** - e.g., `feature/heatmap-improvements`, `fix/data-validation-error`

### Agent Branch Strategy
- **Always create branches for multi-step tasks** - Use todo list to track complex work
- **Branch naming**: `fix/` for bug fixes, `feature/` for enhancements
- **Commit strategy**: Commit logical milestones, not every small change
- **PR creation**: Create PRs when work is complete and tested
- **Merge target**: Always merge to `main` branch first, then `main` → `live` for releases

## Commit Guidelines

### When to Commit

**DO commit when:**
- User explicitly directs: "commit", "commit this", "commit it"
- User switches topics with uncommitted changes (remind user first)
- Completing a validated, working feature milestone
- Before starting risky or experimental work (create checkpoint)

**DON'T commit when:**
- Iterating on fixes or refinements (wait for validation)
- During active debugging sessions
- Making incremental improvements to UI/metrics/quality
- User explicitly says "don't commit until verified" or similar

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

### Commit Message Style

**Structure:**
- Summary line (imperative mood, 50-70 chars)
- Blank line
- Bullet points grouped by category
- Rationale section (2-3 lines explaining technical decisions)

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

**Guidelines:**
- Scannable bullet points for human review
- Rationale section for agent context and technical decisions
- Focus on "why" and architectural choices
- Avoid test results, line numbers, verbose descriptions

**What to Include:**
- Technical approach and key decisions
- Problem→solution mapping
- Cross-references to external analysis
- Brief rationale for non-obvious choices

**What to Avoid:**
- Test results and output samples
- Line-by-line change descriptions
- Code snippets (diff shows this)
- Preamble like "This commit..."

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
