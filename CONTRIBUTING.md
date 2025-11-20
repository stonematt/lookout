# Contributing Guidelines

## Development Setup

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks (if configured)
pre-commit install
```

## Development Commands

### Code Quality
- **Format code**: `black .`
- **Lint code**: `flake8 .`
- **Type checking**: `mypy .` (if configured)

### Testing
- **Run all tests**: `pytest`
- **Run specific test**: `pytest path/to/test_file.py::test_function_name`
- **Run with coverage**: `pytest --cov=.`

### Running the Application
- **Development**: `python main.py` or `streamlit run streamlit_app.py` (adjust as needed)

## Code Style Standards

### Formatting
- **Line limit**: 88 characters (Black default)
- **Formatter**: Black
- **Linter**: flake8

### Import Organization
```python
# Standard library imports
import os
import sys
from typing import List, Optional

# Third-party imports
import pandas as pd
import numpy as np

# Local imports
from project.module import function
from project.utils import helper
```

### Type Hints
Type hints are **required** for all function parameters and return values:

```python
def process_data(data: pd.DataFrame, threshold: float) -> List[str]:
    """Process the input data and return filtered results.
    
    :param data: Input DataFrame to process
    :param threshold: Minimum threshold value
    :return: List of filtered results
    """
    pass
```

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private members**: Leading underscore `_private_method`

### Documentation
Docstrings are **required** for all modules and public functions:

```python
def calculate_metrics(values: List[float]) -> dict:
    """Calculate statistical metrics for a list of values.
    
    :param values: List of numeric values to analyze
    :return: Dictionary containing mean, median, and std deviation
    """
    pass
```

### Error Handling
Use structured logging instead of print statements:

```python
import logging

logger = logging.getLogger(__name__)

def process_file(filepath: str) -> None:
    try:
        # Processing logic
        logger.info("Successfully processed file: %s", filepath)
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        raise
```

## Branching Strategy

### Branch Structure
- **`main`**: Development/pre-release branch (always start new features from here)
- **`live`/`production`**: Production branch (only receives merges from main)
- **`feature/feature-name`**: New features and enhancements
- **`fix/issue-description`**: Bug fixes and patches

### Development Workflow
1. **Create feature branch**: `git checkout -b feature/your-feature-name main`
2. **Create fix branch**: `git checkout -b fix/issue-description main`
3. **Develop and test** your changes
4. **Create Pull Request**: `feature/` or `fix/` branches → `main`
5. **Code review** and merge to main
6. **Release**: `main` → `live`/`production`

### Branch Guidelines
- **Prefer feature/fix branches** over working directly on main or production
- **Use descriptive branch names**: `feature/user-authentication`, `fix/memory-leak`
- **Delete branches** after merge to keep repository clean
- **Keep branches focused** on a single feature or fix

## Commit Guidelines

### When to Commit

**DO commit when:**
- User explicitly requests: "commit", "commit this", "commit it"
- Switching topics with uncommitted changes (ask first)
- Completing a validated, working feature milestone
- Before starting risky or experimental work (create checkpoint)

**DON'T commit when:**
- Iterating on fixes or refinements (wait for validation)
- During active debugging sessions
- Making incremental improvements
- User explicitly says "don't commit until verified"

### Commit Message Format

**Structure:**
```
Summary line (imperative mood, 50-70 characters)

Category changes:
- Specific change description using bullet points
- Group related changes together
- Use clear, concise language

Rationale: Brief explanation of technical decisions and approach.
This helps future developers understand the "why" behind changes.
```

**Example:**
```
Add user authentication and session management

Backend changes:
- Implement JWT token generation and validation
- Add user registration and login endpoints
- Create session middleware for protected routes

Frontend changes:
- Add login form with validation
- Implement token storage and refresh logic
- Update navigation to show auth status

Rationale: JWT provides stateless authentication suitable for microservices
architecture. Session middleware ensures consistent auth checking across
all protected endpoints without repetitive code.
```

### Commit Message Guidelines

**What to Include:**
- Functional changes and their purpose
- Technical approach and key decisions
- Problem→solution mapping
- Brief rationale for non-obvious choices

**What to Avoid:**
- Test results and output samples
- Line-by-line change descriptions
- Code snippets (the diff shows this)
- Preamble like "This commit..."
- Debugging details and temporary fixes

**Focus on:**
- "Why" and architectural choices
- Scannable bullet points for human review
- Clean, focused commits on delivered functionality

## Pull Request Process

### Before Creating PR
1. **Run tests**: Ensure all tests pass
2. **Code quality**: Run `black .` and `flake8 .`
3. **Update documentation**: If applicable
4. **Self-review**: Review your own changes first

### PR Requirements
- **Clear title**: Summarize changes in imperative mood
- **Detailed description**: Explain what changed and why
- **Test coverage**: Include tests for new functionality
- **Documentation**: Update relevant documentation
- **Screenshots**: For UI changes, include before/after

### Review Process
1. **Code review**: At least one approval required
2. **Tests must pass**: CI/CD pipeline validation
3. **Merge to main**: After approval and successful tests
4. **Delete branch**: Clean up after merge

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests for individual functions
├── integration/    # Integration tests for component interaction
├── e2e/           # End-to-end tests for full workflows
└── fixtures/      # Test data and mock objects
```

### Writing Tests
- **Descriptive names**: `test_function_returns_expected_result_for_valid_input`
- **Arrange-Act-Assert**: Structure tests clearly
- **Mock external dependencies**: Isolate code under test
- **Test edge cases**: Include error conditions and boundary values

### Test Coverage
- **Aim for 80%+ coverage** on critical paths
- **Focus on business logic** over simple getters/setters
- **Test error paths** as well as success paths

## Getting Help

### Resources
- **Project documentation**: Check README.md and docs/
- **Code examples**: Look at existing implementations
- **Team communication**: Use project channels for questions

### Troubleshooting
- **Check logs**: Look for error messages and stack traces
- **Run locally**: Reproduce issues in development environment
- **Search issues**: Check if problem has been reported before

---

Thank you for contributing! Following these guidelines helps maintain code quality and makes collaboration easier for everyone.