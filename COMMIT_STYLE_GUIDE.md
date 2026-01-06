# Commit Message Style Guide for Agents

## Format
```
[Title Line - 70 chars max]

[User Impact - very tactical, what user gets]

[Technical Changes - terse, what changed technically]

Rationale: [why this approach, for future AI agents]
```

## Title Line
- Imperative mood (Add, Fix, Update, Remove, Refactor)
- 50-70 characters maximum
- No period at end
- Focus on what changed, not how

## User Impact
- Very tactical - what the user gets immediately
- One line maximum
- Focus on functionality, not implementation
- Omit entirely for internal changes (Unix silence principle)

## Technical Changes
- Terse bullet points
- What changed technically
- No implementation details unless critical
- Group related changes

## Rationale
- Explain why this approach was chosen
- Help future AI agents understand design decisions
- 2-3 lines maximum
- Focus on architectural or pattern decisions

## Examples

### Good: Feature Implementation
```
Add rain carryover corrections with automatic runtime application

Users get accurate rainfall measurements across daily, weekly, monthly, yearly periods.

Core corrections apply automatically to all weather data loaded through the application.
- Add apply_corrections() function with period-aware adjustments
- Apply corrections once at end of get_history_since_last_archive()
- Error handling with fallback to uncorrected data

Rationale: Corrections applied at single integration point ensures consistent application across all data loading operations without coupling merge logic to correction system.
```

### Good: Bug Fix
```
Fix S3 bucket error in rain corrections catalog operations

Catalog generation and display now work correctly.

Replaced legacy save_json_to_path() with direct S3 client operations to fix bucket parsing bug.

Rationale: Legacy functions designed for local file backups caused S3 path parsing failures; direct client operations follow established patterns from working CLI modules.
```

### Good: Refactor
```
Clean rain corrections catalog loading logic

Replaced legacy read_json_from_path() with direct S3 client operations for consistency.

Rationale: Aligns with established S3 storage patterns used successfully in other CLI modules, reducing maintenance burden.
```

## Rules
- **Never enumerate test counts** (e.g., "21 tests passing")
- **No implementation details** unless critical for understanding
- **Focus on user impact** over technical minutiae
- **Rationale helps future agents** make similar decisions
- **Keep total message under 300 characters** when possible
- **Omit user impact for internal changes** - Unix silence principle