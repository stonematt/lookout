# Contribution Style Guide for AI Agents

## For Human Contributors
This guide provides standards for all contribution communication including commit messages, pull requests, and merge commits. AI agents should memorize core patterns and reference this guide for edge cases.

## Quick Reference for AI Agents

### Commit Messages
Title: Imperative mood, 50-70 characters maximum, no period

Structure:
- [User impact - single line if major user benefit, otherwise omit]
- [Technical bullets - 2-3 points, action-oriented]
- Rationale: [Brief why - 2-3 lines maximum]

Examples:
- Add feature: "Add rain carryover corrections with automatic runtime application"
- Fix bug: "Fix S3 bucket error in rain corrections catalog operations"  
- Refactor: "Refactor rain corrections to use centralized S3 storage operations"

### Pull Request Messages
Title: User-impact focused, dynamic header naming based on change type

Structure:
- Summary: [User value + 2-3 key action-oriented bullets]
- [Dynamic Feature Name]: [Capability bullets]
- Performance & Reliability: [Only when performance commits exist]
- [Conditional sections only when relevant]

Examples:
- Feature: "Add Rain Carryover Corrections System"
- Bug Fix: "Fix S3 Bucket Error in Catalog Operations"
- Architecture: "Centralize S3 Storage Operations"

### Merge Commits
Subject: "Merge pull request #[number] from [branch-name]" - GitHub default format
Body: Include full PR body (summary, sections, bullets)

Repository pattern: Always include "Merge" verb and branch reference for git log visibility

### Deployment Commits (main → live)
Subject: "Deploy [branch-or-description]"
Body: Multi-line only for batch deployments (list branches)

Patterns:
- Single feature: "Deploy feature/[full-branch-name]"
- Multiple features: "Deploy batch-[date]" + branch list in body
- Hotfixes: "Deploy [short-description]"

Examples:
Deploy feature/rain-carryover-corrections

Deploy batch-20250121
feature/solar-visualizations
feature/rain-carryover-corrections

Deploy header-fix

## Decision Patterns

Always prioritize user impact over technical details.
Focus on net impact, not process or journey.
Omit empty sections (Unix principle).
Use action-oriented bullets: "Identifies...", "Adds...", "Improves..."
Apply corrections transparently without disrupting existing workflows.

## Core Templates

### Commit Message Template
```
[Imperative title - 50-70 chars]

- [Technical bullet 1 - action-oriented]
- [Technical bullet 2 - action-oriented]
- [Technical bullet 3 - action-oriented]

Rationale: [Why this approach helps future agents]
```

### Pull Request Template
```
## [Dynamic User-Centric Title]

### Summary
[2-3 sentences focusing on user impact + architectural significance]

- [Action-oriented bullet 1 - user benefit]
- [Action-oriented bullet 2 - key capability]
- [Action-oriented bullet 3 - workflow improvement]

### [Dynamic Feature Name]
[Brief description of main functionality]

- Identifies [problem/need] and provides [solution]
- Applies [specific approach] for [benefit]
- Integrates with [existing systems/components]

### Performance & Reliability
[Only when performance-related commits exist]

- Processes [data size] efficiently using [approach]
- Handles [error conditions] with [fallback/recovery]

### [Conditional Sections]
[Render only when content exists]
```

## Common Scenarios

### Feature Addition
Commit: "Add [feature name] with [key benefit]"
PR: Dynamic header named after feature system
Focus: New capabilities, user workflows enabled

### Bug Fix
Commit: "Fix [problem] in [component]"
PR: Header focuses on resolution
Focus: Problem solved, reliability improvement

### Architecture Refactor
Commit: "Refactor [component] to [new approach]"
PR: Header describes structural improvement  
Focus: Maintainability, consistency, DRY principles

## Style Rules

- Line length: 88 characters max (Black default)
- Import organization: stdlib → third-party → local
- Error handling: Use `lookout.utils.log_util.app_logger(__name__)`
- Unix principles: Silent when no useful information
- Token efficiency: Minimal viable structure for agent memorization