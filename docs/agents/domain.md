# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This is a **single-context** repo — one `GLOSSARY.md` + `docs/adr/` at the root.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root
- **`docs/adr/`** — read ADRs that touch the area you're about to work in

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

```
/
├── GLOSSARY.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── lookout/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Where a grill's glossary and ADR edits land

`/grill-with-docs` writes glossary terms and ADRs as they settle. Ship them
in a `docs/*` PR into the branch the grilled work targets (its spec names
the target branch), not inside a code PR.

Before opening one, list the open PRs that touch `GLOSSARY.md` or
`docs/adr/`:
`gh pr list --state open --json number,headRefName,baseRefName,files --jq '.[] | select((.headRefName | startswith("docs/")) and any(.files[].path; test("GLOSSARY.md|docs/adr/"))) | "\(.number) \(.headRefName) -> \(.baseRefName)"'`

- **None open:** branch off the target and PR into it.
- **Some open:** branch off the newest one and PR into its branch. State
  the merge order in the PR body ("merge #N first"). Take the next free
  ADR number across the whole stack, not only the target branch.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
