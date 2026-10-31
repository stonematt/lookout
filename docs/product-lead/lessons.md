# Lookout product-lead lessons

One dated section per session with something to learn. Unfolded lessons load at every product-lead open.

Each lesson carries a scope. A `[project]` lesson folds into this repo's product brief or a persona file; mark it `folded <sha>` once that commit lands. A `[skill]` lesson is about the product-lead skill itself: roll it up as a comment on stonematt/stonematt-skills#137 and mark it `rolled up <comment url>`, then `folded <sha>` once SKILL.md changes.

<!--
## YYYY-MM-DD — <session topic>
- [project] **<What happened, one line.>** <Why it mattered.> Fold into: <brief section or docs/panel/<persona>.md, and the proposed wording>.
- [skill] **<What happened, one line.>** <Why it mattered.> Skill line to change: <section of SKILL.md, and the proposed wording>. rolled up <comment url>
-->

## 2026-10-30 — setup

- [skill] **The status-page default `2a/<area>/<project>-status.md` is wrong for a fleet of product leads.** Lookout has no area, and the director rolls pages up by folder. Skill line to change: SETUP.md step 2 "Vault page" and `templates/status-page.md` — every page goes to `product-leads/<repo-name>.md` at the vault root, with frontmatter `Created`, `updated`, `repo`, `area` (wikilink or empty), `tags: [product-lead/status]`, `waits_on_matt` (integer); create the folder through the obsidian CLI if missing. The director offered to roll this up on #137.
- [skill] **The panel setup only produces tactical reviewers.** Matt approved the four Lookout personas as tactical, and said larger design and strategy efforts want a panel of market experts. Skill line to change: SETUP.md step 3 and SKILL.md step 5 — two panels, a tactical one seated at QA, UAT and deploy, and a strategy panel of market experts seated at wayfind, grill and prototype; setup drafts the tactical panel and leaves the strategy panel for the first strategy effort. rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6536032935

## 2026-10-30 — first status run and spec #66 prep

- [skill] **The brief existed only on the unmerged setup branch, so step 1's "no brief → run setup" nearly ran setup again.** PR #88 was still open; only a `git log --all -- docs/product-lead` check found it. Skill line to change: SKILL.md step 1.1 — before running setup, look for an open `docs(product-lead)` PR or branch, read the brief and lessons from it, and name the open PR as a waits-on-Matt item. rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6539983937

## 2026-10-31 — #95 repair follow-ups and #132/#133 triage

- [project] **A bug fix and the data cleanup it enables are separate tickets.** Matt wants the agent ticket to fix code only, and the cleanup filed as its own `ready-for-human` ticket, blocked by the fix, so the product lead remembers to queue it as an operations session (#132 → #143). Fold into: brief, "Where tickets live" — "When a fix leaves bad data in Storj, file the cleanup as a `ready-for-human` ticket blocked by the fix; queue it as its own session once the blockers merge." folded d16014f
- [skill] **Issues go through full triage before reaching a swarm.** #132 and #133 went to a swarm untriaged and Matt corrected it: verify, check redundancy, recommend, wait for go. Skill line to change: SKILL.md step 4, "When he says go" — a ticket enters a build or swarm only with a posted triage brief and `ready-for-agent`.
- [skill] **Multi-step operations threads get their own session from the start.** The 2026-10-30 session hit 250k tokens by running the #95 repair, three triage rounds and two swarms' traffic inline. Skill line to change: SKILL.md intro and step 4 — repairs, deploys and investigations with follow-ups go to a fresh session with a launch line; this session keeps decisions only.
