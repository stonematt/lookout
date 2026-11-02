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

- [project] **A bug fix and the data cleanup it enables are separate tickets.** Matt wants the agent ticket to fix code only, and the cleanup filed as its own `ready-for-human` ticket, blocked by the fix, so the product lead remembers to queue it as an operations session (#132 → #143). Fold into: brief, "Where tickets live" — "When a fix leaves bad data in Storj, file the cleanup as a `ready-for-human` ticket blocked by the fix; queue it as its own session once the blockers merge." folded d16014f · also [skill], rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6549992604
- [skill] **Issues go through full triage before reaching a swarm.** #132 and #133 went to a swarm untriaged and Matt corrected it: verify, check redundancy, recommend, wait for go. Skill line to change: SKILL.md step 4, "When he says go" — a ticket enters a build or swarm only with a posted triage brief and `ready-for-agent`.  rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6549992604
- [skill] **Multi-step operations threads get their own session from the start.** The 2026-10-30 session hit 250k tokens by running the #95 repair, three triage rounds and two swarms' traffic inline. Skill line to change: SKILL.md intro and step 4 — repairs, deploys and investigations with follow-ups go to a fresh session with a launch line; this session keeps decisions only.  rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6549992604

## 2026-10-31 — product-lead stance

- [skill] **Matt wants an opinionated product lead that finishes what we start.** He corrected the first draft three times: scope that the work uncovers is a judgment call, not a rule; agents in flight and their collisions matter more than ticket counts; and issues are named by objective ("the rain gauge data fix"), never by bare number. Skill line to change: a new "Stance" section after the SKILL.md intro, and a WIP and collision line in the step 4 report (stonematt/stonematt-skills#142). rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6550171650
- [skill] **Check before you recommend.** The product lead asked Matt to have the repair session check its code path against the swarm's storage fix, when three commands of its own showed there was no collision. Skill line to change: Stance section, "Check before you recommend" (stonematt/stonematt-skills#142). rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6550171650
- [skill] **A relayed go does not clear a gate that is Matt's.** The rain correction repair session kept asking "Run the `--write` now?" after the product lead relayed Matt's approval. That is correct for an irreversible write. Skill line to change: Stance section, "Know the agents in flight" — relay the facts, never the approval (stonematt/stonematt-skills#142). rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6550171650

## 2026-11-01 — storage read outcomes grilling (#172)

- [skill] **Tickets were filed by hand, skipping `/to-spec` and `/to-tickets`.** Both skills set `disable-model-invocation`, so they were missing from the session's skill list, and the product lead filled the gap with `gh issue create`. Matt: "use the pocock skills for their key steps grill->spec (if needed) ->tickets - don't roll your own". The hand-filed tickets were closed and the split was redone through the skills. Skill line to change: SKILL.md step 1.3 — each rung runs through its ladder skill; read hidden skills from `~/.claude/skills/<name>/SKILL.md`; ask Matt if one is missing. rolled up https://github.com/stonematt/stonematt-skills/issues/137#issuecomment-6563142813
- [project] **"Lookout Post" is this station's name, not the general term.** The glossary used "lookout post" for any Ambient station, and the grilling repeated it. The general term is **station**. The second station, Salmon Run, made the difference matter. Fold into: GLOSSARY.md **station** entry (PR #175). folded 6296e10
