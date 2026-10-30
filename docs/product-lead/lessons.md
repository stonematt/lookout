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
- [skill] **The panel setup only produces tactical reviewers.** Matt approved the four Lookout personas as tactical, and said larger design and strategy efforts want a panel of market experts. Skill line to change: SETUP.md step 3 and SKILL.md step 5 — two panels, a tactical one seated at QA, UAT and deploy, and a strategy panel of market experts seated at wayfind, grill and prototype; setup drafts the tactical panel and leaves the strategy panel for the first strategy effort.
