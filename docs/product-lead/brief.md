# Lookout product brief

The product lead reads this on every run. Keep it short: point at the docs that already say a thing rather than restating it.

## Goals

- A glance on a phone gives current conditions in the context of recent history.
- Deeper views for rain, Rain Events and solar energy, each stating its finding.
- The dashboard is a Vite site on Vercel that reads a snapshot a VPS cron precomputes (ADR-0003).

Product context: `PRODUCT.md` (users, principles, constraints), `DESIGN.md` (The Annotated Almanac), `GLOSSARY.md`, `docs/adr/`.

## Initiatives

One line each. The product lead infers the rung; write one here only when Matt overrides it.

- **Move Lookout off Streamlit** — replace the Streamlit app with a cron-built snapshot on Storj and a Vite + TypeScript + Plotly.js site on Vercel · containers: map #51, spec #66 and its sub-issues #67–#86, bug #63 (sub-issue of #66)
- **Deepen storage reads, archive merge and daily rain** — the three refactors kept from the 2026-11-01 architecture review · containers: specs #172, #173, #174 and their sub-issues #180–#199, goal PR #201 (merged to dev 2026-11-02), held ticket #191, follow-ups #219, #220, #224 · also #140 (duplicated energy updaters; grill with #224 as one refactor: both cover the catalog backup routes), #169 (repair backup and raw read-back, #170 folded in; lands before #224) · #173 closed 2026-11-09
- **Release 2026.11.1** — shipped dev to main and introduced versioning (ADR-0007), 2026-11-03 · ledger #277, release PR #289, tag `v2026.11.1`
- **Serve a second station: Salmon Run** — shipped 2026-11-03: Salmon Run is archived and served beside Lookout Post, from a station inventory on Storj (ADR-0006). Four releases under an autonomy grant (`docs/agents/autonomy.md`, removed when #294 closed; in git history), program ledger #294: A 2026.11.2 ops and logging (#296), B 2026.11.3 inventory and multi-station refresh run (#314), C 2026.11.4 station index, site and archive memory cuts (#325), D measured the memory peak and closed #234's gate as met-enough (Matt). Salmon Run added 21:15 UTC. · open: #275 (rain events after Salmon Run's first rain, human), #329 (archive load sets the memory peak, needs-triage), #247 step 4 (revoke the old credential, Matt) · deep retro: PR #331; case study in Matt's vault (2026-11-03)
- **Tiles That Talk redesign** — shipped 2026-11-04 as release 2026.11.6 (release PR #398, `main` `6213e96`, tag `v2026.11.6`): the site as tiles that state their findings, a phone layout with an accordion of findings, dark and light themes. Its prerequisite release 2026.11.5 (PR #356, tag `v2026.11.5`) shipped first: the site reads snapshot schema 2 and 3, and the refresh run writes 3 with the sun and moon block, so UAT ran on a goal-branch preview with live data. That rule is now ADR-0008. · containers: spec #332, goal PR #358, ledger #348 (both releases) · deferred: #374 (ten-day bar colour, needs a snapshot contract change), #328 and #134 (rain near midnight, data bugs), #353 (local UAT environment, parked) · #375, #399, #226 and #401 shipped in 2026.11.8 · open for Matt: delete the `shots/*` branches
- **Release 2026.11.8** — shipped `dev` as it stood to `main`, 2026-11-11 08:44 PT: repair backup and raw read-back (#417, closes #169), the catalog median against completed events only (#418, closes #375; changes the live snapshot), no Vercel previews for `shots/*` and `lane/*` (#404, closes #401), the N−1 snapshot constant (#413, closes #399), `diagnostics event-catalog` (#414, closes #258), docs #415, #416 (closes #226), #420 · ledger #421, bump PR #434, release PR #436, `main` `f2002a1`, tag `v2026.11.8` · CI Layout check waived by Matt (runner could not resolve the Ubuntu apt mirrors) and run locally instead; the CI fix is #437
- **Rain event data accuracy** — Python data bugs in rain events and daily rain; each needs investigation before a build · containers: #328 and #134 (rain near local midnight; likely one root cause, #328 the symptom), #116 (event end includes the dry tail), #115 (counter edge cases in corrections and merge), #135 (date and storage risks from the audit)
- **Rain Events tab** — finish the tab; every ticket edits `web/src/events/tab.ts`, so run them in sequence, never as parallel lanes · order: #239 (rough edges), then #254 (filters), then #154 (stat strip, rate chart) with #419 (total and rate on one chart, replaces the Total / Rate switch; settle with #154's rate-chart item), #261 (escaping `html` template, `refactor`), #162 (polish and test gaps)
- **New site views** — Streamlit carry-overs and new views · containers: #252 (year-over-year period selector), #253 (numbers-first conditions strip), #255 and #390 (wind: merge into one wind spec; needs a snapshot block, so a schema bump under ADR-0008)
- **Brand and product direction** — grill first · containers: #256 (consumer review and brand pass), #409 (iOS widget; draft PR #408 adds `widget_view` to the snapshot)
- **Ops and tooling** — containers: #266 (uv lockfile and `pyproject.toml`; touches CI and `itri`), #257 (post-release log and monitor checks; split into its three items before a build), #258 (rename `diagnostics rain-catalog` to `event-catalog`, BSE)
- **architecture review** · due (285 files since `a43d894`, threshold 30) · next: `/improve-codebase-architecture`

## Where the tickets live

Tracker: GitHub Issues on `stonematt/lookout`. Labels: one category (`bug`, `enhancement`) and one state (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`); see `docs/agents/`. When a fix leaves bad data in Storj, the cleanup is its own `ready-for-human` ticket, blocked by the fix; the product lead queues it as a separate session once the blockers merge (example: #143 after #132). `bse` marks a bug or small enhancement that ships on its own: no dependency on or overlap with other open work, no schema bump, little data risk; `/bse-monitor` builds those also labelled `ready-for-agent` (Matt, 2026-11-09). Specs: issues titled `Spec: …`, tickets as native sub-issues. Maps: issues labelled `wayfinder:map`. No milestones.

Architecture review candidates Matt keeps become issues labelled `refactor` + `needs-triage`, each carrying its full grilling brief; he grills from the issue (2026-11-01).

Branch flow: feature branch → `dev` (integration) → `main` (release). Vercel deploys `main` to production, https://lookout.stonecover.com; there is no separate deploy branch (Streamlit and its `live` branch retired by #86). Research notes: `docs/research/`, wayfinder research on unmerged `research/<name>` branches.
Each `dev` → `main` release bumps `VERSION` through a `chore(release)` PR into `dev` and is tagged `v<version>` on `main` after production is verified (ADR-0007). Merge the `chore(release)` PR, then open the release PR at once; the release head must equal that PR's tested head.
Vercel is on the Hobby plan: about 100 deployments a day, and every branch push builds a preview, `shots/*` and swarm lane branches included. A build-and-UAT cycle can spend it in a morning (2026-11-04), and the limit blocks the production deploy from `main` too. Plan pushes against it: one push per fix PR in UAT, and no `shots/*` pushes once UAT starts. PR #404 (#401) skips previews for `shots/*` and `lane/*` through `ignoreCommand`; lane branches only skip once the swarm skill's lane brief cuts `lane/<slug>` (it still hard-codes `fix/<slug>` in `~/.claude/skills/swarm/LANE-BRIEF.md`).

## Drift

| Last architecture review | Date | Threshold |
|---|---|---|
| `a43d894` (head of `goal/off-streamlit`; report kept 5 candidates, 3 became #172–#174) | 2026-11-01 | 30 files or 1,500 lines |

Threshold history: none yet.

## Vault page

Vault: `tyee` · path: `product-leads/lookout.md` (director convention, 2026-10-30) · area: none
