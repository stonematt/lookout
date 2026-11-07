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
- **Deepen storage reads, archive merge and daily rain** — the three refactors kept from the 2026-11-01 architecture review · containers: specs #172, #173, #174 and their sub-issues #180–#199, goal PR #201 (merged to dev 2026-11-02), held ticket #191, follow-ups #219, #220, #224
- **Release 2026.11.1** — shipped dev to main and introduced versioning (ADR-0007), 2026-11-03 · ledger #277, release PR #289, tag `v2026.11.1`
- **Serve a second station: Salmon Run** — shipped 2026-11-03: Salmon Run is archived and served beside Lookout Post, from a station inventory on Storj (ADR-0006). Four releases under an autonomy grant (`docs/agents/autonomy.md`, removed when #294 closed; in git history), program ledger #294: A 2026.11.2 ops and logging (#296), B 2026.11.3 inventory and multi-station refresh run (#314), C 2026.11.4 station index, site and archive memory cuts (#325), D measured the memory peak and closed #234's gate as met-enough (Matt). Salmon Run added 21:15 UTC. · open: #275 (rain events after Salmon Run's first rain, human), #329 (archive load sets the memory peak, needs-triage), #247 step 4 (revoke the old credential, Matt) · deep retro: PR #331; case study in Matt's vault (2026-11-03)
- **Tiles That Talk redesign** — shipped 2026-11-04 as release 2026.11.6 (release PR #398, `main` `6213e96`, tag `v2026.11.6`): the site as tiles that state their findings, a phone layout with an accordion of findings, dark and light themes. Its prerequisite release 2026.11.5 (PR #356, tag `v2026.11.5`) shipped first: the site reads snapshot schema 2 and 3, and the refresh run writes 3 with the sun and moon block, so UAT ran on a goal-branch preview with live data. That rule is now ADR-0008. · containers: spec #332, goal PR #358, ledger #348 (both releases) · deferred: #374 (ten-day bar colour, needs a snapshot contract change), #375 (two medians for one event, Python, predates the redesign), #328 and #134 (rain near midnight, data bugs), #353 (local UAT environment, parked), #399 (the N−1 contract test) · open for Matt: delete the `shots/*` branches; decide whether Vercel previews stay on for `shots/*` and lane branches
- **architecture review** · due (285 files since `a43d894`, threshold 30) · next: `/improve-codebase-architecture`

## Where the tickets live

Tracker: GitHub Issues on `stonematt/lookout`. Labels: one category (`bug`, `enhancement`) and one state (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`); see `docs/agents/`. When a fix leaves bad data in Storj, the cleanup is its own `ready-for-human` ticket, blocked by the fix; the product lead queues it as a separate session once the blockers merge (example: #143 after #132). Specs: issues titled `Spec: …`, tickets as native sub-issues. Maps: issues labelled `wayfinder:map`. No milestones.

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
