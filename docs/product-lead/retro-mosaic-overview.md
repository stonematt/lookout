# Mosaic Overview program retro

Deep retro over spec #460 and program #484, from the first dispatch (2026-11-11 22:58 PT) to the close (2026-11-13 10:22 PT). Sources: 41 session transcripts and their 219 subagent transcripts, #460, #426, #484, ledgers #485–#489, 102 PRs, 314 CI runs and 140 Vercel deployments. Times are Pacific. A number marked (est.) is estimated; every other number was counted. The close-out lessons (PR #626) are validated in `lessons.md`, 2026-11-13.

## Did this take too long, and why

Partly the expectation, mostly one avoidable gap. The program was about three times the size of the second-station program (#294): 102 PRs against 37, 9 releases against 3, 60 user stories. It took 35 hours against 24, so per unit of work it was faster. But 14 of those 35 hours were nights. The Mac slept from 01:20 to 07:30 on 11-12 with Release A half built, and every site release waited on A. That pushed B2, the long pole, so its UAT went to Matt at 18:08. It then crossed the second night and passed at 06:45. With the host awake on night one, B2's UAT would have landed in the day, and the program would have finished on the evening of 11-12, about 15 hours sooner (est.). The other costs were smaller: three extra Python releases for site numbers no ticket decided, a 75-minute stack conflict on B2, and C1 shipping alone on the second morning. The Vercel limit cost about 30 minutes on the critical path. Its real cost was the lost preview UAT path and agents polling a failed check.

## The spec and the plan at the start

Spec #460 (opened 11-11 17:21) has no edit history, so its first version is the current body plus dated comments. It set five releases:

| Release | Scope at the start |
|---|---|
| 1 | Web only: #458 |
| A | Python fields: #461–#467, plus #491 |
| B1 | Site: hero swap, feature tile, next-unlock line, battery line (#468–#471) |
| B2 | Site: mosaic, zooms, Inside tile, wind rose, Rain tab order, a11y, design docs (#472–#478) |
| C | Stop writing `hero_lead`: schema bump in two steps under ADR-0008 (#479 site, then #480 Python) |

Program #484's first body (11-11 23:00) had the same five rows, A as one serial lane, B2 stacked on B1's goal branch, and C already in two steps. So "C was split" was in the plan from the start; it was forced by ADR-0008. Matt's grant at 23:07–23:13: merge on green to `dev` and `main`, UAT only for 1, B1 and B2, A and both C steps checked in production, stop and ask before any raw archive write.

## How the plan changed

| When (PT) | Change | Why | Who decided |
|---|---|---|---|
| 11-11 23:13 | #479 gets more blockers | The site reads `hero_lead` in two more files | Product lead |
| 11-11 23:13 | B2 builds stacked on B1 during B1's UAT | Planned overlap | Product lead |
| 11-12 00:04 | UAT becomes judgement-only | Matt: "This UAT was too tactical and didn't really require my judgement." | Matt |
| 11-12 09:30 | A2 added (#518) | B1 lane A found no snapshot value for the hero's delta line (ADR-0003) | Product lead |
| 11-12 09:45–11:27 | A3 added: 3-hour feature hold, gust out (#533) | Matt's worry about the feature flipping; a 60-day replay | Matt: "file both, 3 hour hold and drop gust" |
| 11-12 11:36 | Sources footer and version move to B2 only | Scope read | Matt |
| 11-12 11:51 | A3 grows #539, #540, then #554 | B2 tile numbers had no Python field | Release manager proposed, product lead filed |
| 11-12 12:08 | Vercel's 100th deployment in 24 hours | Three swarms in parallel | None (found at 12:26) |
| 11-12 14:30 | Deploy rule 1: #565, deploys per branch, skip list dropped | Skipped builds still count | Matt asked; orchestrator built it |
| 11-12 14:41 | Deploy rule 2: #573 allowlist, only `main`, `dev`, `uat/**`; ADR-0010 | Matt: "uat per release, go ahead and build it" | Matt |
| 11-12 14:41 | B2's desktop feature becomes a full-width row (story 43) | The B1–B2 merge conflict exposed it | Matt (asked 13:27) |
| 11-12 15:17 | A3-site release added (#578) | The A3 panel found the held feature worded wrongly on the site | Release manager |
| 11-12 21:30 | Sources footer cut before merge (#604) | Matt: "souces and methods are 3x the body of the page. ridiculous." | Matt |
| 11-13 07:08 | Release manager relaunched | The first manager and the product lead had exited overnight | Orchestrator |
| 11-13 10:22 | #584, #597 deferred; #608 deferred at B2 UAT | Site correct without them | Matt: "yes, defer them" |

Five planned releases became nine: 1, A, A2, B1, A3-site, A3, B2, C1, C2 (2026.11.10–2026.11.18).

## Timeline

| Release | Version | Build dispatched | Goal PR ready | Merged to `dev` | Production verified | Wall time |
|---|---|---|---|---|---|---|
| 1 | 2026.11.10 | 11-11 23:04 | 23:30 | 00:02 | 00:12 | 1 h 08 m |
| A | 2026.11.11 | 11-11 23:04 | 08:33 | 09:02 | 09:22 | 10 h 18 m (6 h 10 m asleep) |
| A2 | 2026.11.12 | 09:31 (est.) | 10:07 | 10:14 | 10:35 | 1 h 04 m |
| B1 | 2026.11.13 | 09:03 | 11:50 | 13:26 | 14:15 | 5 h 12 m |
| A3-site | 2026.11.14 | 15:17 | 15:29 | 15:37 | 15:56 | 39 m |
| A3 | 2026.11.15 | 11:28 | 15:09 | 16:12 | 16:53 | 5 h 25 m |
| B2 | 2026.11.16 | 11:11 | 17:13 | 11-13 06:45 | 11-13 07:34 | 20 h 23 m (about 8 h night) |
| C1 | 2026.11.17 | 11-13 07:10 | 07:56 | 08:00 | 08:34 | 1 h 24 m |
| C2 | 2026.11.18 | 11-13 08:33 | 09:16 | 09:26 | 10:02 | 1 h 29 m |

Where the 35 hours went (22:58 on 11-11 to 10:02 on 11-13):

- **Nights, about 14 h.** Host asleep 01:20–07:30 on 11-12 (6 h 10 m), with swarm A and the manager stopped. Night two, about 8 h from 22:40 to 06:45, with B2's fixes done and its UAT pass waiting on Matt. The first manager and the product lead exited in that idle stretch; B2 sat on `dev` with no owner 06:45–07:10.
- **Waits on Matt in the day, on the critical path, about 4.5 h (est.).** B2's 8 UAT calls went out at 18:08 and were answered at 21:30 (3 h 20 m); the B2 layout call took 1 h 14 m; B1's UAT took about 50 min, off the B2 path.
- **Build, about 16 h of the 21 awake hours (est.).** B2's build ran 11:11–22:40, including about 2.5 h held for A3 on `dev` (lane G) and 1 h 15 m on the layout conflict.
- **CI.** About 5.5 min a PR (pytest 5–11 min, web 2–3 min); no failed run blocked a lane. Release sessions waited 8–28 min for CI per release.
- **Vercel limit, about 30 min on the critical path.** B1's production deploy was blocked 13:41–14:08 and went out by CLI. A3's `main` merge made no deployment (no `web/` change mattered). B1's and B2's UATs moved to lane servers.

The release and go-live sessions were short: 10–47 min each, 3–9 min of it active (est.).

## Load

| | Mosaic (35 h) | Second station #294 (24 h) |
|---|---|---|
| PRs merged | 102 (57 lane → goal, 33 → `dev`, 9 → `main`) | 37 |
| CI runs | 314, all `pull_request` | 107 |
| CI wall minutes | about 1,569 (tests 1,140, web 429) | about 232 |
| CI failures / re-runs | 2 / 0 | 0 / 0 |
| Vercel deployments | 140 | not counted for the same window |

CI by branch: lane 138 runs (668 min), goal 102 (515), docs 22, `dev` 18, chore 18, fix 10, feat 6. The minutes per run doubled from the baseline (5.0 against 2.2), mostly the tests workflow at about 7.3 min a run.

Vercel deployments in the window (11-12 04:00Z to 11-13 19:00Z), by state: 77 ready, 40 cancelled, 23 error. By branch: lane 36 (all cancelled by the Ignored Build Step, and all counted), goal 33, shots 27 (22 of the 23 errors, `NOW_SANDBOX_WORKER_ROOTDIR_NOT_EXIST`), `dev` 25, `main` 7, docs 5, chore 3, `uat` 2, CLI 1. The 100th deployment in 24 hours came at 12:08 PT on 11-12; agents saw "Deployment rate limited — retry in 24 hours" from 12:20.

| | Deployments | Hours | Rate |
|---|---|---|---|
| Before #565 | 112 | 18.5 | about 145 a day |
| After #573 | 27 | 20.3 | about 32 a day |

Under ADR-0010's rule the window's deployments would have been 34 (est.), about 20 a day. That saves about 105 deployments over the window (est.), and on the second-station days 11-03 and 11-04 (97 and 99 deployments) it would have given about 14 a day.

## Models

Every main loop ran on Opus 5.5. Subagents ran on Opus 5.5 unless the brief named another model.

| Role | Sessions | Main turns (Opus) | Subagent turns: Opus / Sonnet / Haiku | Output tokens |
|---|---|---|---|---|
| Swarm orchestrators and lanes | 6 | 905 | 4,087 / 1,569 / 2 | 3.35 M |
| Release managers | 2 | 490 | 0 / 17 / 0 | 326 k |
| Panels | 9 | 274 | 0 / 269 / 0 | 279 k |
| Product leads and close-out | 3 | 244 | 9 / 41 / 2 | 164 k |
| C builds, fixes, deploy rule, live read | 5 | 274 | 23 / 93 / 0 | 157 k |
| Release and go-live | 12 | 431 | none | 128 k |
| UAT | 3 | 115 | 0 / 109 / 0 | 106 k |
| Orchestrator | 1 | 127 | 0 / 11 / 0 | 72 k |
| **All** | **41** | **2,860** | **4,119 / 2,109 / 4** | **4.58 M** |

Opus wrote 80% of the output tokens (3.66 M of 4.58 M). Opus lanes alone wrote 2.25 M, half of the program. Cache reads were 1.13 B tokens on Opus and 142 M on Sonnet. Dollar cost was not computed.

What the transcripts show, role by role:

- **Swarm lanes.** Opus ran most implementers: 9 of 11 in A, 17 of 22 in B2. Sonnet lanes passed review in every swarm (A's D and H, seven B1 lanes, A3's B and D, B2's M and #543). One Sonnet lane failed: B2 lane N shipped the wrong "now" label, a panel blocker that cost a second link (#602) and a delta panel. Opus did mechanical lanes: merge-forwards (A lane I, A3 #567 at 101 turns, B2 lane H at 86 turns), list-driven cleanups (B2 J, K) and screenshot sets (B1 99 turns, B2 598 views). Keep Opus for first-of-kind and design lanes; default the rest to Sonnet.
- **Release and go-live sessions.** About 90% mechanical: bump `VERSION`, open the PR, watch CI, curl production. The judgement moments were few and small (stop on the Vercel limit, read a cache as by design). Merges stayed with the manager. Sonnet would do; the go-live check fits inside the release session as a monitor, which saves 3 cold starts.
- **Panels and UAT.** The main loops dispatched Sonnet seats and collated. Opus earned its place on verdicts and the judgement calls drafted for Matt. Panels C1 and C2 ran no subagents and found nothing to change; panels R1 and A2 found notes only.
- **Release managers and the orchestrator.** About 150 `gh` ledger edits and status relays were Opus work a cheaper model could do, but the judgement calls (merge pinned to the tested head, hold A3 for #578, stop an empty brief, route scope to the product lead) were real. Keep Opus here.
- **No cheaper-model failure outside lane N.** Haiku ran 4 turns in the whole program.

## Time to market: counterfactuals

| Change | What it saves | Cost or risk |
|---|---|---|
| Keep the host awake on night one (`caffeinate`) | A live by about 04:00; B1 and B2 start about 5–7 h sooner; B2's UAT lands in the afternoon; finish about 11-12 19:00, about 15 h sooner (est.) | None |
| Ask the long pole's UAT before 15:00, or batch B1 and B2 into one daytime UAT | Avoids the night: 8 of B2's 20 h | B1 ships later |
| Trace every site number to a Python field in the scope read | A2, A3-site and parts of A3: 3 releases, 6 PRs and their CI; lane G's 2.5 h hold (est.) | One longer scope read |
| Put C1 (site reads schema 3 and 4) into B2 | One release on the second morning, about 1.4 h (est.) | Bigger B2 diff |
| Merge B1 into B2 | The 1 h 15 m stack conflict, lane H, one panel, one UAT, one release | No early hero in production; a bigger UAT |
| Run B1 and B2 in parallel | Already done (B2 started 11:11, B1 still building) | — |
| More lanes in A or A3 | Little: lanes there ran serially because every one regenerates the same goldens | Golden conflicts |
| UAT on a lane server instead of previews | Already forced by the limit; ADR-0010's `uat/<slug>` makes it the default | — |
| Smaller first slice | Release 1 was already 68 minutes to production | — |

**Recommended shape for the next program this size.** Three releases and a daytime human gate. First a Python release that carries every number the site stories print (the scope read traces each one). Then one site release that includes the schema read-side step, built in parallel lanes and checked on `uat/<slug>`, with its single judgement-only UAT asked by early afternoon. Then the Python schema step. Start in the evening, keep the host awake, and let the build run overnight so the UAT is ready when Matt wakes. Lanes default to Sonnet, with Opus for the first lane of a new surface and for design lanes; release and go-live run as one Sonnet session per release, and merges stay with the Opus manager.

## Lessons

Validated and new lessons are in `lessons.md`, 2026-11-13. The [skill] lessons are rolled up on stonematt/stonematt-skills#137.
