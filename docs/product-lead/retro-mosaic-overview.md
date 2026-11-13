# Mosaic Overview program retro

Deep retro over spec #460 and program #484, from the first dispatch (Wed 2026-11-11 22:58 PT) to the close (Fri 2026-11-13 10:22 PT). Sources: 41 session transcripts, their 219 subagent transcripts, #460, #426, #484, ledgers #485–#489, 102 PRs, 314 CI runs and 140 Vercel deployments. Times are Pacific. A number marked (est.) is estimated; every other number was counted. The close-out lessons (PR #626) are validated in `lessons.md`, 2026-11-13. The page version, with the charts: https://claude.ai/artifact/Hb69rAaMWaRvMA9KfmpetX

## Did this take too long, and why

A little, and the cause was one sleeping laptop. The Mac slept for six hours on night one with Release A half built. Every site release waited on A, so B2's UAT reached Matt at 6:08 pm and sat through a second night.

The scope was about three times the second-station program (#294): 102 PRs against 37, 9 releases against 3. It ran 35 hours against 24, so the build itself moved faster. Nights took 14 of those hours. With the Mac awake on night one, B2's UAT lands Thursday afternoon and the program finishes Thursday evening, about 15 hours sooner (est.). The rest of the delay was shape: three extra Python releases for numbers the site printed without a Python field, a 75-minute conflict from stacking B2 on B1, and C1 shipping alone on Friday morning. The Vercel limit cost about 30 minutes on the critical path.

## What we do next time

Three releases and one daytime gate. Python first, carrying every number the site prints. Then one site release with the schema read step inside it, checked on `uat/<slug>`, its UAT in front of Matt by early afternoon. Then the Python schema step.

Start in the evening with the Mac held awake, so the build runs overnight and the UAT is waiting in the morning. Matt talks to one manager, and the manager can be relaunched from the ledger. Lanes run on Sonnet unless the work is new ground or design. Each release gets one Sonnet session, and merges stay with the Opus manager.

## Timeline

B2 was the long pole: 20 h 23 m from dispatch to production, and about 8 h of that was a night waiting for its UAT pass.

| Release | Version | Build dispatched | Goal PR ready | Merged to `dev` | Production verified | Wall time |
|---|---|---|---|---|---|---|
| 1 | 2026.11.10 | Wed 23:04 | 23:30 | Thu 00:02 | 00:12 | 1 h 08 m |
| A | 2026.11.11 | Wed 23:04 | Thu 08:33 | 09:02 | 09:22 | 10 h 18 m (6 h 10 m asleep) |
| A2 | 2026.11.12 | 09:31 (est.) | 10:07 | 10:14 | 10:35 | 1 h 04 m |
| B1 | 2026.11.13 | 09:03 | 11:50 | 13:26 | 14:15 | 5 h 12 m |
| A3-site | 2026.11.14 | 15:17 | 15:29 | 15:37 | 15:56 | 39 m |
| A3 | 2026.11.15 | 11:28 | 15:09 | 16:12 | 16:53 | 5 h 25 m |
| B2 | 2026.11.16 | 11:11 | 17:13 | Fri 06:45 | Fri 07:34 | 20 h 23 m (about 8 h night) |
| C1 | 2026.11.17 | Fri 07:10 | 07:56 | 08:00 | 08:34 | 1 h 24 m |
| C2 | 2026.11.18 | Fri 08:33 | 09:16 | 09:26 | 10:02 | 1 h 29 m |

Where the 35 hours went:

- **Nights, about 14 h.** The Mac slept 01:20–07:30 Thursday, stopping swarm A and the manager. On night two B2's fixes were done and its pass waited on Matt. The manager and product lead exited in that stretch, and B2 sat on `dev` with no owner from 06:45 to 07:10.
- **Build, about 16 h of 21 awake hours (est.).** B2 built from 11:11 to 22:40. That included about 2.5 h waiting for A3 on `dev` and 1 h 15 m on the layout conflict.
- **Daytime waits on Matt on the critical path, about 4.5 h (est.).** B2's eight UAT calls (18:08–21:30) and the B2 layout call (74 m). B1's 50-minute UAT sat off that path.
- **CI.** Release sessions waited 8–28 m a release. Lanes waited about 5.5 m per PR, and no failed run blocked a lane.
- **Vercel limit, about 30 m on the critical path.** B1's production deploy was blocked 13:41–14:08 and went out by CLI. B1 and B2 UAT moved to lane servers.

## The shape of delegation

One manager carried the program. The two manager sessions dispatched 36 of the 40 sessions and sat on 261 of the 275 messages between sessions. The orchestrator was Matt's window into it, and Matt took his decisions straight to the session doing the work.

Method: dispatch edges come from the `claude --bg -n <name>` commands in each transcript, message edges from SendMessage calls and cross-session receipts, and Matt's turns from the typed human turns in each transcript. The handover from manager 1 to manager 2 is inferred: no message passed between them.

**Where the work and the cost sat.** Output tokens by branch of the dispatch tree. The page draws this as an icicle, coloured by model, with tool calls on hover.

| Branch | Output tokens | Share |
|---|---|---|
| Orchestrator and everything below it | 4.58 M | 100% |
| Manager 1 and the 27 sessions it dispatched | 4.07 M | 89% |
| Swarm B2 (main loop and 82 subagents) | 1.78 M | 39% |
| Swarm B1 | 578 k | 13% |
| Swarm A | 496 k | 11% |
| Swarm A3 | 341 k | 7% |
| Manager 2 and the 9 sessions it dispatched | 289 k | 6% |
| Product lead 1 | 123 k | 3% |
| Orchestrator's own loop | 72 k | 2% |

**Who asked whom.** Messages between sessions. The page draws this as a network with dispatches, weighted messages and Matt's typed turns.

| From → to | Messages |
|---|---|
| Product lead 1 → manager 1 | 30 |
| Manager 1 → product lead 1 | 26 |
| Manager 1 → swarm B2 | 24 |
| Swarm B2 → manager 1 | 21 |
| Orchestrator → manager 1 | 11 |
| Manager 1 → swarm A3 / swarm A3 → manager 1 | 9 / 8 |
| Manager 1 → swarm A / swarm A → manager 1 | 7 / 9 |
| Manager 1 → swarm B1 / swarm B1 → manager 1 | 7 / 6 |
| Manager 1 and manager 2 → orchestrator | 5 + 6 |
| Manager 1 → manager 2 | 0 (handover through #484, inferred) |

Manager 1 dispatched 27 sessions and manager 2 dispatched 9. The orchestrator dispatched 4 in the program: product lead 1 and manager 1 at the start, the deploy-rule session, and manager 2.

Matt typed 57 turns into 11 sessions: the orchestrator 23 (8 of them status checks), UAT R1 12, product lead 1 5, UAT B2 5, each manager 3, swarm A 2, and one each into swarm A3, UAT B1, panel A3 and the deploy-rule session.

**Who was working when.** At most six sessions were active at once, and about two on average through the awake hours. Most sessions were short release, panel and go-live sessions running in a line, one after another.

**Was the orchestration shape right?** Right at the centre, thin at the edges. The manager as a hub worked: it held the plan, the merges and 95% of the traffic, and the product lead's scope calls reached it in 56 messages that mattered (A2, A3, the footer). The orchestrator added a layer without adding decisions. It dispatched four sessions and carried 33 messages. Eight of Matt's 23 turns there were status checks, and its real work was the deploy rule and the overnight relaunch. Matt's decisions went where the work sat: UAT passes in UAT sessions, scope with the product lead, the merge grant with the manager. That kept the merge path to two hops, and it is the right instinct. The hub was also a single point of failure: when manager 1 went idle overnight, nothing moved until the orchestrator relaunched it. The twelve release and go-live sessions mostly relayed CI and curl results back to the manager. Fold the orchestrator into the manager as Matt's one window, make the manager restartable from the ledger by design, and give each release one Sonnet session.

## Load

Three times the PRs and CI runs of #294, and almost seven times the CI minutes, because each run took twice as long.

| | Mosaic (35 h) | Second station #294 (24 h) |
|---|---|---|
| PRs merged | 102 (57 lane → goal, 33 → `dev`, 9 → `main`) | 37 |
| CI runs | 314, all `pull_request` | 107 |
| CI wall minutes | about 1,569 (tests 1,140, web 429) | about 232 |
| Minutes per run | 5.0 | 2.2 |
| CI failures / re-runs | 2 / 0 | 0 / 0 |

CI by branch: lane 138 runs (668 min), goal 102 (515), docs 22, `dev` 18, chore 18, fix 10, feat 6.

Skipped builds still counted toward Vercel's daily 100. In the window (Thu 04:00Z to Fri 19:00Z) there were 140 deployments: 77 ready, 40 cancelled, 23 error. By branch: lane 36 (all cancelled by Vercel's skip step, all counted), goal 33, shots 27 (22 of the 23 errors, `NOW_SANDBOX_WORKER_ROOTDIR_NOT_EXIST`), `dev` 25, `main` 7, docs 5, chore 3, `uat` 2, CLI 1. The 100th deployment in 24 hours came at 12:08 on Thursday, and agents saw "Deployment rate limited — retry in 24 hours" from 12:20.

| | Deployments | Hours | A day |
|---|---|---|---|
| Before #565 | 112 | 18.5 | about 145 |
| After #573 | 27 | 20.3 | about 32 |
| ADR-0010 rule over the window (est.) | 34 | 39.0 | about 20 |

ADR-0010 saves about 105 deployments over the window (est.). On the second-station days 11-03 and 11-04 (97 and 99 deployments) the same rule would have given about 14 a day.

## Models

Opus wrote 80% of the output tokens, and Opus lanes alone wrote half of the program's 4.58 M. Sonnet lanes did the same work at the same quality on precise tickets. Every main loop ran on Opus 5.5. Subagents ran on Opus unless the brief named another model.

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

Cache reads were 1.13 B tokens on Opus and 142 M on Sonnet. Dollar cost was not computed.

- **Lanes.** Sonnet lanes passed review in every swarm. One failed: B2 lane N shipped the wrong "now" label (#602). Opus also ran the merge-forwards, cleanup lists and screenshot sets, which is checklist work. Put lanes on Sonnet by default and keep Opus for the first lane on new ground, design lanes and the whole-run review.
- **Release and go-live.** About 90% of it was bump, open, watch CI, curl. Sonnet handles that, with the go-live check running as a monitor inside the release session.
- **Panels and UAT.** The Opus main loops sent out Sonnet reviewers and gathered what came back. Opus earned its place on the verdicts and on the calls drafted for Matt. Panels C1 and C2 found nothing to change.
- **Managers and orchestrator.** Keep Opus. The ledger edits were routine, but merging only the tested head, holding A3 for #578 and stopping a session with an empty brief were real judgement.

## The plan against what happened

Five planned releases became nine. C's split was in the plan from the start, because ADR-0008 ships a schema change in two steps. A2, A3 and A3-site came from numbers the site printed that no Python ticket decided.

Spec #460 (opened 11-11 17:21) has no edit history, so its first version is the current body plus dated comments. Program #484's first body (11-11 23:00) had the same five rows, A as one serial lane, and B2 stacked on B1's goal branch. Matt's grant at 23:07–23:13: merge on green to `dev` and `main`, UAT only for 1, B1 and B2, A and both C steps checked in production, stop and ask before any raw archive write.

| Planned | Scope at the start | Shipped as |
|---|---|---|
| 1 | Web only, #458 | 2026.11.10, Thu 00:12 |
| A | Python fields, #461–#467 and #491 | A 2026.11.11 (09:22), A2 2026.11.12 (10:35), A3 2026.11.15 (16:53) |
| B1 | Hero swap, feature tile, next-unlock and battery lines (#468–#471) | 2026.11.13 (14:15), then A3-site 2026.11.14 (15:56) |
| B2 | Mosaic, zooms, Inside tile, wind rose, design docs (#472–#478) | 2026.11.16, Fri 07:34 |
| C | Stop writing `hero_lead`, two steps (#479, #480) | C1 2026.11.17 (08:34), C2 2026.11.18 (10:02) |

| When | Change | Why, and who decided |
|---|---|---|
| Thu 00:04 | UAT becomes judgement-only | Matt: "This UAT was too tactical and didn't really require my judgement." |
| 09:30 | A2 added (#518) | B1 lane A found no snapshot value for the hero delta; product lead |
| 09:45–11:27 | A3: 3-hour feature hold, gust out (#533) | A 60-day replay; Matt: "file both, 3 hour hold and drop gust" |
| 11:36 | Sources footer and version move to B2 only | Matt |
| 11:51 | A3 grows #539, #540, #554 | B2 tile numbers had no Python field; manager proposed, product lead filed |
| 12:08 | Vercel's 100th deployment in 24 h | Three swarms in parallel; found at 12:26 |
| 14:30 | Deploy rule 1, #565 | Skipped builds still count; Matt asked |
| 14:41 | Deploy rule 2, #573 and ADR-0010 | Only `main`, `dev` and `uat/**` deploy; Matt: "uat per release, go ahead and build it" |
| 14:41 | Desktop feature becomes a full-width row | The B1–B2 stack conflict exposed it; Matt, asked at 13:27 |
| 15:17 | A3-site release (#578) | A3 panel blocker on the held feature's wording; manager |
| 21:30 | Sources footer cut before merge (#604) | Matt: "souces and methods are 3x the body of the page. ridiculous." |
| Fri 07:08 | Release manager relaunched | The first manager and product lead exited overnight; orchestrator |
| Fri 10:22 | #584, #597 deferred (#608 at B2 UAT) | The site is correct without them; Matt: "yes, defer them" |

## What else could have happened

Keeping the Mac awake is worth about as much as the rest of this list combined (est.).

| Change | What it saves | Cost or risk |
|---|---|---|
| Keep the Mac awake on night one | A live about 04:00; B starts 5–7 h sooner; B2's UAT lands in the afternoon; finish Thursday about 19:00, about 15 h sooner (est.) | None |
| Ask the long pole's UAT before 15:00, or batch B1 and B2 | Avoids the night: 8 of B2's 20 h | B1 ships later |
| Trace every site number to a Python field in the scope read | A2, A3-site and parts of A3; lane G's 2.5 h hold (est.) | A longer scope read |
| Put C1 into B2 | One release on Friday morning, about 1.4 h (est.) | A bigger B2 diff |
| Merge B1 into B2 | The 75 m stack conflict, a merge-forward lane, one panel, one UAT, one release | No early hero in production; a bigger UAT |
| More lanes in A or A3 | Little: every lane regenerates the same goldens, so they ran one at a time | Golden conflicts |
| B1 and B2 in parallel; lane server UAT; a smaller first slice | Already done: B2 started while B1 built, the limit forced lane servers, and Release 1 took 68 m | — |

## Lessons

All eleven close-out lessons held up against the transcripts, and four are sharper now. Ten are new: eight for skills, two for the project. They are in `lessons.md`, 2026-11-13, and the [skill] lessons are rolled up on stonematt/stonematt-skills#137.
