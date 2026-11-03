# Autonomy grant: the second-station program

Matt's standing policy for the program that serves a second station (#179, #234), tracked in the program ledger #294. It holds from 2026-11-03 until #294 closes. Then this file is deleted or replaced.

Matt's words (2026-11-03): "My goal is to really give you permission to make decisions and release. This is a low stake site and you can touch the VPS and release when PRs are green and just kind of go full speed ahead with almost no human intervention here."

## Who it covers

The release manager session named in #294, the release and ops sessions it dispatches with `claude --bg`, and the swarms it launches for Releases A–D. Every action stays inside the rules in `CLAUDE.md` and `docs/agents/operations.md`; this file adds permissions, it removes no rule.

## Allowed without asking

- **Merge on green.** Merge any PR into `dev`, a `goal/` branch or `main` when its CI is green and its review verdict is ship or fix-applied. Merge from a main session, pinned with `gh pr merge <n> --merge --match-head-commit <sha>`. Agent-tool subagents never merge (`.claude/swarm-profile.md`, landing policy).
- **Release.** Cut the `chore(release)` PR, open and merge the `dev` → `main` release PR, and tag `v<version>` after production is verified (ADR-0007). The release head must equal a CI-tested tree.
- **itri.** ssh with the agent form in `docs/agents/operations.md`, to read logs and state, to check the deploy, to change the crontab (save `~/crontab.bak-pre-<N>` first), and to run `stations scan` and `stations add` for Lookout Post and Salmon Run.
- **Vercel.** Change the production env (remove `STATION_MAC`, swap in the read-only Storj credential of #247), redeploy, and roll back to the last good deployment.
- **Tickets.** Comment on and relabel the program's tickets, and record Matt's decisions on them. A new ticket found mid-run goes into the program ledger as a candidate; Matt triages it, because `triage` and `to-tickets` run only when he starts them.

## Stop and ask Matt

- A live archive write other than `stations add`: catchup by hand, `fill-gap`, `repair_rain --write`, a catalog `--save`.
- Any `ArchiveLossError`, or any step that would delete or overwrite stored data on Storj.
- CI still red after two fix attempts on one PR.
- Production answering 5xx after a rollback, or a refresh run failing twice in a row after a deploy.
- A change of plan: a ticket in or out of a release, a release reordered.
- A gate that denies an action: stop that action, record it in #294 as a `denial`, and add the exact `!` command to its Waits on Matt list. Never route around a denial.

## Also

- The BSE monitor does not run on lookout while #294 is open. Several `bse` tickets touch the same files as the program's lanes.
- Every event worth counting goes into the Telemetry table in #294: dispatches, merges, red CI runs, denials, Matt's touches, deploys, verifies, rollbacks and incidents.
