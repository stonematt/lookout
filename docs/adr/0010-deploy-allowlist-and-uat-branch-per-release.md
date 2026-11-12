# Only main, dev and uat/** deploy, and each release gets a uat/<slug> branch ahead of dev

## Status

accepted (2026-11-12). Decided with Matt while the Mosaic Overview releases
were in flight. Built by #565 (the first cut) and #573 (the allowlist).
Replaces the skip list of #404.

## Context

The site deploys on Vercel's Hobby plan, which allows about 100
deployments in a rolling 24 hours. When the budget runs out, no branch
deploys, `main` included, until older deployments age out of the window.

Vercel deploys every pushed branch unless told otherwise. The first guard,
#404, used `ignoreCommand` to skip `lane/*` and `shots/*` builds. That
saved nothing: a skipped build still creates a deployment, CANCELED, and
Vercel counts it. In the 24 hours before #565 the project made 128
deployments. Of those, 36 were canceled `lane/*` builds and 18 were
`shots/*` builds, and all of them counted.

## Decision

1. **Deploys are an allowlist.** `git.deploymentEnabled` in
   `web/vercel.json` sets a catch-all `"**": false` and true only for
   `main` (production), `dev` (integration) and `uat/**`. Vercel then
   creates no deployment at all for any other branch, so nothing is
   counted. `web/scripts/vercel-ignore.sh` repeats the allowlist as a
   backstop.
2. **Each release gets its own `uat/<slug>` branch, ahead of `dev`.** Lanes
   merge into `goal/<slug>`, which does not deploy. When the release is
   ready for UAT, the goal tip is pushed as `uat/<slug>`, and Matt's UAT
   runs on that preview. Fixes land on the goal branch and are merged
   forward. On Matt's pass the goal PR merges into `dev`. The UAT branch is
   deleted after the release. The steps are in `.claude/swarm-profile.md`
   ("UAT branch per release").

## Considered options

- **A skip list** (#404, then #565: deny named prefixes, and skip goal
  pushes that left `web/` alone). Rejected: a skip by `ignoreCommand` still
  counts, and any branch prefix the list does not name deploys by default.
- **One fixed `uat` branch, reset to each release.** Rejected: moving it to
  a new release needs a force-push, and two releases in UAT at the same
  time would collide on it.

## Consequences

- When the budget runs out anyway, it frees up on its own as the rolling
  24 hours moves on; nothing needs to be undone.
- A new branch prefix does not deploy until someone adds it to the
  allowlist.
- A preview of anything other than `dev` or a release needs a `uat/`
  branch. Mechanical checks (the compliance pass, screenshots) use the
  lane server and need no deploy.
- Vercel runs the rule from the commit it builds, so a branch cut before
  the change keeps the old rule until `dev` is merged into it.
