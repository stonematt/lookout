# Releases carry a calendar version and client-facing APIs carry a version in the path

## Status

accepted (2026-11-02). Decided for the release "ship dev to main and
introduce versioning" (#277). Leaves the snapshot contract's
`SCHEMA_VERSION` as it is.

## Context

Lookout has no release number. Nothing tells which code the refresh run on
`itri` and the site on Vercel are running, and no tag marks a release on
`main`. Both ship together from one `main` merge, so they can share one
number.

Matt plans an iOS app and a desktop widget. Each will read an API contract
that must stay stable while the site and the refresh run change. A release
number cannot carry that promise: releases happen often, and most of them
do not touch any contract.

## Decision

There are two kinds of version number.

**Release version.** The repo has one calendar version, `YYYY.MM.N`. N
counts the releases in that month from 1, and the month has no leading
zero. The first release is `2026.11.1`. A root `VERSION` file holds the
bare number, and `lookout.__version__` reads it. A test checks its format.

Each release cuts the number the same way:

1. A `chore(release): <number>` PR into `dev` bumps `VERSION`. CI tests it.
2. The `dev` → `main` release PR head equals that tested tree.
3. After production is verified, the `main` merge commit gets an annotated
   tag `v<number>`, and the tag is pushed.

**API contract version.** Each client-facing API carries an integer major
version in its path: `/api/v1/…`. An additive change stays in the same
version. A breaking change adds `v2`, and `v1` keeps answering until its
clients move. `SCHEMA_VERSION` is the existing contract of this kind,
between the refresh run and the site, and keeps its own rules.

No versioned API exists yet. The first client-facing API, for the iOS app
or the widget, starts at `v1`. The existing `/api/*` routes are read only
by the site, so they are not renamed now.

## Considered options

- **One SemVer for everything, from `v0.1.0`.** Rejected: one number
  cannot say both "a new release shipped" and "this contract broke". The
  app and the widget need the second promise for each API, not for the
  repo.

## Consequences

The site does not show the version in this release. The Vercel build has
Root Directory `web` and does not read `../VERSION`. The refresh
provenance work (#268) can add the version to `snapshot/<MAC>.context.json`
later, and a site footer can follow from there.

Amended 2026-11-12 (#536): the site now shows its own release version.
`web/vite.config.ts` reads `../VERSION` at build time (`releaseDefine` in
`web/server/release.ts`) and bakes it into the page with the short
`VERCEL_GIT_COMMIT_SHA` ("local" outside a Vercel build). A missing or
malformed `VERSION` fails the build. This works because a Vercel build
with a Root Directory can read files outside it, unless the project turns
off "Include files outside the root directory in the Build Step". The page
reads the values from `web/src/release.ts`. debug.html sets them beside
the refresh run's version and SHA from `/status`, and says when they
differ. `/api/v1/version` is the first versioned API: it answers
`{version, sha}` from the same `VERSION` file, which Vercel's function
trace ships with the function, and from the SHA Vercel sets at run time.
A `VERSION` bump changes all three with no other edit.

Every release now has one more step before the release PR, and one after
production is verified.
