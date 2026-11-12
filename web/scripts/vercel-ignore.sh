#!/bin/sh
# Vercel ignoreCommand: exit 0 skips the build, exit 1 builds it.
#
# This is the second gate. The first is `git.deploymentEnabled` in vercel.json,
# which stops Vercel from creating a deployment at all for dev, feat/**, fix/**,
# docs/**, chore/**, lane/**, shots/** and prototype/**. A deployment this
# script skips is CANCELED, and it still counts toward the Hobby budget of about
# 100 deployments in a rolling 24 hours. So this script only decides the
# branches that get a deployment, and it skips a goal branch push that did not
# change the site.
#
# Vercel runs it from the Root Directory (web/), so `.` below is web/.
case "${VERCEL_GIT_COMMIT_REF:-}" in
  # Rule 1: main is production. It always builds.
  main) exit 1 ;;

  # Rule 2: preview/* is the opt-in for a branch someone wants to look at.
  preview/*) exit 1 ;;

  # Rule 3: a goal branch is the UAT preview. It builds only when the push
  # changed web/ since the branch's last successful deployment.
  goal/*)
    prev="${VERCEL_GIT_PREVIOUS_SHA:-}"
    # A first deployment on the branch has no previous SHA: build.
    [ -n "$prev" ] || exit 1
    # Vercel clones shallow (depth 10), so the previous SHA can be missing from
    # the clone: build.
    git cat-file -e "$prev^{commit}" 2>/dev/null || exit 1
    # git diff --quiet exits 0 for no change (skip), 1 for a change (build),
    # and above 1 on an error, which also builds.
    if git diff --quiet "$prev" HEAD -- .; then exit 0; fi
    exit 1
    ;;

  # Rule 4: any other branch skips. vercel.json already stops most of them
  # from deploying; this catches a branch prefix it does not name.
  *) exit 0 ;;
esac
