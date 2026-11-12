#!/bin/sh
# Vercel ignoreCommand: exit 0 skips the build, exit 1 builds it.
#
# The allowlist lives in `git.deploymentEnabled` in vercel.json: a catch-all
# `**: false`, and true for main, dev and uat/**. Vercel creates no deployment
# for any other branch, so this script rarely decides anything. It repeats the
# same allowlist as a backstop: a deployment it skips is CANCELED, and that
# still counts toward the Hobby budget of about 100 in a rolling 24 hours.
case "${VERCEL_GIT_COMMIT_REF:-}" in
  # main is production.
  main) exit 1 ;;

  # dev is the integration branch.
  dev) exit 1 ;;

  # uat/<slug> is a release's UAT preview, pushed from its goal branch tip.
  uat/*) exit 1 ;;

  # Every other branch skips.
  *) exit 0 ;;
esac
