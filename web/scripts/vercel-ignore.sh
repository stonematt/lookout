#!/bin/sh
# Vercel ignoreCommand: exit 0 skips the build, exit 1 builds it.
# Screenshot branches (shots/*) and swarm lane branches (lane/*) build no
# preview, so a swarm run and its UAT stay inside the Hobby deployment budget.
# Every other branch builds, including main, dev, goal/**, feat/*, fix/*, docs/*.
case "${VERCEL_GIT_COMMIT_REF:-}" in
  shots/* | lane/*) exit 0 ;;
  *) exit 1 ;;
esac
