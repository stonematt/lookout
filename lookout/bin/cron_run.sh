#!/usr/bin/env bash

set -euo pipefail

print_help() {
  cat <<EOF
Usage: $(basename "$0") [--sleep MAX_SECONDS] <script>

Run a Python script from the current git repo root with virtualenv activation and logging.

Arguments:
  <script>        Path to the Python script relative to repo root (e.g., scripts/uptime_check.py)

Options:
  --sleep N       Sleep for a random delay up to N seconds before execution (default: 900)

Environment:
  Assumes virtualenv at: <repo_root>/.venv
  Logs status to: logs/cron_<script>.status (slashes converted to underscores)

Setup:
  git clone --single-branch --depth 1 --branch main git@github.com:stonematt/lookout.git
  cd lookout
  uv venv --python 3.12 .venv && uv pip install -r requirements.txt

Crontab Example (runs every 8 hours with jitter):
  0 */8 * * * /path/to/cron_run.sh --sleep 900 scripts/uptime_check.py

EOF
}

# Defaults
SLEEP_MAX=900

# Parse args
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    print_help
    exit 0
    ;;
  --sleep)
    SLEEP_MAX="$2"
    shift 2
    ;;
  -*)
    echo "Unknown option $1"
    print_help
    exit 1
    ;;
  *)
    POSITIONAL+=("$1")
    shift
    ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ne 1 ]]; then
  echo "Error: Missing <script> argument"
  print_help
  exit 1
fi

SCRIPT_REL="${POSITIONAL[0]}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Random sleep (jitter)
if [[ "$SLEEP_MAX" -gt 0 ]]; then
  SLEEP_TIME=$((RANDOM % SLEEP_MAX))
  echo "Sleeping $SLEEP_TIME seconds..."
  sleep "$SLEEP_TIME"
fi

SCRIPT_PATH="$REPO_ROOT/$SCRIPT_REL"
SCRIPT_KEY="${SCRIPT_REL//\//_}"
LOG_FILE="$REPO_ROOT/logs/cron_${SCRIPT_KEY%.*}.status"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

# Ensure logs directory exists
mkdir -p "$REPO_ROOT/logs"

# Git pull: a failed pull fails the run before python starts, so a cron
# never runs stale code unnoticed. No python means no Healthchecks start
# ping, and the check goes late.
if ! git pull --rebase; then
  echo "$TIMESTAMP FAIL git pull --rebase failed" >>"$LOG_FILE"
  echo "Error: git pull --rebase failed; not running $SCRIPT_REL" >&2
  exit 1
fi

# Activate virtualenv
source "$REPO_ROOT/.venv/bin/activate"

# Run the script and exit with its status. The `if` keeps set -e from
# aborting before the status is logged.
STATUS=0
PYTHONPATH=. python "$SCRIPT_PATH" || STATUS=$?

if [[ "$STATUS" -eq 0 ]]; then
  echo "$TIMESTAMP SUCCESS" >>"$LOG_FILE"
else
  echo "$TIMESTAMP FAIL exit $STATUS" >>"$LOG_FILE"
fi
exit "$STATUS"
