#!/usr/bin/env bash
# QuantLab: get the newest version from GitHub, then start the app.
set -e
cd "$(dirname "$0")"
echo "Getting the newest version ..."
if ! git pull --ff-only; then
  echo
  echo "Update failed. If it says your local changes would be overwritten, you edited a file that"
  echo "was also updated: run 'git status' to see which, or ask for help. The app was not started."
  exit 1
fi
exec ./start.sh "$@"
