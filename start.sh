#!/usr/bin/env bash
# QuantLab: one-command start for Mac/Linux.   ./start.sh   (extra options go to the app, e.g. --port 8050)
# First run: creates a private Python environment in .venv and installs the packages.
# Later runs: packages are only reinstalled when requirements.txt has changed.
set -e
cd "$(dirname "$0")"

PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: creating a private Python environment in .venv ..."
  "$PY" -m venv .venv
fi

if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing packages (first run, or after an update). This takes a minute ..."
  .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
  cp requirements.txt .venv/requirements.installed
fi

exec .venv/bin/python -m quantlab.app "$@"
