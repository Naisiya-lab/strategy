#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-$HOME/.conda/envs/mmtab-table-llava/bin/python}"
REPO="$("$PYTHON" -c 'import json; from pathlib import Path; print((Path.cwd()/json.load(open("config.json"))["official_repo"]).resolve())')"
test -f "$REPO/llava/model/builder.py" || { echo "Official Table-LLaVA source not found: $REPO"; exit 1; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if [ "$#" -eq 0 ]; then set -- smoke; fi
exec "$PYTHON" evaluate.py "$@"
