#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-$HOME/.conda/envs/mmtab-syntab/bin/python}"
export PYTHONPATH="$PWD/vendor/SynTab-LLaVA${PYTHONPATH:+:$PYTHONPATH}"
if [ "$#" -eq 0 ]; then set -- smoke; fi
exec "$PYTHON" evaluate.py "$@"
