#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-/nc1test1/ENTER/envs/tabflash/bin/python}"
if [ "$#" -eq 0 ]; then set -- smoke; fi
exec "$PYTHON" -u evaluate.py "$@"
