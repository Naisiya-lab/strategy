#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "$#" -eq 0 ]; then set -- smoke; fi
exec conda run --no-capture-output -n refocus python -u evaluate.py "$@"
