#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-$HOME/.conda/envs/mmtab-qwen/bin/python}"
MODE="${1:-smoke}"
if [ "$#" -gt 0 ]; then shift; fi
exec "$PYTHON" evaluate.py "$MODE" "$@"
