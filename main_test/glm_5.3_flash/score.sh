#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${SCORE_PYTHON:-$HOME/.conda/envs/mmtab-score/bin/python}"
exec "$PYTHON" score.py "$@"
