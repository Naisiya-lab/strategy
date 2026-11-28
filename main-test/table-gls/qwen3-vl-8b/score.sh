#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${SCORE_PYTHON:-$HOME/.conda/envs/mmtab-score/bin/python}" ../score.py --run "$PWD/outputs/full" "$@"
