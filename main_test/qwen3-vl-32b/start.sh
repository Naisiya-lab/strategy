#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-$HOME/.conda/envs/mmtab-qwen/bin/python}"
exec "$PYTHON" launch.py start "$@"
