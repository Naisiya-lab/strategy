#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${GLM_PYTHON:-python3}" -u evaluate.py "$@"
