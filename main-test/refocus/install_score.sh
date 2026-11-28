#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_PATH="${SCORE_ENV_PATH:-$HOME/.conda/envs/mmtab-score}"
if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.11 -y
fi
"$ENV_PATH/bin/python" -m pip install -r score_requirements.txt
"$ENV_PATH/bin/python" -m pip check
