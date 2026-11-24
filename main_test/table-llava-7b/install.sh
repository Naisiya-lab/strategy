#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_PATH="${ENV_PATH:-$HOME/.conda/envs/mmtab-table-llava}"
if [ ! -x "$ENV_PATH/bin/python" ]; then conda create -p "$ENV_PATH" python=3.10 -y; fi
mkdir -p vendor logs
if [ ! -d vendor/Table-LLaVA/.git ]; then
  git clone https://github.com/SpursGoZmy/Table-LLaVA.git vendor/Table-LLaVA
fi
"$ENV_PATH/bin/python" -m pip install --upgrade pip
"$ENV_PATH/bin/python" -m pip install 'numpy<2' 'protobuf<5' 'huggingface-hub<1' 'tqdm==4.70.1'
"$ENV_PATH/bin/python" -m pip install -e vendor/Table-LLaVA
"$ENV_PATH/bin/python" -m pip check
git -C vendor/Table-LLaVA rev-parse HEAD > logs/official_code_commit.txt
"$ENV_PATH/bin/python" -m pip freeze > logs/environment.txt
echo 'Installed original Table-LLaVA environment; no GPU services are required.'
