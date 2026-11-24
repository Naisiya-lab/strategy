#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_PATH="${ENV_PATH:-$HOME/.conda/envs/mmtab-syntab}"
if [ ! -x "$ENV_PATH/bin/python" ]; then conda create -p "$ENV_PATH" python=3.10 -y; fi
mkdir -p vendor logs
if [ ! -d vendor/SynTab-LLaVA/.git ]; then
  git clone https://github.com/bang123-box/SynTab-LLaVA.git vendor/SynTab-LLaVA
fi
"$ENV_PATH/bin/python" -m pip install --upgrade pip
"$ENV_PATH/bin/python" -m pip install -r requirements-inference.txt
"$ENV_PATH/bin/python" -m pip install -c requirements-inference.txt -e vendor/SynTab-LLaVA
if [ "${INSTALL_FLASH_ATTN:-0}" = 1 ]; then
  "$ENV_PATH/bin/python" -m pip install ninja packaging
  "$ENV_PATH/bin/python" -m pip install 'flash-attn==2.5.8' --no-build-isolation
fi
"$ENV_PATH/bin/python" -m pip check
git -C vendor/SynTab-LLaVA rev-parse HEAD > logs/official_code_commit.txt
"$ENV_PATH/bin/python" -m pip freeze > logs/environment.txt
"$ENV_PATH/bin/python" - <<'PY'
import torch, transformers, llava
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
print('torch:',torch.__version__,'CUDA:',torch.version.cuda)
print('transformers:',transformers.__version__)
print('llava source:',llava.__file__)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('SynTab environment imports passed')
PY
