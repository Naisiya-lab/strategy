#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_PATH="${ENV_PATH:-$HOME/.conda/envs/mmtab-qwen}"
if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.11 -y
fi
"$ENV_PATH/bin/python" -m pip install --upgrade pip
"$ENV_PATH/bin/python" -m pip install 'vllm==0.11.0' 'qwen-vl-utils==0.0.14' openai pillow
"$ENV_PATH/bin/python" -m pip check
mkdir -p logs
"$ENV_PATH/bin/python" -m pip freeze > logs/environment.txt
nvidia-smi > logs/nvidia-smi.txt
"$ENV_PATH/bin/python" - <<'PY'
import torch, vllm
print('vLLM:', vllm.__version__, 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA not available'
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.ones(16, device=f'cuda:{i}').sum().item())
PY
