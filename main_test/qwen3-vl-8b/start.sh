#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-$HOME/.conda/envs/mmtab-qwen/bin/python}"
MODEL="$($PYTHON -c 'import json; print(json.load(open("config.json"))["model_path"])')"
test -f "$MODEL/config.json"
mkdir -p logs pids
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -ne 4 ]; then echo 'GPUS must contain four available GPU IDs'; exit 1; fi
# Check every port before starting any process; do not duplicate existing services.
"$PYTHON" - <<'PY'
import socket
for p in range(8100,8104):
    with socket.socket() as s:
        s.bind(('127.0.0.1',p))
PY
for IDX in 0 1 2 3; do
  PORT=$((8100+IDX))
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$IDX]}" nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen3-vl-8b \
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-model-len 16384 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image":1,"video":0}' --generation-config vllm \
    > "logs/server_${PORT}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  echo $! > "pids/server_${PORT}.pid"
  echo "GPU ${GPU_LIST[$IDX]} -> port $PORT; PID $!"
done
echo 'Processes launched; run bash test.sh health to check readiness.'
