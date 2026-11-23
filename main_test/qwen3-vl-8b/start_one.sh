#!/usr/bin/env bash
# Start one missing replica without disturbing other running services.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU="${1:?Usage: bash start_one.sh GPU_ID API_PORT}"
PORT="${2:?Usage: bash start_one.sh GPU_ID API_PORT}"
[[ "$GPU" =~ ^[0-9]+$ && "$PORT" =~ ^810[0-3]$ ]] || exit 1
PYTHON="${PYTHON:-$HOME/.conda/envs/mmtab-qwen/bin/python}"
MODEL="$("$PYTHON" -c 'import json; print(json.load(open("config.json"))["model_path"])')"
test -f "$MODEL/config.json"
INTERNAL_PORT=$((52000+(PORT-8100)*100))
"$PYTHON" - "$PORT" "$INTERNAL_PORT" <<'PY'
import socket, sys
for port in map(int, sys.argv[1:]):
    with socket.socket() as s:
        s.bind(('', port))
PY
mkdir -p logs pids
LOG="logs/server_${PORT}_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="$GPU" VLLM_PORT="$INTERNAL_PORT" nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3-vl-8b \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":1,"video":0}' --generation-config vllm \
  > "$LOG" 2>&1 &
echo $! > "pids/server_${PORT}.pid"
echo "Launched GPU=$GPU API=$PORT internal-start=$INTERNAL_PORT PID=$! log=$LOG"
