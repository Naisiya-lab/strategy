#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for PORT in 8100 8101 8102 8103; do
  FILE="pids/server_${PORT}.pid"
  [ -f "$FILE" ] || continue
  PID=$(cat "$FILE")
  if [[ "$PID" =~ ^[0-9]+$ ]] && [ -r "/proc/$PID/cmdline" ]; then
    CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline")
    if [[ "$CMD" == *"vllm.entrypoints.openai.api_server"* && "$CMD" == *"--port $PORT "* ]]; then
      kill -TERM "$PID"
      echo "Stopped owned service: port $PORT, PID $PID"
    else
      echo "PID $PID no longer matches service; skipped"
    fi
  fi
done
