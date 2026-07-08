#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-results/vllm_logs}"
found=0
for pid_file in "$LOG_DIR"/*_"$PORT".pid; do
  [[ -e "$pid_file" ]] || continue
  found=1
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop_vllm] killing pid=$pid ($pid_file)"
    kill "$pid" || true
  fi
  rm -f "$pid_file"
done

if [[ "$found" == "0" ]]; then
  echo "[stop_vllm] no pid file found for port $PORT"
fi
