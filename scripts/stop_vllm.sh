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
  # Warn loudly if something is still listening anyway (e.g. started outside
  # this script's pid-file convention) -- otherwise a subsequent serve_vllm.sh
  # call can silently mistake that untracked process for its own new one.
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[stop_vllm] WARNING: port $PORT is still answering requests despite" >&2
    echo "[stop_vllm] no known pid file -- an untracked server is running and" >&2
    echo "[stop_vllm] was NOT stopped. Find and kill it manually if a reload" >&2
    echo "[stop_vllm] was expected to replace it." >&2
  fi
fi
