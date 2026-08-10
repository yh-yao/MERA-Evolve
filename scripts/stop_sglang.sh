#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-results/sglang_logs}"
found=0
for pid_file in "$LOG_DIR"/*_"$PORT".pid; do
  [[ -e "$pid_file" ]] || continue
  found=1
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop_sglang] killing process group=$pid ($pid_file)"
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop_sglang] process group did not exit; sending SIGKILL" >&2
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
done

# A replacement server advertises the same model ID, so waiting for the old
# endpoint to disappear is required before the next health check can identify
# the new process reliably.
for _ in $(seq 1 30); do
  if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if [[ "$found" == "0" ]]; then
  echo "[stop_sglang] no pid file found for port $PORT"
fi
