#!/usr/bin/env bash
set -euo pipefail

ROOT="/shared_home/yuhang.yao/MERA-Evolve"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/tau2_4cycle_nogrpo_retry}"
PID_FILE="$RESULTS_DIR/pipeline.pid"
LAUNCH_LOG="$RESULTS_DIR/launch.log"

mkdir -p "$RESULTS_DIR"

if [[ -s "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "noGRPO pipeline already running (pid=$pid)"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

for endpoint in \
  "http://127.0.0.1:8200/v1/models" \
  "http://127.0.0.1:8201/v1/models"; do
  if ! curl -sf "$endpoint" >/dev/null; then
    echo "FATAL: required model server is unavailable: $endpoint" >&2
    exit 1
  fi
done

cd "$ROOT/experiments/tau-2"
nohup setsid env \
  RESULTS_DIR="$RESULTS_DIR" \
  AGENT_GPU=4 AGENT_PORT=8200 \
  USER_GPU=7 USER_PORT=8201 \
  GRPO_GPU=6 ENABLE_GRPO=0 N_CYCLES=4 \
  bash run_cycle_pipeline.sh >> "$LAUNCH_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "FATAL: noGRPO pipeline exited during startup; see $LAUNCH_LOG" >&2
  exit 1
fi

echo "started noGRPO pipeline (pid=$pid, results=$RESULTS_DIR)"
