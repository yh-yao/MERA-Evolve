#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
VLLM_BIN="${VLLM_BIN:-vllm}"
LOG_DIR="${LOG_DIR:-results/vllm_logs}"
mkdir -p "$LOG_DIR"

SAFE_NAME="$(echo "$SERVED_NAME" | tr '/:' '__')"
PID_FILE="$LOG_DIR/vllm_${SAFE_NAME}_${PORT}.pid"
LOG_FILE="$LOG_DIR/vllm_${SAFE_NAME}_${PORT}.log"

LORA_ARGS=()
MODEL_TO_LOAD="$MODEL"
if [[ -f "$MODEL/adapter_config.json" ]]; then
  BASE="$("${PYTHON:-python}" - "$MODEL" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1] + "/adapter_config.json"))
print(cfg.get("base_model_name_or_path", ""))
PY
)"
  MODEL_TO_LOAD="$BASE"
  LORA_ARGS=(--enable-lora --lora-modules "$SERVED_NAME=$MODEL" --enforce-eager)
fi

echo "[serve_vllm] model=$MODEL_TO_LOAD served=$SERVED_NAME port=$PORT gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" \
  nohup "$VLLM_BIN" serve "$MODEL_TO_LOAD" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
    --dtype "${VLLM_DTYPE:-bfloat16}" \
    --trust-remote-code \
    "${LORA_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "[serve_vllm] pid=$PID log=$LOG_FILE pid_file=$PID_FILE"

for _ in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[serve_vllm] ready: http://127.0.0.1:$PORT/v1"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[serve_vllm] server exited; tailing log" >&2
    tail -50 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "[serve_vllm] timed out; tailing log" >&2
tail -50 "$LOG_FILE" >&2 || true
kill "$PID" 2>/dev/null || true
exit 1
