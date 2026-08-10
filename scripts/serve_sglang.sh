#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-${MODEL_PATH:-Qwen/Qwen3.5-2B}}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
PYTHON="${PYTHON:-python}"
CUDA_HOME="${CUDA_HOME:-}"
LOG_DIR="${LOG_DIR:-results/sglang_logs}"
mkdir -p "$LOG_DIR"

SAFE_NAME="$(echo "$SERVED_NAME" | tr '/:' '__')"
PID_FILE="$LOG_DIR/sglang_${SAFE_NAME}_${PORT}.pid"
LOG_FILE="$LOG_DIR/sglang_${SAFE_NAME}_${PORT}.log"

if [[ -n "$CUDA_HOME" ]]; then
  export PATH="$CUDA_HOME/bin:$(dirname "$PYTHON"):$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
else
  export PATH="$(dirname "$PYTHON"):$PATH"
fi

MODEL_TO_LOAD="$MODEL"
LORA_ARGS=()
RADIX_ARGS=()
[[ "${SGLANG_DISABLE_RADIX_CACHE:-0}" == "1" ]] && RADIX_ARGS=(--disable-radix-cache)
GRAPH_ARGS=()
[[ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" == "1" ]] && GRAPH_ARGS=(--disable-cuda-graph)
if [[ -f "$MODEL/adapter_config.json" ]]; then
  BASE="$($PYTHON - "$MODEL" <<'PY'
import json, sys
with open(sys.argv[1] + "/adapter_config.json") as f:
    print(json.load(f).get("base_model_name_or_path", ""))
PY
)"
  [[ -n "$BASE" ]] || { echo "[serve_sglang] adapter has no base model" >&2; exit 1; }
  MODEL_TO_LOAD="$BASE"
  LORA_ARGS=(--enable-lora --lora-paths "{\"lora_name\":\"$SERVED_NAME\",\"lora_path\":\"$MODEL\",\"pinned\":true}")
fi

if [[ "$MODEL_TO_LOAD" == *"Qwen3.5"* ]]; then
  export PYTHONPATH="$PWD/experiments/humaneval_mbpp/compat/sglang_qwen35:${PYTHONPATH:-}"
fi

echo "[serve_sglang] model=$MODEL_TO_LOAD served=$SERVED_NAME port=$PORT gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" nohup setsid "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL_TO_LOAD" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --context-length "${SGLANG_CONTEXT_LENGTH:-4096}" \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.50}" \
  --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-24}" \
  --linear-attn-backend "${SGLANG_LINEAR_ATTN_BACKEND:-flashinfer}" \
  --mamba-scheduler-strategy "${SGLANG_MAMBA_SCHEDULER_STRATEGY:-extra_buffer}" \
  --trust-remote-code \
  "${LORA_ARGS[@]}" \
  "${RADIX_ARGS[@]}" \
  "${GRAPH_ARGS[@]}" \
  >"$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

server_is_us() {
  curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | SERVED_NAME="$SERVED_NAME" "$PYTHON" -c '
import json, os, sys
try:
    ids = {item["id"] for item in json.load(sys.stdin).get("data", [])}
    raise SystemExit(0 if os.environ["SERVED_NAME"] in ids else 1)
except Exception:
    raise SystemExit(1)
'
}

for _ in $(seq 1 180); do
  if server_is_us; then
    echo "[serve_sglang] ready: http://127.0.0.1:$PORT/v1"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[serve_sglang] server exited; tailing log" >&2
    tail -80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "[serve_sglang] timed out; tailing log" >&2
tail -80 "$LOG_FILE" >&2 || true
kill "$PID" 2>/dev/null || true
exit 1
