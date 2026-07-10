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

# Default on: a recurring `torch.AcceleratorError: CUDA error: an illegal
# memory access` crash was reproduced twice under concurrent load with CUDA
# graphs enabled (consistently after ~480-510 requests) and confirmed fixed
# by --enforce-eager. Set ENFORCE_EAGER=0 to opt back into CUDA graphs (more
# throughput, but re-exposes that crash) once/if it's root-caused upstream.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
EAGER_ARGS=()
[[ "$ENFORCE_EAGER" == "1" ]] && EAGER_ARGS=(--enforce-eager)

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
  LORA_ARGS=(--enable-lora --lora-modules "$SERVED_NAME=$MODEL")
fi

# Default on: this node's driver caps out below what flashinfer's JIT-compiled
# sampler kernel targets (it builds against the system's local CUDA toolchain,
# not torch's bundled runtime), causing a hard crash on first sampler call.
# Set VLLM_USE_FLASHINFER_SAMPLER=1 to re-enable if a matching toolchain is
# ever installed.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

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
    "${EAGER_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "[serve_vllm] pid=$PID log=$LOG_FILE pid_file=$PID_FILE"

# Check the *served model name*, not just that something answers on the port:
# if a prior, untracked process is still bound to $PORT (e.g. started outside
# this script, so stop_vllm.sh had no pid file to kill it), our new process
# fails to bind and exits, but a bare curl check would see the stale process
# respond and wrongly report success -- silently leaving the wrong model
# (e.g. missing the LoRA adapter we just tried to load) serving.
server_is_us() {
  curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | SERVED_NAME="$SERVED_NAME" "${PYTHON:-python}" -c '
import json, os, sys
try:
    data = json.load(sys.stdin)
    ids = {m["id"] for m in data.get("data", [])}
    sys.exit(0 if os.environ["SERVED_NAME"] in ids else 1)
except Exception:
    sys.exit(1)
'
}

for _ in $(seq 1 150); do
  if server_is_us; then
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
