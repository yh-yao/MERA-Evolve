#!/usr/bin/env bash
set -euo pipefail

# Default local reload contract for run_full_pipeline.sh.  For a remote serving
# fleet, set SMALL_RELOAD_CMD to the corresponding deployment command instead.
cd "$(dirname "$0")/.."

: "${MODEL_PATH:?MODEL_PATH must point to the newly trained model or LoRA adapter}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
SERVED_NAME="${SERVED_NAME:-$MODEL_PATH}"

PORT="$PORT" scripts/stop_vllm.sh
MODEL_PATH="$MODEL_PATH" PORT="$PORT" GPU="$GPU" SERVED_NAME="$SERVED_NAME" scripts/serve_vllm.sh

# serve_vllm has already waited for readiness; print the advertised models so
# the pipeline log records exactly which process accepted the new checkpoint.
curl -fsS "http://127.0.0.1:${PORT}/v1/models"
