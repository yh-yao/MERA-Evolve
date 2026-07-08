#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
PORT="${PORT:-8000}"

"$PYTHON" -m verl_code_rl.eval_vllm \
  --model "$MODEL" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  "$@"
