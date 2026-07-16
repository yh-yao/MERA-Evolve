#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-4B fallback runtime smoke test. It uses a CUDA 12.8 toolkit matching
# the PyTorch wheel, serves with the native qwen3_xml parser, and executes one
# real tau2 task.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${TRAIN_VENV:-$ROOT/.venv_qwen35}"
CUDA_HOME="${QWEN35_CUDA_HOME:-$ROOT/.deps/cuda-12.8}"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
TAU2_PY="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
FALLBACK_GPU="${FALLBACK_GPU:-2}"
FALLBACK_PORT="${FALLBACK_PORT:-8254}"
USER_PORT="${USER_PORT:-8242}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/tau2_4b_fallback_smoke_$(date -u +%Y%m%d_%H%M%S)}"
MODEL_PATH="${FALLBACK_MODEL_PATH:-Qwen/Qwen3.5-4B}"
CACHED_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"
if [[ "$MODEL_PATH" == "Qwen/Qwen3.5-4B" && -d "$CACHED_MODEL" ]]; then
  MODEL_PATH="$(find "$CACHED_MODEL" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
fi
mkdir -p "$RESULTS_DIR"

if [[ ! -x "$VENV/bin/python" || ! -x "$VENV/bin/vllm" ]]; then
  echo "FATAL: Qwen3.5 environment is incomplete at $VENV; set TRAIN_VENV" >&2
  exit 2
fi
if [[ ! -x "$TAU2_PY" ]]; then
  echo "FATAL: tau2 Python not found at $TAU2_PY; set TAU2_WORKSPACE or TAU2_PYTHON" >&2
  exit 2
fi
VENV_SITE_PACKAGES="${TRAIN_SITE_PACKAGES:-$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')}"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "FATAL: CUDA 12.8 toolkit missing at $CUDA_HOME" >&2
  exit 2
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES="$FALLBACK_GPU" VLLM_USE_FLASHINFER_SAMPLER=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$ROOT/experiments/tau-2/compat/qwen35_torch_fallback" \
nohup "$VENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name fallback-4b-fixed --port "$FALLBACK_PORT" \
  --gpu-memory-utilization 0.55 --max-model-len 32768 --max-num-batched-tokens 32768 \
  --max-num-seqs 16 --enable-prefix-caching --dtype bfloat16 --trust-remote-code \
  --enforce-eager --no-async-scheduling --language-model-only \
  --gdn-prefill-backend flashinfer --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  > "$RESULTS_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$FALLBACK_PORT/v1/models" >/dev/null && break
  sleep 5
done
curl -sf "http://127.0.0.1:$FALLBACK_PORT/v1/models" >/dev/null

PYTHONPATH="$ROOT/experiments/tau-2:$VENV_SITE_PACKAGES" \
"$TAU2_PY" -m tau2_evolve.collect_traces \
  --bucket TRAIN --domain airline --limit 1 --workers 1 --max-steps 40 \
  --agent-max-tokens 512 --agent-model openai/fallback-4b-fixed \
  --agent-base-url "http://127.0.0.1:$FALLBACK_PORT/v1" \
  --user-model openai/evol-llm-user --user-base-url "http://127.0.0.1:$USER_PORT/v1" \
  --no-agent-thinking --no-user-thinking --out "$RESULTS_DIR/smoke.jsonl" \
  > "$RESULTS_DIR/run.log" 2>&1

test "$(wc -l < "$RESULTS_DIR/smoke.jsonl")" -eq 1
"$VENV/bin/python" - "$RESULTS_DIR/smoke.jsonl" <<'PY'
import json
import sys

row = json.loads(open(sys.argv[1]).readline())
if row.get("error") or not row.get("n_steps"):
    raise SystemExit(f"4B runtime smoke failed: {row.get('error') or row}")
PY
echo "[fallback-4b-smoke] passed runtime smoke: $RESULTS_DIR/smoke.jsonl"
