#!/usr/bin/env bash
set -euo pipefail

# Isolates SFT's own effect, with no skill text injected anywhere (so the
# comparison is purely "trained vs untrained", not entangled with skills):
#   1. collect oracle traces (small-first, fallback on failure, no router).
#   2. build teacher/self-repair SFT pairs from those SAME traces (no skillbook).
#   3. SFT-train the small model (LoRA, verl's native SFT trainer).
#   4. eval baseline (untouched checkpoint) vs the SFT-trained checkpoint,
#      both with no skill prefix, on the same held-out split.
#
# Prerequisites: small + large OpenAI-compatible vLLM servers already running
# at SMALL_BASE_URL / LARGE_BASE_URL. Needs a free GPU for SFT_GPU, separate
# from whatever GPU serves SMALL_BASE_URL (training and serving can't safely
# share one GPU's memory budget).
#
# Usage: bash experiments/humaneval_mbpp/02_sft_only.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATA="${DATA:-data/raw/he_mbpp.jsonl}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SMALL_BASE_URL="${SMALL_BASE_URL:-http://127.0.0.1:8000/v1}"
LARGE_BASE_URL="${LARGE_BASE_URL:-http://127.0.0.1:8001/v1}"
API_KEY="${API_KEY:-EMPTY}"
TRAIN_LIMIT="${TRAIN_LIMIT:--1}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"
WORKERS="${WORKERS:-32}"
SFT_GPU="${SFT_GPU:-4}"
SFT_PORT="${SFT_PORT:-8020}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}"
SFT_SERVED_NAME="${SFT_SERVED_NAME:-sft-only-adapter}"
RESULTS_DIR="${RESULTS_DIR:-results/exp_sft_only_$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$RESULTS_DIR"
echo "[sft-only] results dir: $RESULTS_DIR"

echo
echo "[sft-only] step 1: collect oracle traces (small-first, fallback on failure, no router)"
"$PYTHON" -m verl_code_rl.collect_traces \
  --data "$DATA" --split train --limit "$TRAIN_LIMIT" \
  --small-model "$SMALL_MODEL" --large-model "$LARGE_MODEL" \
  --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL" --api-key "$API_KEY" \
  --workers "$WORKERS" --probe-only \
  --out "$RESULTS_DIR/traces.jsonl"

echo
echo "[sft-only] step 2: build SFT pairs (no skillbook -- isolates SFT alone)"
"$PYTHON" -m verl_code_rl.traces_to_sft \
  --traces "$RESULTS_DIR/traces.jsonl" \
  --output "$RESULTS_DIR/sft_pairs.parquet"

echo
echo "[sft-only] step 3: SFT training (LoRA, verl native trainer) on GPU $SFT_GPU"
CUDA_VISIBLE_DEVICES="$SFT_GPU" \
SFT_DATA="$RESULTS_DIR/sft_pairs.parquet" \
SFT_OUTPUT_DIR="$RESULTS_DIR/sft_adapter" \
MODEL_PATH="$SMALL_MODEL" \
SFT_TOTAL_EPOCHS="$SFT_TOTAL_EPOCHS" \
  scripts/train_sft.sh

if [[ ! -d "$RESULTS_DIR/sft_adapter" ]]; then
  echo "[sft-only] ERROR: no adapter produced by SFT training" >&2
  exit 1
fi

echo
echo "[sft-only] step 4: baseline eval (untouched small model, no skill)"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "$SMALL_BASE_URL" --api-key "$API_KEY" \
  --model "$SMALL_MODEL" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --out "$RESULTS_DIR/baseline_eval.jsonl"

echo
echo "[sft-only] step 5: serve the SFT-trained adapter on GPU $SFT_GPU port $SFT_PORT"
MODEL_PATH="$RESULTS_DIR/sft_adapter" SERVED_NAME="$SFT_SERVED_NAME" \
PORT="$SFT_PORT" GPU="$SFT_GPU" \
  scripts/serve_vllm.sh

echo
echo "[sft-only] step 6: eval the SFT-trained checkpoint (no skill prefix)"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "http://127.0.0.1:${SFT_PORT}/v1" --api-key "$API_KEY" \
  --model "$SFT_SERVED_NAME" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --out "$RESULTS_DIR/sft_eval.jsonl"

PORT="$SFT_PORT" SERVED_NAME="$SFT_SERVED_NAME" scripts/stop_vllm.sh

echo
echo "[sft-only] done. Compare pass@1 printed above for baseline_eval vs sft_eval."
echo "[sft-only] results in $RESULTS_DIR"
