#!/usr/bin/env bash
set -euo pipefail

# Isolates GRPO's own effect, with no skill text injected anywhere, using the
# hyperparameters found (by an LR sweep of 1e-5/2e-5/5e-5) to actually move the
# policy within a short training budget:
#   - LR=5e-5 (the sweep winner; the project's old default of 1e-6 was ~10-50x
#     too conservative and left GRPO barely learning within 64 steps).
#   - ENFORCE_EAGER=True: this node's vLLM reproduces a CUDA-graph-only crash
#     ("illegal instruction" / "illegal memory access" / "unspecified launch
#     failure") and, short of crashing, gives numerically unreliable results
#     under CUDA graphs -- eager mode is the only setting confirmed to give
#     stable, reproducible pass rates on this hardware. train_grpo.sh already
#     defaults to this, set here for explicitness.
#   - Bigger micro-batch (32 vs the old default of 8): confirmed via
#     perf/mfu/actor to raise GPU utilization from ~8% MFU to ~29% MFU and cut
#     timing_s/update_actor from ~36s to ~10s per step, with memory still well
#     under the 80GB budget (~34GB observed).
#
# Prerequisites: small + large OpenAI-compatible vLLM servers already running
# at SMALL_BASE_URL / LARGE_BASE_URL. Needs a free GPU for TRAIN_GPU, separate
# from whatever GPU serves SMALL_BASE_URL.
#
# Usage: bash experiments/03_grpo_only.sh

cd "$(dirname "$0")/../.."

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
TRAIN_GPU="${TRAIN_GPU:-4}"
GRPO_PORT="${GRPO_PORT:-8021}"
GRPO_SERVED_NAME="${GRPO_SERVED_NAME:-grpo-only-adapter}"
LR="${LR:-5e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}"
RESULTS_DIR="${RESULTS_DIR:-results/exp_grpo_only_$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$RESULTS_DIR"
echo "[grpo-only] results dir: $RESULTS_DIR"

echo
echo "[grpo-only] step 1: collect oracle traces (small-first, fallback on failure, no router)"
"$PYTHON" -m verl_code_rl.collect_traces \
  --data "$DATA" --split train --limit "$TRAIN_LIMIT" \
  --small-model "$SMALL_MODEL" --large-model "$LARGE_MODEL" \
  --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL" --api-key "$API_KEY" \
  --workers "$WORKERS" --probe-only \
  --out "$RESULTS_DIR/traces.jsonl"

echo
echo "[grpo-only] step 2: prepare GRPO data (no skillbook -- isolates GRPO alone)"
"$PYTHON" -m verl_code_rl.prepare_data \
  --traces "$RESULTS_DIR/traces.jsonl" \
  --input "$DATA" \
  --out-dir "$RESULTS_DIR/processed"

echo
echo "[grpo-only] step 3: GRPO training (LoRA, eager, LR=$LR) on GPU $TRAIN_GPU"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
TRAIN_FILE="$RESULTS_DIR/processed/train.parquet" \
VAL_FILE="$RESULTS_DIR/processed/val.parquet" \
MODEL_PATH="$SMALL_MODEL" \
LR="$LR" \
TOTAL_EPOCHS="$TOTAL_EPOCHS" \
ENFORCE_EAGER=True \
PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
PPO_MICRO_BATCH_SIZE_PER_GPU="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
PROJECT_NAME="grpo_only_exp" \
EXPERIMENT_NAME="$(basename "$RESULTS_DIR")" \
OUTPUT_DIR="$RESULTS_DIR/verl_checkpoints" \
  scripts/train_grpo.sh

latest="$(find "$RESULTS_DIR/verl_checkpoints" -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1)"
if [[ -z "$latest" ]]; then
  echo "[grpo-only] ERROR: no checkpoint produced by training" >&2
  exit 1
fi
if [[ -f "$latest/actor/lora_adapter/adapter_config.json" ]]; then
  TRAINED_MODEL="$latest/actor/lora_adapter"
else
  TRAINED_MODEL="$latest/actor"
fi

echo
echo "[grpo-only] step 4: baseline eval (untouched small model, no skill)"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "$SMALL_BASE_URL" --api-key "$API_KEY" \
  --model "$SMALL_MODEL" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --out "$RESULTS_DIR/baseline_eval.jsonl"

echo
echo "[grpo-only] step 5: serve the GRPO-trained adapter (eager) on GPU $TRAIN_GPU port $GRPO_PORT"
ENFORCE_EAGER=1 MODEL_PATH="$TRAINED_MODEL" SERVED_NAME="$GRPO_SERVED_NAME" \
PORT="$GRPO_PORT" GPU="$TRAIN_GPU" \
  scripts/serve_vllm.sh

echo
echo "[grpo-only] step 6: eval the GRPO-trained checkpoint (no skill prefix)"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "http://127.0.0.1:${GRPO_PORT}/v1" --api-key "$API_KEY" \
  --model "$GRPO_SERVED_NAME" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --out "$RESULTS_DIR/grpo_eval.jsonl"

PORT="$GRPO_PORT" SERVED_NAME="$GRPO_SERVED_NAME" scripts/stop_vllm.sh

echo
echo "[grpo-only] done. Compare pass@1 printed above for baseline_eval vs grpo_eval."
echo "[grpo-only] IMPORTANT: only trust numbers from an eager-mode server (this script always uses one)."
echo "[grpo-only] results in $RESULTS_DIR"
