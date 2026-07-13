#!/usr/bin/env bash
set -euo pipefail

# Full 4-cycle closed loop (collect -> skillbook -> SFT -> router -> ablation),
# GRPO skipped so the reported gains are attributable to Skills+SFT alone.
# Probe-only oracle collection: small model tried first, large model only
# invoked when small's execution fails. Eager mode throughout (see
# 03_grpo_only.sh's comment on why -- irrelevant here since GRPO is skipped,
# kept for parity with 05_4cycle_sft_grpo.sh).
#
# Prerequisites: small + large OpenAI-compatible vLLM servers already running
# at SMALL_BASE_URL / LARGE_BASE_URL. Needs a free GPU for SFT_GPU, separate
# from whatever GPU serves SMALL_BASE_URL/LARGE_BASE_URL. SMALL_RELOAD_CMD
# must point at a command that restarts the small server with the new
# checkpoint (default: scripts/reload_small_vllm.sh on SMALL_RELOAD_GPU).
#
# Usage: bash experiments/humaneval_mbpp/04_4cycle_sft.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA="${DATA:-data/raw/he_mbpp.jsonl}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SMALL_BASE_URL="${SMALL_BASE_URL:-http://127.0.0.1:8000/v1}"
LARGE_BASE_URL="${LARGE_BASE_URL:-http://127.0.0.1:8001/v1}"
SFT_GPU="${SFT_GPU:-4}"
SMALL_RELOAD_GPU="${SMALL_RELOAD_GPU:-2}"
N_CYCLES="${N_CYCLES:-4}"
LIMIT="${LIMIT:--1}"
DISTILLER_MODEL="${DISTILLER_MODEL:-openai/gpt-5.5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_4cycle_sft_$(date -u +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="$SFT_GPU" \
SMALL_BASE_URL="$SMALL_BASE_URL" LARGE_BASE_URL="$LARGE_BASE_URL" \
SMALL_MODEL="$SMALL_MODEL" LARGE_MODEL="$LARGE_MODEL" DATA="$DATA" \
SMALL_RELOAD_CMD="GPU=$SMALL_RELOAD_GPU scripts/reload_small_vllm.sh" \
ENABLE_SFT=1 SKIP_GRPO=1 \
DISTILLER_MODEL="$DISTILLER_MODEL" \
  bash scripts/run_full_pipeline.sh \
  --n-cycles "$N_CYCLES" --limit "$LIMIT" --probe-only \
  --experiment-name "$EXPERIMENT_NAME"

echo
echo "[4cycle-sft] done: results/$EXPERIMENT_NAME"
echo "[4cycle-sft] per-cycle ablation: results/$EXPERIMENT_NAME/cycle_N/e2e_ablation_summary.json"
