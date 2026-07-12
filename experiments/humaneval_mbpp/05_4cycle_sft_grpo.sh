#!/usr/bin/env bash
set -euo pipefail

# Full 4-cycle closed loop (collect -> skillbook -> SFT -> GRPO -> router ->
# ablation): the "full" pipeline, contrasted against 04_4cycle_sft.sh's
# SFT-only variant to isolate GRPO's added value on top of Skills+SFT. Each
# cycle, SFT warm-starts the LoRA adapter and GRPO continues training it
# (LORA_ADAPTER_PATH threading, see CLAUDE.md); the adapter also continues
# across cycles. Probe-only oracle collection (small model tried first,
# large model only invoked on failure).
#
# GRPO uses the hyperparameters found by an LR sweep to actually move the
# policy (see 03_grpo_only.sh's header for the full rationale): LR=5e-5,
# eager mode (mandatory for numerically reliable results on this hardware),
# and a bigger micro-batch for GPU utilization (~8%->~29% MFU observed).
#
# Prerequisites: small + large OpenAI-compatible vLLM servers already running
# at SMALL_BASE_URL / LARGE_BASE_URL. Needs a free GPU for GRPO_GPU, separate
# from whatever GPU serves SMALL_BASE_URL/LARGE_BASE_URL. SMALL_RELOAD_CMD
# must point at a command that restarts the small server with the new
# checkpoint (default: scripts/reload_small_vllm.sh on SMALL_RELOAD_GPU).
#
# Usage: bash experiments/05_4cycle_sft_grpo.sh

cd "$(dirname "$0")/../.."

DATA="${DATA:-data/raw/he_mbpp.jsonl}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SMALL_BASE_URL="${SMALL_BASE_URL:-http://127.0.0.1:8000/v1}"
LARGE_BASE_URL="${LARGE_BASE_URL:-http://127.0.0.1:8001/v1}"
GRPO_GPU="${GRPO_GPU:-7}"
SMALL_RELOAD_GPU="${SMALL_RELOAD_GPU:-5}"
N_CYCLES="${N_CYCLES:-4}"
LIMIT="${LIMIT:--1}"
LR="${LR:-5e-5}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}"
DISTILLER_MODEL="${DISTILLER_MODEL:-openai/gpt-5.5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_4cycle_sft_grpo_$(date -u +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="$GRPO_GPU" \
SMALL_BASE_URL="$SMALL_BASE_URL" LARGE_BASE_URL="$LARGE_BASE_URL" \
SMALL_MODEL="$SMALL_MODEL" LARGE_MODEL="$LARGE_MODEL" DATA="$DATA" \
SMALL_RELOAD_CMD="GPU=$SMALL_RELOAD_GPU scripts/reload_small_vllm.sh" \
ENABLE_SFT=1 SKIP_GRPO=0 \
LR="$LR" \
PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
PPO_MICRO_BATCH_SIZE_PER_GPU="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
DISTILLER_MODEL="$DISTILLER_MODEL" \
  bash scripts/run_full_pipeline.sh \
  --n-cycles "$N_CYCLES" --limit "$LIMIT" --probe-only \
  --experiment-name "$EXPERIMENT_NAME"

echo
echo "[4cycle-sft-grpo] done: results/$EXPERIMENT_NAME"
echo "[4cycle-sft-grpo] per-cycle ablation: results/$EXPERIMENT_NAME/cycle_N/e2e_ablation_summary.json"
