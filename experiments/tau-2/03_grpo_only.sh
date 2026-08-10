#!/usr/bin/env bash
set -euo pipefail

# Strict Qwen3.5 GRPO ablation: no SkillBook and no SFT initialization.
# TRAIN_FILE must be produced by tau2_evolve.prepare_grpo_data without
# --skillbook. The user-simulator endpoint is embedded in that parquet.
#
# Required environment:
#   TRAIN_FILE, TRAIN_GPU, ROLLOUT_GPU
# Optional environment:
#   OUTPUT_DIR, TRAIN_VENV, MODEL_PATH, TOTAL_STEPS

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_VENV="${TRAIN_VENV:-$ROOT/.venv_qwen35}"
: "${TRAIN_FILE:?set TRAIN_FILE to the GRPO parquet}"
: "${TRAIN_GPU:?set TRAIN_GPU}"
: "${ROLLOUT_GPU:?set ROLLOUT_GPU}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/tau2_qwen35_grpo_only_$(date -u +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
TOTAL_STEPS="${TOTAL_STEPS:-9}"

export PYTHONPATH="$ROOT/experiments/tau-2/compat/qwen35_torch_fallback:${PYTHONPATH:-}"
export QWEN35_ENABLE_VERL_PATCHES=1
export QWEN35_TRAIN_TRITON_OVERLAY="${QWEN35_TRAIN_TRITON_OVERLAY:-$ROOT/.deps/qwen35-triton33}"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU,$ROLLOUT_GPU"
export TRAIN_VENV TRAIN_FILE OUTPUT_DIR MODEL_PATH TOTAL_STEPS
export N_GPUS=1 ROLLOUT_N_GPUS=1 SEPARATE_ROLLOUT=1
export SAVE_FREQ="${SAVE_FREQ:-3}"
export ACTOR_LR="${ACTOR_LR:-2e-6}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export N_GENERATIONS="${N_GENERATIONS:-8}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.98}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-16}"
export REWARD_WORKERS="${REWARD_WORKERS:-2}"
export MAX_TURNS="${MAX_TURNS:-12}"
export AGENT_THINKING="${AGENT_THINKING:-False}"
export TOOL_FORMAT="${TOOL_FORMAT:-hermes}"
export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-24}"
export CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-nixl_tau2}"
export CHECKPOINT_CUSTOM_BACKEND_MODULE="${CHECKPOINT_CUSTOM_BACKEND_MODULE:-tau2_evolve.nixl_checkpoint}"
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-True}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-tau2_qwen35_grpo_only}"

exec bash "$ROOT/experiments/tau-2/scripts/train_verl_grpo.sh" data.shuffle=False "$@"
