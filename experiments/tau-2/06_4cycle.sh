#!/usr/bin/env bash
set -euo pipefail

# Recommended full TAU-2 reproduction: four cycles of
# student rollout -> local 4B OPD -> SkillBook -> SFT -> GRPO -> router -> eval.
# The runner starts the student and user/teacher vLLM servers itself.
#
# Required GPU/port assignments:
#   AGENT_GPU, AGENT_PORT, USER_GPU, USER_PORT, TRAIN_GPU, ROLLOUT_GPU
# Optional resume settings:
#   RESULTS_DIR, START_CYCLE, REUSE_EXISTING_ARTIFACTS,
#   INITIAL_ADAPTER, INITIAL_SKILLBOOK
# All low-level pipeline settings remain overridable through environment vars.
#
# Example:
#   AGENT_GPU=0 AGENT_PORT=8260 USER_GPU=1 USER_PORT=8261 \
#   TRAIN_GPU=2 ROLLOUT_GPU=3 \
#     scripts/run_experiment.sh tau2 4cycle

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${AGENT_GPU:?set AGENT_GPU}"
: "${AGENT_PORT:?set AGENT_PORT}"
: "${USER_GPU:?set USER_GPU}"
: "${USER_PORT:?set USER_PORT}"
: "${TRAIN_GPU:?set TRAIN_GPU}"
: "${ROLLOUT_GPU:?set ROLLOUT_GPU}"

export RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/tau2_qwen35_opd_router_4cycle_$(date -u +%Y%m%d_%H%M%S)}"
export TRAIN_VENV="${TRAIN_VENV:-$ROOT/.venv_qwen35}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-Qwen/Qwen3.5-4B}"

export N_CYCLES="${N_CYCLES:-4}"
export ENABLE_OPD="${ENABLE_OPD:-1}"
export ENABLE_ROUTER="${ENABLE_ROUTER:-1}"
export ENABLE_GRPO="${ENABLE_GRPO:-1}"

# Validated defaults from the Qwen3.5 run. Override these for smaller machines.
export COLLECT_WORKERS="${COLLECT_WORKERS:-96}"
export EVAL_WORKERS="${EVAL_WORKERS:-96}"
export OPD_WORKERS="${OPD_WORKERS:-16}"
export OPD_BRANCH_ATTEMPTS="${OPD_BRANCH_ATTEMPTS:-3}"
export SFT_LR="${SFT_LR:-1e-7}"
export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
export SFT_BALANCE_DOMAINS="${SFT_BALANCE_DOMAINS:-0}"
export GRPO_TOTAL_STEPS="${GRPO_TOTAL_STEPS:-10}"
export GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-8}"
export GRPO_N_GENERATIONS="${GRPO_N_GENERATIONS:-4}"
export GRPO_ACTOR_LR="${GRPO_ACTOR_LR:-5e-6}"

# The local Qwen3.5-4B service supplies user simulation, OPD correction,
# SkillBook distillation, and routed fallback. No hosted teacher is required.
export DISTILLER_MODEL="${DISTILLER_MODEL:-openai/evol-llm-user}"
export DISTILLER_BASE_URL="${DISTILLER_BASE_URL:-http://127.0.0.1:$USER_PORT/v1}"
export DISTILLER_API_KEY="${DISTILLER_API_KEY:-EMPTY}"
export OPD_TEACHER_MODEL="${OPD_TEACHER_MODEL:-openai/evol-llm-user}"
export OPD_TEACHER_BASE_URL="${OPD_TEACHER_BASE_URL:-$DISTILLER_BASE_URL}"
export OPD_TEACHER_API_KEY="${OPD_TEACHER_API_KEY:-$DISTILLER_API_KEY}"

exec bash "$ROOT/experiments/tau-2/scripts/run_cycle_pipeline.sh"
