#!/usr/bin/env bash
set -euo pipefail

# MBPP Qwen3.5-2B evolution with a fully cached GPT-5.6 Luna teacher/fallback.
# Only GPT-5.6 Sol is called online, once per cycle, to distill the SkillBook.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "FATAL: $ROOT/.env is missing" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
: "${COMMONSTACK_API_KEY:?COMMONSTACK_API_KEY must be set in .env}"

export SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen3.5-2B}"
export LARGE_MODEL="${LARGE_MODEL:-openai/gpt-5.6-luna}"
export LARGE_BASE_URL="${COMMONSTACK_BASE_URL:-https://api.commonstack.ai/v1}"
export LARGE_CACHE="${LARGE_CACHE:-$ROOT/results/rebuttal_luna_code_cache_20260728_020810.jsonl}"
export LARGE_MAX_TOKENS="${LARGE_MAX_TOKENS:-2048}"
export DISTILLER_MODEL="${DISTILLER_MODEL:-openai/gpt-5.6-sol}"
export DISTILLER_BASE_URL="${DISTILLER_BASE_URL:-$LARGE_BASE_URL}"
export DISTILLER_API_KEY="${DISTILLER_API_KEY:-$COMMONSTACK_API_KEY}"
export RUN_PRETRAIN_EVAL="${RUN_PRETRAIN_EVAL:-1}"
export N_CYCLES="${N_CYCLES:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_2b_mbpp_luna_sol_4cycle_$(date -u +%Y%m%d_%H%M%S)}"

# The previous Qwen3.5-2B run peaked at cycle 2. Reduce the final-cycle update
# while retaining four checkpoints so the best held-out cycle can be selected.
export SFT_LR_SCHEDULE="${SFT_LR_SCHEDULE:-1e-7,1e-7,5e-8,2e-8}"
export GRPO_LR_SCHEDULE="${GRPO_LR_SCHEDULE:-2e-6,1e-6,5e-7,2e-7}"

exec bash experiments/humaneval_mbpp/06_qwen35_2b_mbpp_4cycle.sh
