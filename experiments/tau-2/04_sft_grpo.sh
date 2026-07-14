#!/usr/bin/env bash
set -euo pipefail

# Multi-cycle SkillBook + SFT + verl GRPO experiment. Use 03_grpo_only.sh for
# the strict GRPO ablation without either SkillBook or SFT initialization.
#
# Required environment:
#   AGENT_GPU, AGENT_PORT, USER_GPU, USER_PORT, TRAIN_GPU
# Optional environment:
#   RESULTS_DIR, N_CYCLES, START_CYCLE, INITIAL_ADAPTER, INITIAL_SKILLBOOK

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/tau2_sft_grpo_$(date -u +%Y%m%d_%H%M%S)}"

export RESULTS_DIR
export ENABLE_GRPO=1

exec bash "$ROOT/experiments/tau-2/scripts/run_cycle_pipeline.sh"
