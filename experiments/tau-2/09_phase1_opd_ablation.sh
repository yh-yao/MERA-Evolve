#!/usr/bin/env bash
# Phase 1: on-policy distillation ablation under one rollout budget.
#
#   B  GRPO only                    reward = official + annealed action recall
#   C  distillation only            reverse-KL to the teacher, no task reward
#   D  GRPO + distillation (main)   reward as B, plus reverse-KL to the teacher
#   E  D without shaping            official reward only from step 0
#
# B vs D isolates the teacher signal, D vs E isolates action-recall shaping.
# Every group trains from the base student on the same TRAIN parquet with the
# same steps x prompts x generations, then is evaluated with the Phase 0
# protocol (08_phase0_eval_protocol.sh) on VAL. EVAL is never touched here.
#
# Required:
#   RESULTS_DIR          absolute, one per configuration (immutable evidence)
#   PHASE1_GROUPS        subset of "B C D E" (default: all)
#   TRAIN_GPU ROLLOUT_GPU AGENT_GPU USER_GPU
#   MERA_TEACHER_BASE_URL  external teacher /v1 (needed by C, D, E)
#   TEACHER_MODEL_PATH     teacher weights (tokenizer check)
#   BASE_MODEL USER_MODEL_PATH VLLM_BIN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
TAU2_PY="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
TRAIN_VENV="${TRAIN_VENV:-$ROOT/.venv_qwen35}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR}"
[[ "$RESULTS_DIR" == /* ]] || { echo "RESULTS_DIR must be absolute" >&2; exit 2; }
# Not GROUPS: that is a read-only bash array (the user's unix groups).
PHASE1_GROUPS="${PHASE1_GROUPS:-B C D E}"
: "${TRAIN_GPU:?}" "${ROLLOUT_GPU:?}" "${AGENT_GPU:?}" "${USER_GPU:?}"
: "${BASE_MODEL:?}" "${USER_MODEL_PATH:?}" "${VLLM_BIN:?}"
AGENT_PORT="${AGENT_PORT:-8280}" USER_PORT="${USER_PORT:-8281}"
export BASE_MODEL USER_MODEL_PATH VLLM_BIN AGENT_GPU USER_GPU

# One rollout budget for every group.
TOTAL_STEPS="${TOTAL_STEPS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
N_GENERATIONS="${N_GENERATIONS:-8}"
ACTOR_LR="${ACTOR_LR:-2e-6}"
TOTAL_ROLLOUTS=$((TOTAL_STEPS * TRAIN_BATCH_SIZE * N_GENERATIONS))
DISTILL_LOSS_MODE="${DISTILL_LOSS_MODE:-k3}"
DISTILL_LOSS_COEF="${DISTILL_LOSS_COEF:-1.0}"

export TAU2_WORKSPACE
export TAU2_PARTITION_PATH="${TAU2_PARTITION_PATH:-$SCRIPT_DIR/data/partition_v2.json}"
export TAU2_SITE_PACKAGES="${TAU2_SITE_PACKAGES:-$(ls -d "$TAU2_WORKSPACE"/.venv_tau2/lib/python3*/site-packages | head -1)}"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/compat:${PYTHONPATH:-}"
export PATH="$TRAIN_VENV/bin:$PATH"
export CUDA_HOME="${QWEN35_CUDA_HOME:-$ROOT/.deps/cuda-12.8}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MERA_WEIGHT_TRANSFER_SHM="${MERA_WEIGHT_TRANSFER_SHM:-1}"
export CHECKPOINT_BUCKET_MB="${CHECKPOINT_BUCKET_MB:-2560}"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/phase1.log"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

cat > "$RESULTS_DIR/config_snapshot.json" <<JSON
{"GROUPS": "$PHASE1_GROUPS", "BASE_MODEL": "$BASE_MODEL", "USER_MODEL_PATH": "$USER_MODEL_PATH",
 "TEACHER_MODEL_PATH": "${TEACHER_MODEL_PATH:-}", "MERA_TEACHER_BASE_URL": "${MERA_TEACHER_BASE_URL:-}",
 "TOTAL_STEPS": $TOTAL_STEPS, "TRAIN_BATCH_SIZE": $TRAIN_BATCH_SIZE, "N_GENERATIONS": $N_GENERATIONS,
 "TOTAL_ROLLOUTS": $TOTAL_ROLLOUTS, "ACTOR_LR": "$ACTOR_LR",
 "DISTILL_LOSS_MODE": "$DISTILL_LOSS_MODE", "DISTILL_LOSS_COEF": "$DISTILL_LOSS_COEF",
 "PARTITION": "$TAU2_PARTITION_PATH"}
JSON

serve_common() {
  # Student (base, for TRAIN prompt collection) and the training-time user
  # simulator; the eval phase restarts the student with each adapter.
  RESULTS_DIR="$RESULTS_DIR/servers" BUCKETS="" AGENT_PORT="$AGENT_PORT" USER_PORT="$USER_PORT" \
    bash "$SCRIPT_DIR/08_phase0_eval_protocol.sh" >> "$LOG" 2>&1
}

# ---- 1. TRAIN prompts from the base student (one trial, no fallback) ----
serve_common
TRAIN_TRACES="$RESULTS_DIR/train_traces_base.jsonl"
PARQUET="$RESULTS_DIR/grpo_train.parquet"
if [[ ! -s "$PARQUET" ]]; then
  log "collecting TRAIN prompts with the base student"
  "$TAU2_PY" -m tau2_evolve.collect_traces --bucket TRAIN --workers "${WORKERS:-64}" --max-steps 40 --resume \
    --agent-model openai/evol-llm-agent --agent-base-url "http://127.0.0.1:$AGENT_PORT/v1" \
    --user-model openai/evol-llm-user --user-base-url "http://127.0.0.1:$USER_PORT/v1" \
    --agent-max-tokens 1024 --no-agent-thinking --no-user-thinking --out "$TRAIN_TRACES" >> "$LOG" 2>&1
  "$TRAIN_VENV/bin/python" -m tau2_evolve.prepare_grpo_data --traces "$TRAIN_TRACES" --output "$PARQUET" \
    --user-base-url "http://127.0.0.1:$USER_PORT/v1" --no-user-thinking >> "$LOG" 2>&1
fi

group_env() {
  case "$1" in
    B) echo "DISTILLATION=0 TAU2_ACTION_RECALL_WEIGHT=0.5 TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS=$TOTAL_ROLLOUTS" ;;
    C) echo "DISTILLATION=1 DISTILL_USE_TASK_REWARDS=False TAU2_ACTION_RECALL_WEIGHT=0.5 TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS=$TOTAL_ROLLOUTS" ;;
    D) echo "DISTILLATION=1 DISTILL_USE_TASK_REWARDS=True TAU2_ACTION_RECALL_WEIGHT=0.5 TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS=$TOTAL_ROLLOUTS" ;;
    E) echo "DISTILLATION=1 DISTILL_USE_TASK_REWARDS=True TAU2_ACTION_RECALL_WEIGHT=0" ;;
    *) echo "unknown group $1" >&2; exit 2 ;;
  esac
}

for group in $PHASE1_GROUPS; do
  out="$RESULTS_DIR/group_$group"
  mkdir -p "$out"
  if [[ ! -s "$out/verl/final_adapter_path.txt" ]]; then
    log "group $group: training ($(group_env "$group"))"
    # shellcheck disable=SC2046
    env $(group_env "$group") DISTILL_LOSS_MODE="$DISTILL_LOSS_MODE" DISTILL_LOSS_COEF="$DISTILL_LOSS_COEF" \
      TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-}" \
      RAY_WORKER_SETUP_HOOK=tau2_evolve.qwen35_worker_setup.install_qwen35_worker_patches \
      TRAIN_FILE="$PARQUET" TRAIN_GPU="$TRAIN_GPU" ROLLOUT_GPU="$ROLLOUT_GPU" MODEL_PATH="$BASE_MODEL" \
      TOTAL_STEPS="$TOTAL_STEPS" SAVE_FREQ="$TOTAL_STEPS" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
      N_GENERATIONS="$N_GENERATIONS" ACTOR_LR="$ACTOR_LR" OUTPUT_DIR="$out/verl" \
      EXPERIMENT_NAME="tau2_phase1_$group" \
      bash "$SCRIPT_DIR/03_grpo_only.sh" > "$out/train.log" 2>&1 \
      || { log "group $group: training FAILED (see $out/train.log)"; continue; }
  fi
  adapter="$(cat "$out/verl/final_adapter_path.txt")"
  log "group $group: evaluating $adapter on VAL"
  # Student restarts with the adapter on the eval port; the user simulator is shared.
  if [[ -f "$RESULTS_DIR/servers/evol-llm-agent_server.pid" ]]; then
    kill "$(cat "$RESULTS_DIR/servers/evol-llm-agent_server.pid")" 2>/dev/null || true
    rm -f "$RESULTS_DIR/servers/evol-llm-agent_server.pid"; sleep 15
  fi
  RESULTS_DIR="$out/eval" AGENT_ADAPTER="$adapter" RUN_LABEL="$group" BUCKETS=VAL RUN_SEEDS="300 1300" \
    AGENT_PORT="$AGENT_PORT" USER_PORT="$USER_PORT" \
    bash "$SCRIPT_DIR/08_phase0_eval_protocol.sh" >> "$LOG" 2>&1
  kill "$(cat "$out/eval/evol-llm-agent_server.pid")" 2>/dev/null || true; sleep 15
done

log "phase1 done; compare each group with: python -m tau2_evolve.eval_stats compare --baseline <base_VAL_seed300.jsonl> --candidate <group>/eval/<G>_VAL_seed300.jsonl"
