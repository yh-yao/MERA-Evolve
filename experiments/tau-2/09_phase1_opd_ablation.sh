#!/usr/bin/env bash
# Phase 1: on-policy distillation ablation under one rollout budget.
#
#   A  current method               teacher-suffix SFT (verified 27B rescues of
#                                   failed student prefixes), then GRPO as B
#   B  GRPO only                    reward = official + annealed action recall
#   C  distillation only            reverse-KL to the teacher, no task reward
#   D  GRPO + distillation (main)   reward as B, plus reverse-KL to the teacher
#   E  D without shaping            official reward only from step 0
#   O  GRPO, official reward only   no teacher, no shaping (4B + AReaL RL plan)
#   F  error-point correction SFT   only the teacher's verified fix at the
#                                   student's first wrong turn is trained
#                                   (CORRECTIONS_PARQUET, error_points_to_sft)
#   G  F, then GRPO as B            from F's adapter
#
# B vs D isolates the teacher signal, D vs E isolates action-recall shaping.
# Every group trains from the base student on the same TRAIN parquet with the
# same steps x prompts x generations, then is evaluated with the Phase 0
# protocol (08_phase0_eval_protocol.sh) on VAL. EVAL is never touched here.
#
# Required:
#   RESULTS_DIR          absolute, one per configuration (immutable evidence)
#   PHASE1_GROUPS        subset of "A B C D E F G" (default: B C D E)
#   CORRECTIONS_PARQUET  groups F/G: output of tau2_evolve.error_points_to_sft
#   TRAIN_GPU ROLLOUT_GPU AGENT_GPU USER_GPU
#   MERA_TEACHER_BASE_URL  external teacher /v1 (needed by A, C, D, E)
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
# A pre-built GRPO parquet (e.g. frontier-filtered prompts) replaces the
# default all-TRAIN prompts; it must embed this run's USER_PORT.
[[ -n "${GRPO_PARQUET:-}" ]] && { mkdir -p "$RESULTS_DIR"; cp "$GRPO_PARQUET" "$RESULTS_DIR/grpo_train.parquet"; }
DISTILL_LOSS_MODE="${DISTILL_LOSS_MODE:-k3}"
DISTILL_LOSS_COEF="${DISTILL_LOSS_COEF:-1.0}"

export TAU2_WORKSPACE
export TAU2_PARTITION_PATH="${TAU2_PARTITION_PATH:-$SCRIPT_DIR/data/partition_v2.json}"
export TAU2_SITE_PACKAGES="${TAU2_SITE_PACKAGES:-$(ls -d "$TAU2_WORKSPACE"/.venv_tau2/lib/python3*/site-packages | head -1)}"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/compat:${PYTHONPATH:-}"
export PATH="$TRAIN_VENV/bin:$PATH"
export CUDA_HOME="${QWEN35_CUDA_HOME:-$ROOT/.deps/cuda-12.8}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# RTX 5090 settings validated by a 2-step smoke run (2026-10-24): no CUDA
# peer access -> host-shm weight sync; the fp32 embedding (2 GB) must fit
# one bucket; fused log-prob kernels (which require remove-padding) avoid
# materializing 248k-vocab logits that OOM a 32 GB card.
export MERA_WEIGHT_TRANSFER_SHM="${MERA_WEIGHT_TRANSFER_SHM:-1}"
export CHECKPOINT_BUCKET_MB="${CHECKPOINT_BUCKET_MB:-2560}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
# A 20-step run OOMed at step 17 from allocator fragmentation; checkpoint
# every SAVE_EVERY steps so VERL's resume_mode=auto restarts mid-run.
export MERA_ACTOR_CUDA_ALLOC_CONF="${MERA_ACTOR_CUDA_ALLOC_CONF:-expandable_segments:True}"
SAVE_EVERY="${SAVE_EVERY:-5}"
TRAIN_EXTRA_ARGS=(actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS:-True}")
# FSDP CPU offload made 32 GB actor GPUs OOM *earlier* (compute_log_prob,
# step 2 at a 16k context), so it is off by default.
TRAIN_EXTRA_ARGS+=(
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
)
# Native Qwen3.5 protocol (tau2_evolve.native_chat): XML tool calls, tools
# rendered by the chat template, and horizons long enough for a tau2 task
# (base frontier conversations: median 28 messages; eval allows 60 steps).
export TOOL_FORMAT="${TOOL_FORMAT:-qwen3_coder}"
export MAX_TURNS="${MAX_TURNS:-30}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
# 12288 OOMed a 32 GB actor GPU intermittently (fragmentation + 5.5 GB
# checkpoint-engine buffers on top of ~17 GB allocated); 8192 keeps clipping
# low (mean response ~5.5k tokens) with a 16k total context.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
# Extra VERL overrides from the caller (e.g. memory settings for a larger student).
read -r -a _extra <<< "${EXTRA_TRAIN_ARGS:-}"
TRAIN_EXTRA_ARGS+=("${_extra[@]}")

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
  SERVERS_DIR="$RESULTS_DIR/servers" AGENT_PORT="$AGENT_PORT" USER_PORT="$USER_PORT" \
    bash "$SCRIPT_DIR/scripts/chunked_collect.sh" --bucket TRAIN --workers "${WORKERS:-64}" --max-steps 40 \
    --agent-model openai/evol-llm-agent --user-model openai/evol-llm-user \
    --agent-max-tokens 1024 --no-agent-thinking --no-user-thinking --out "$TRAIN_TRACES" >> "$LOG" 2>&1
  "$TRAIN_VENV/bin/python" -m tau2_evolve.prepare_grpo_data --traces "$TRAIN_TRACES" --output "$PARQUET" \
    --user-base-url "http://127.0.0.1:$USER_PORT/v1" --no-user-thinking >> "$LOG" 2>&1
fi

# A: the pre-plan recipe with the teacher swapped for the 27B. OPD replays
# failed base-student prefixes and keeps only verified teacher suffixes; SFT
# trains on those suffixes (prefix tokens are context only), then GRPO
# continues from the SFT adapter with group B's reward and budget.
train_group_A() {
  local out="$1"
  local opd="$out/opd_traces.jsonl" sft_pairs="$out/sft_pairs.parquet" sft_adapter="$out/sft_adapter"
  : "${MERA_TEACHER_BASE_URL:?group A needs the teacher endpoint for OPD}"
  if [[ ! -f "$out/opd.done" ]]; then
    log "group A: collecting verified teacher suffixes (OPD)"
    rm -f "$opd" "$out/router_data.jsonl"
    "$TAU2_PY" -m tau2_evolve.collect_opd --traces "$TRAIN_TRACES" --output "$opd" \
      --router-data "$out/router_data.jsonl" \
      --teacher-model "openai/${MERA_TEACHER_MODEL:-evol-llm-teacher}" --teacher-base-url "$MERA_TEACHER_BASE_URL" \
      --teacher-max-tokens 1024 --no-teacher-thinking \
      --user-base-url "http://127.0.0.1:$USER_PORT/v1" --max-steps 40 \
      --branch-attempts "${OPD_BRANCH_ATTEMPTS:-3}" --workers "${OPD_WORKERS:-16}" >> "$out/opd.log" 2>&1 \
      && touch "$out/opd.done" || return 1
  fi
  [[ -s "$sft_pairs" ]] || "$TRAIN_VENV/bin/python" -m tau2_evolve.traces_to_sft \
    --traces "$TRAIN_TRACES" --traces "$opd" --output "$sft_pairs" >> "$out/sft.log" 2>&1
  if [[ ! -s "$sft_adapter/adapter_model.safetensors" ]]; then
    log "group A: teacher-suffix SFT"
    ( cd "$ROOT" && source "$TRAIN_VENV/bin/activate" && \
      env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHON="$TRAIN_VENV/bin/python" \
        SFT_DATA="$sft_pairs" SFT_OUTPUT_DIR="$sft_adapter" MODEL_PATH="$BASE_MODEL" \
        SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}" SFT_MICRO_BATCH_SIZE_PER_GPU=1 \
        SFT_LR="${SFT_LR:-1e-7}" SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}" \
        SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-24576}" SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_LENGTH:-24576}" \
        QWEN35_ENABLE_VERL_PATCHES=1 QWEN35_TRAIN_TRITON_OVERLAY="$ROOT/.deps/qwen35-triton33" FLA_TILELANG=0 \
        SFT_ENABLE_THINKING=False SFT_ATTN_IMPLEMENTATION=sdpa SFT_USE_TORCH_COMPILE=False \
        SFT_DATASET_PATH="$SCRIPT_DIR/tau2_evolve/sft_dataset.py" SFT_DATASET_NAME=Qwen35MultiTurnSFTDataset \
        SFT_SAVE_FREQ=-1 SFT_PROJECT_NAME=tau2_phase1 SFT_EXPERIMENT_NAME=group_A_sft \
        bash scripts/train_sft.sh data.num_workers=0 \
          model.use_remove_padding=True model.use_fused_kernels="${USE_FUSED_KERNELS:-True}" \
          model.enable_gradient_checkpointing=True ) > "$out/sft.log" 2>&1 || return 1
  fi
  log "group A: GRPO from the SFT adapter"
  run_rl "$out" A "$(group_env B) LORA_ADAPTER_PATH=$sft_adapter"
}

# F/G: SFT on error-point corrections. Rows carry the student's own context
# and one trainable turn (the verified teacher fix); see collect_error_points.
train_correction_sft() {
  local adapter="$1" log_file="$2"
  : "${CORRECTIONS_PARQUET:?groups F and G need CORRECTIONS_PARQUET}"
  [[ -s "$adapter/adapter_model.safetensors" ]] && return 0
  log "correction SFT on $CORRECTIONS_PARQUET -> $adapter"
  ( cd "$ROOT" && source "$TRAIN_VENV/bin/activate" && \
    env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHON="$TRAIN_VENV/bin/python" \
      SFT_DATA="$CORRECTIONS_PARQUET" SFT_OUTPUT_DIR="$adapter" MODEL_PATH="$BASE_MODEL" \
      SFT_BATCH_SIZE="${CORR_SFT_BATCH_SIZE:-8}" SFT_MICRO_BATCH_SIZE_PER_GPU=1 \
      SFT_LR="${CORR_SFT_LR:-1e-5}" SFT_TOTAL_EPOCHS="${CORR_SFT_EPOCHS:-2}" \
      SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-24576}" SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_LENGTH:-24576}" \
      QWEN35_ENABLE_VERL_PATCHES=1 QWEN35_TRAIN_TRITON_OVERLAY="$ROOT/.deps/qwen35-triton33" FLA_TILELANG=0 \
      SFT_ENABLE_THINKING=False SFT_ATTN_IMPLEMENTATION=sdpa SFT_USE_TORCH_COMPILE=False \
      SFT_DATASET_PATH="$SCRIPT_DIR/tau2_evolve/sft_dataset.py" SFT_DATASET_NAME=Qwen35NativeSFTDataset \
      SFT_SAVE_FREQ=-1 SFT_PROJECT_NAME=tau2_phase1 SFT_EXPERIMENT_NAME=correction_sft \
      bash scripts/train_sft.sh data.num_workers=0 \
        model.use_remove_padding=True model.use_fused_kernels="${USE_FUSED_KERNELS:-True}" \
        model.enable_gradient_checkpointing=True ) > "$log_file" 2>&1
}

# One RL training run with a single resume retry: VERL restarts from the
# latest SAVE_EVERY checkpoint under OUTPUT_DIR.
run_rl() {
  local out="$1" name="$2" extra_env="$3" attempt
  for attempt in 1 2; do
    # shellcheck disable=SC2086
    env $extra_env DISTILL_LOSS_MODE="$DISTILL_LOSS_MODE" DISTILL_LOSS_COEF="$DISTILL_LOSS_COEF" \
      TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-}" \
    RAY_WORKER_SETUP_HOOK=tau2_evolve.qwen35_worker_setup.install_qwen35_worker_patches \
    TRAIN_FILE="$PARQUET" TRAIN_GPU="$TRAIN_GPU" ROLLOUT_GPU="$ROLLOUT_GPU" MODEL_PATH="$BASE_MODEL" \
    TOTAL_STEPS="$TOTAL_STEPS" SAVE_FREQ="$SAVE_EVERY" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    N_GENERATIONS="$N_GENERATIONS" ACTOR_LR="$ACTOR_LR" OUTPUT_DIR="$out/verl" \
      EXPERIMENT_NAME="tau2_phase1_$name" \
      bash "$SCRIPT_DIR/03_grpo_only.sh" "${TRAIN_EXTRA_ARGS[@]}" >> "$out/train.log" 2>&1 \
      && [[ -s "$out/verl/final_adapter_path.txt" ]] && return 0
    log "group $name: training attempt $attempt failed; resuming from the latest checkpoint"
  done
  return 1
}

group_env() {
  case "$1" in
    B) echo "DISTILLATION=0 TAU2_ACTION_RECALL_WEIGHT=0.5 TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS=$TOTAL_ROLLOUTS" ;;
    C) echo "DISTILLATION=1 DISTILL_USE_TASK_REWARDS=False TAU2_ACTION_RECALL_WEIGHT=0.5 TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS=$TOTAL_ROLLOUTS" ;;
    D) echo "DISTILLATION=1 DISTILL_USE_TASK_REWARDS=True TAU2_ACTION_RECALL_WEIGHT=0.5 TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS=$TOTAL_ROLLOUTS" ;;
    E) echo "DISTILLATION=1 DISTILL_USE_TASK_REWARDS=True TAU2_ACTION_RECALL_WEIGHT=0" ;;
    O) echo "DISTILLATION=0 TAU2_ACTION_RECALL_WEIGHT=0" ;;  # GRPO on the official tau2 reward only
    *) echo "unknown group $1" >&2; exit 2 ;;
  esac
}

for group in $PHASE1_GROUPS; do
  out="$RESULTS_DIR/group_$group"
  mkdir -p "$out"
  if [[ ! -s "$out/verl/final_adapter_path.txt" ]]; then
    if [[ "$group" == "A" ]]; then
      train_group_A "$out" || { log "group A: training FAILED (see $out/*.log)"; continue; }
    elif [[ "$group" == "F" ]]; then
      train_correction_sft "$out/sft_adapter" "$out/sft.log" \
        || { log "group F: SFT FAILED (see $out/sft.log)"; continue; }
      mkdir -p "$out/verl"; echo "$out/sft_adapter" > "$out/verl/final_adapter_path.txt"
    elif [[ "$group" == "G" ]]; then
      # Reuse F's adapter when F ran in this RESULTS_DIR (same data, same seed).
      sft="$RESULTS_DIR/group_F/sft_adapter"
      [[ -s "$sft/adapter_model.safetensors" ]] || sft="$out/sft_adapter"
      train_correction_sft "$sft" "$out/sft.log" \
        || { log "group G: SFT FAILED (see $out/sft.log)"; continue; }
      log "group G: GRPO from $sft"
      run_rl "$out" G "$(group_env B) LORA_ADAPTER_PATH=$sft" \
        || { log "group G: training FAILED (see $out/train.log)"; continue; }
    else
      log "group $group: training ($(group_env "$group"))"
      run_rl "$out" "$group" "$(group_env "$group")" \
        || { log "group $group: training FAILED (see $out/train.log)"; continue; }
    fi
  fi
  adapter="$(cat "$out/verl/final_adapter_path.txt")"
  log "group $group: evaluating $adapter on VAL"
  # Student restarts with the adapter on the eval port; the user simulator is shared.
  if [[ -f "$RESULTS_DIR/servers/evol-llm-agent_server.pid" ]]; then
    kill "$(cat "$RESULTS_DIR/servers/evol-llm-agent_server.pid")" 2>/dev/null || true
    rm -f "$RESULTS_DIR/servers/evol-llm-agent_server.pid"; sleep 15
  fi
  RESULTS_DIR="$out/eval" AGENT_ADAPTER="$adapter" RUN_LABEL="$group" BUCKETS=VAL RUN_SEEDS="${EVAL_RUN_SEEDS:-300 1300}" \
    AGENT_PORT="$AGENT_PORT" USER_PORT="$USER_PORT" \
    bash "$SCRIPT_DIR/08_phase0_eval_protocol.sh" >> "$LOG" 2>&1
  kill "$(cat "$out/eval/evol-llm-agent_server.pid")" 2>/dev/null || true; sleep 15
done

log "phase1 done; compare each group with: python -m tau2_evolve.eval_stats compare --baseline <base_VAL_seed300.jsonl> --candidate <group>/eval/<G>_VAL_seed300.jsonl"
