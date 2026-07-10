#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Defaults to verl's own native SFT trainer (`python -m verl.trainer.
# sft_trainer`, LoRA by default -- matches train_grpo.sh's algorithm). Set
# SFT_TRAIN_CMD to override with a different launcher if your verl
# installation's SFT trainer differs.
PYTHON="${PYTHON:-python}"
: "${SFT_DATA:?set SFT_DATA to cycle_N/sft_pairs.parquet}"
: "${SFT_OUTPUT_DIR:?set SFT_OUTPUT_DIR}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"

if [[ -n "${SFT_TRAIN_CMD:-}" ]]; then
  export SFT_DATA SFT_OUTPUT_DIR MODEL_PATH
  eval "$SFT_TRAIN_CMD"
  exit 0
fi

N_GPUS="${N_GPUS:-1}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-8}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-2048}"
SFT_LR="${SFT_LR:-1e-4}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}"
SFT_PROJECT_NAME="${SFT_PROJECT_NAME:-verl_code_rl_sft}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-sft}"

# LoRA (default; same algorithm as train_grpo.sh). Set LORA_RANK=0 for
# full-parameter SFT -- note the adapter-extraction step below is skipped in
# that case, since a full-parameter run already saves loadable HF weights.
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"

LORA_ARGS=()
if [[ "$LORA_RANK" -gt 0 ]]; then
  LORA_ARGS=(
    model.lora_rank="$LORA_RANK"
    model.lora_alpha="$LORA_ALPHA"
    model.target_modules="$LORA_TARGET_MODULES"
  )
fi

if [[ ! -f "$SFT_DATA" ]]; then
  echo "[train_sft] missing $SFT_DATA. Run verl_code_rl.traces_to_sft first." >&2
  exit 2
fi

CKPT_DIR="$SFT_OUTPUT_DIR/verl_checkpoints"

set -x
torchrun --standalone --nnodes=1 --nproc_per_node="$N_GPUS" \
  -m verl.trainer.sft_trainer \
  data.train_files="$SFT_DATA" \
  data.train_batch_size="$SFT_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$SFT_MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="$SFT_MAX_LENGTH" \
  model.path="$MODEL_PATH" \
  model.trust_remote_code=True \
  optim.lr="$SFT_LR" \
  trainer.total_epochs="$SFT_TOTAL_EPOCHS" \
  trainer.project_name="$SFT_PROJECT_NAME" \
  trainer.experiment_name="$SFT_EXPERIMENT_NAME" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.save_freq=1 \
  'trainer.logger=["console"]' \
  "${LORA_ARGS[@]}" \
  "$@"
set +x

# verl's SFT checkpoint saver doesn't write a lora_adapter/adapter_config.json
# the way the GRPO actor does (that logic lives only in fsdp_workers.py's PPO
# path) -- extract a clean, serve_vllm.sh-loadable adapter ourselves, written
# directly to SFT_OUTPUT_DIR so run_full_pipeline.sh's existing
# `[[ -d "$OUT/sft_adapter" ]]` check picks it up unchanged.
if [[ "$LORA_RANK" -gt 0 ]]; then
  latest="$(find "$CKPT_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "$latest" ]]; then
    echo "[train_sft] extracting LoRA adapter from $latest"
    "$PYTHON" -m verl_code_rl.extract_sft_lora_adapter \
      --checkpoint-dir "$latest" \
      --base-model "$MODEL_PATH" \
      --output "$SFT_OUTPUT_DIR" \
      --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$LORA_TARGET_MODULES"
  else
    echo "[train_sft] WARNING: no checkpoint found under $CKPT_DIR, nothing to extract" >&2
  fi
fi
