#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/processed/train.parquet}"
VAL_FILE="${VAL_FILE:-data/processed/val.parquet}"
REWARD_FILE="${REWARD_FILE:-$PWD/verl_code_rl/reward.py}"

PROJECT_NAME="${PROJECT_NAME:-verl_code_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_1p5b_he_mbpp_grpo}"
N_GPUS="${N_GPUS:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
N_GENERATIONS="${N_GENERATIONS:-8}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# LoRA is the default training mode, matching router-skills-evolve's HumanEval
# GRPO algorithm (frozen base + trained adapter, r=16). Set LORA_RANK=0 to fall
# back to full-parameter fine-tuning.
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
# Path to a previous cycle's adapter to continue training, keeping MODEL_PATH
# as the original base checkpoint (verl loads base + this adapter together).
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "[train_grpo] missing parquet files. Run scripts/prepare_data.sh first." >&2
  exit 2
fi

# verl's vLLM rollout engine syncs LoRA weights into itself every step via its
# own dynamic add_lora/remove_lora API (see vllm_rollout/utils.py) whenever
# lora_rank>0 -- no --enable-lora server flag needs to be passed by us here.
# load_format=safetensors + layered_summon=True is what makes that adapter-only
# sync path (vs. a full merged-weight sync) kick in.
LORA_ARGS=()
if [[ "$LORA_RANK" -gt 0 ]]; then
  LORA_ARGS=(
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    actor_rollout_ref.model.target_modules="$LORA_TARGET_MODULES"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
  )
  [[ -n "$LORA_ADAPTER_PATH" ]] && LORA_ARGS+=(actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH")
fi

set -x
"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  custom_reward_function.path="$REWARD_FILE" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$N_GENERATIONS" \
  actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$TOP_P" \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  'trainer.logger=["console"]' \
  ${OUTPUT_DIR:+trainer.default_local_dir="$OUTPUT_DIR"} \
  "${LORA_ARGS[@]}" \
  "$@"
