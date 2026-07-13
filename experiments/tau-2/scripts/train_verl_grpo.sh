#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-/shared_home/yuhang.yao/router-skills-evolve}"
TAU2_STAGE2_ROOT="${TAU2_STAGE2_ROOT:-$TAU2_WORKSPACE/tau2_stage2}"
: "${TRAIN_FILE:?set TRAIN_FILE}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
N_GPUS="${N_GPUS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-10}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
N_GENERATIONS="${N_GENERATIONS:-4}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-16}"
REWARD_WORKERS="${REWARD_WORKERS:-1}"
MAX_TURNS="${MAX_TURNS:-12}"
MAX_MODEL_LENGTH="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"

cd "$ROOT"
source venv/bin/activate
export TAU2_WORKSPACE TAU2_STAGE2_ROOT
export PYTHONPATH="$EXPERIMENT_DIR:$EXPERIMENT_DIR/compat:$TAU2_STAGE2_ROOT/code:$TAU2_STAGE2_ROOT/code/vendor/tau2-bench/src:${PYTHONPATH:-}"

LORA_ARGS=(
  actor_rollout_ref.model.lora_rank=16
  actor_rollout_ref.model.lora_alpha=32
  'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj]'
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=True
)
[[ -n "$LORA_ADAPTER_PATH" ]] && LORA_ARGS+=(actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH")

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" data.val_files="$TRAIN_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" data.val_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LENGTH" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.02 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$N_GENERATIONS" \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path="$EXPERIMENT_DIR/config/interaction.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$EXPERIMENT_DIR/config/agent_loop.yaml" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  reward.num_workers="$REWARD_WORKERS" \
  trainer.project_name=tau2_verl_grpo \
  trainer.experiment_name="${EXPERIMENT_NAME:-tau2_verl_grpo}" \
  trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.val_before_train=False trainer.save_freq="$TOTAL_STEPS" trainer.test_freq=-1 \
  trainer.total_epochs=100 trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  'trainer.logger=["console"]' \
  "${LORA_ARGS[@]}" "$@"

ADAPTER_PATH="$(find "$OUTPUT_DIR" -mindepth 3 -maxdepth 3 -type d -path '*/actor/lora_adapter' -print | sort -V | tail -n 1)"
if [[ -z "$ADAPTER_PATH" || ! -s "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  echo "[train_verl_grpo] no valid LoRA adapter checkpoint under $OUTPUT_DIR" >&2
  exit 1
fi
printf '%s\n' "$ADAPTER_PATH" > "$OUTPUT_DIR/final_adapter_path.txt"
echo "[train_verl_grpo] final adapter: $ADAPTER_PATH"
