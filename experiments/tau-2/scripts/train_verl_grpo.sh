#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
TAU2_STAGE2_ROOT="${TAU2_STAGE2_ROOT:-$TAU2_WORKSPACE/tau2_stage2}"
TRAIN_VENV="${TRAIN_VENV:-$ROOT/venv}"
: "${TRAIN_FILE:?set TRAIN_FILE}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
N_GPUS="${N_GPUS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-10}"
SAVE_FREQ="${SAVE_FREQ:-$TOTAL_STEPS}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
N_GENERATIONS="${N_GENERATIONS:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-16}"
REWARD_WORKERS="${REWARD_WORKERS:-1}"
MAX_TURNS="${MAX_TURNS:-12}"
MAX_MODEL_LENGTH="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
AGENT_THINKING="${AGENT_THINKING:-}"
TOOL_FORMAT="${TOOL_FORMAT:-hermes}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
SEPARATE_ROLLOUT="${SEPARATE_ROLLOUT:-0}"
ROLLOUT_N_GPUS="${ROLLOUT_N_GPUS:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
CHECKPOINT_BUCKET_MB="${CHECKPOINT_BUCKET_MB:-256}"
CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-nccl}"
CHECKPOINT_CUSTOM_BACKEND_MODULE="${CHECKPOINT_CUSTOM_BACKEND_MODULE:-}"
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-}"
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-}"
RAY_WORKER_SETUP_HOOK="${RAY_WORKER_SETUP_HOOK:-}"
REUSE_TRAINING_ADAPTER="${REUSE_TRAINING_ADAPTER:-0}"

cd "$ROOT"
source "$TRAIN_VENV/bin/activate"
export TAU2_WORKSPACE TAU2_STAGE2_ROOT
export PYTHONPATH="$EXPERIMENT_DIR:$EXPERIMENT_DIR/compat:$TAU2_STAGE2_ROOT/code:$TAU2_STAGE2_ROOT/code/vendor/tau2-bench/src:${PYTHONPATH:-}"

if [[ "$MODEL_PATH" == *"Qwen3.5"* && "$SEPARATE_ROLLOUT" == "1" ]]; then
  : "${QWEN35_TRAIN_TRITON_OVERLAY:?set QWEN35_TRAIN_TRITON_OVERLAY for Qwen3.5 training}"
  [[ -f "$QWEN35_TRAIN_TRITON_OVERLAY/triton/__init__.py" ]] || {
    echo "[train_verl_grpo] invalid Triton overlay: $QWEN35_TRAIN_TRITON_OVERLAY" >&2
    exit 2
  }
  "$TRAIN_VENV/bin/python" - <<'PY'
import importlib.util
import os

import triton
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

overlay = os.path.realpath(os.environ["QWEN35_TRAIN_TRITON_OVERLAY"])
assert triton.__version__.startswith("3.6."), (
    f"rollout runtime must retain Triton 3.6, got {triton.__version__} from {triton.__file__}"
)
assert importlib.util.find_spec("tilelang") is not None, "tilelang package is missing"
assert is_flash_linear_attention_available(), "fla-core is unavailable"
assert is_causal_conv1d_available(), "causal-conv1d is unavailable"
print(f"[train_verl_grpo] runtime split: rollout=triton {triton.__version__}, actor={overlay}")
PY
fi

if [[ "$MODEL_PATH" == *"Qwen3.5"* && -n "$LORA_ADAPTER_PATH" ]]; then
  TRAINING_ADAPTER_PATH="$OUTPUT_DIR/training_init_adapter"
  if [[ "$REUSE_TRAINING_ADAPTER" == "1" && -s "$TRAINING_ADAPTER_PATH/adapter_model.safetensors" ]]; then
    echo "[train_verl_grpo] reusing training adapter: $TRAINING_ADAPTER_PATH"
  else
    "$TRAIN_VENV/bin/python" -m verl_code_rl.prepare_qwen35_training_adapter \
      --input "$LORA_ADAPTER_PATH" --output "$TRAINING_ADAPTER_PATH"
  fi
  LORA_ADAPTER_PATH="$TRAINING_ADAPTER_PATH"
fi

LORA_ARGS=(
  actor_rollout_ref.model.lora_rank="$LORA_RANK"
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
  actor_rollout_ref.model.target_modules="$LORA_TARGET_MODULES"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=True
)
CHAT_TEMPLATE_ARGS=()
MODEL_CONFIG_ARGS=()
VLLM_ENGINE_ARGS=()
[[ -n "$AGENT_THINKING" ]] && \
  CHAT_TEMPLATE_ARGS+=(+data.apply_chat_template_kwargs.enable_thinking="$AGENT_THINKING")
[[ -n "$MODEL_ATTN_IMPLEMENTATION" ]] && \
  MODEL_CONFIG_ARGS+=(+actor_rollout_ref.model.override_config.attn_implementation="$MODEL_ATTN_IMPLEMENTATION")
[[ -n "$VLLM_GDN_PREFILL_BACKEND" ]] && \
  VLLM_ENGINE_ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend="$VLLM_GDN_PREFILL_BACKEND")
[[ -n "$VLLM_LANGUAGE_MODEL_ONLY" ]] && \
  VLLM_ENGINE_ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only="$VLLM_LANGUAGE_MODEL_ONLY")
[[ -n "$LORA_ADAPTER_PATH" ]] && LORA_ARGS+=(actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH")

TRAIN_ENTRYPOINT="verl.trainer.main_ppo"
SEPARATION_ARGS=()
if [[ "$SEPARATE_ROLLOUT" == "1" ]]; then
  TRAIN_ENTRYPOINT="verl.experimental.one_step_off_policy.main_ppo"
  VERL_TRAINER_CONFIG="${VERL_TRAINER_CONFIG:-$($TRAIN_VENV/bin/python -c \
    'from pathlib import Path; import verl; print(Path(verl.__file__).resolve().parent / "trainer" / "config")')}"
  SEPARATION_ARGS+=(
    "hydra.searchpath=[file://$VERL_TRAINER_CONFIG]"
    actor_rollout_ref.hybrid_engine=False
    rollout.nnodes=1
    rollout.n_gpus_per_node="$ROLLOUT_N_GPUS"
    ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS"
    +ray_kwargs.ray_init.include_dashboard="$RAY_INCLUDE_DASHBOARD"
    actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_BACKEND"
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$CHECKPOINT_BUCKET_MB"
  )
  [[ -n "$CHECKPOINT_CUSTOM_BACKEND_MODULE" ]] && \
    SEPARATION_ARGS+=(actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module="$CHECKPOINT_CUSTOM_BACKEND_MODULE")
  [[ -n "$RAY_WORKER_SETUP_HOOK" ]] && \
    SEPARATION_ARGS+=(+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook="$RAY_WORKER_SETUP_HOOK")
fi

python -m "$TRAIN_ENTRYPOINT" \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" data.val_files="$TRAIN_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" data.val_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
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
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.format="$TOOL_FORMAT" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$EXPERIMENT_DIR/config/agent_loop.yaml" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  reward.num_workers="$REWARD_WORKERS" \
  trainer.project_name=tau2_verl_grpo \
  trainer.experiment_name="${EXPERIMENT_NAME:-tau2_verl_grpo}" \
  trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.val_before_train=False trainer.save_freq="$SAVE_FREQ" trainer.test_freq=-1 \
  trainer.total_epochs=100 trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  'trainer.logger=["console"]' \
  "${LORA_ARGS[@]}" "${CHAT_TEMPLATE_ARGS[@]}" "${MODEL_CONFIG_ARGS[@]}" "${VLLM_ENGINE_ARGS[@]}" \
  "${SEPARATION_ARGS[@]}" "$@"

ADAPTER_PATH="$(find "$OUTPUT_DIR" -mindepth 3 -maxdepth 3 -type d -path '*/actor/lora_adapter' -print | sort -V | tail -n 1)"
if [[ -z "$ADAPTER_PATH" ]]; then
  latest_actor="$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d -path '*/actor' -print | sort -V | tail -n 1)"
  if [[ -n "$latest_actor" && -s "$latest_actor/model_world_size_1_rank_0.pt" ]]; then
    ADAPTER_PATH="$OUTPUT_DIR/final_adapter"
    python -m verl_code_rl.extract_sft_lora_adapter \
      --checkpoint-dir "$latest_actor" --base-model "$MODEL_PATH" \
      --output "$ADAPTER_PATH" --lora-rank "$LORA_RANK" \
      --lora-alpha "$LORA_ALPHA" --target-modules "$LORA_TARGET_MODULES"
  fi
fi
if [[ -z "$ADAPTER_PATH" || ! -s "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  echo "[train_verl_grpo] no valid LoRA adapter checkpoint under $OUTPUT_DIR" >&2
  exit 1
fi
printf '%s\n' "$ADAPTER_PATH" > "$OUTPUT_DIR/final_adapter_path.txt"
echo "[train_verl_grpo] final adapter: $ADAPTER_PATH"
