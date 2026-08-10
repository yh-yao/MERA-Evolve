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
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRAINING_SEED="${TRAINING_SEED:-1}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-}"
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-}"
QWEN35_ENABLE_VERL_PATCHES="${QWEN35_ENABLE_VERL_PATCHES:-0}"
[[ "$MODEL_PATH" == *"Qwen3.5"* ]] && QWEN35_ENABLE_VERL_PATCHES=1
SEPARATE_ROLLOUT="${SEPARATE_ROLLOUT:-0}"
ROLLOUT_N_GPUS="${ROLLOUT_N_GPUS:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-24}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
CHECKPOINT_BUCKET_MB="${CHECKPOINT_BUCKET_MB:-256}"
CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-nccl}"
CHECKPOINT_CUSTOM_BACKEND_MODULE="${CHECKPOINT_CUSTOM_BACKEND_MODULE:-}"
RAY_WORKER_SETUP_HOOK="${RAY_WORKER_SETUP_HOOK:-}"

# This node's local CUDA toolchain reproduces `torch.AcceleratorError: CUDA
# error: unspecified/illegal memory access` under vLLM CUDA graphs once a
# rollout engine has been running for a while (see serve_vllm.sh's identical
# gotcha for the standalone server) -- keep verl's internal rollout engine in
# eager mode too, unless a fixed vLLM version is available later.
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"

# LoRA is the default training mode, matching router-skills-evolve's HumanEval
# GRPO algorithm (frozen base + trained adapter, r=16). Set LORA_RANK=0 to fall
# back to full-parameter fine-tuning.
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
# Path to a previous cycle's adapter to continue training, keeping MODEL_PATH
# as the original base checkpoint (verl loads base + this adapter together).
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"

if [[ "$MODEL_PATH" == *"Qwen3.5"* && -n "$LORA_ADAPTER_PATH" ]]; then
  TRAINING_ADAPTER_PATH="${OUTPUT_DIR:-$PWD/results/$EXPERIMENT_NAME}/training_init_adapter"
  "$PYTHON" -m verl_code_rl.prepare_qwen35_training_adapter \
    --input "$LORA_ADAPTER_PATH" --output "$TRAINING_ADAPTER_PATH"
  LORA_ADAPTER_PATH="$TRAINING_ADAPTER_PATH"
fi

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
MODEL_ARGS=()
VLLM_ENGINE_ARGS=()
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
[[ -n "$MODEL_ATTN_IMPLEMENTATION" ]] && \
  MODEL_ARGS+=(+actor_rollout_ref.model.override_config.attn_implementation="$MODEL_ATTN_IMPLEMENTATION")
[[ -n "$VLLM_GDN_PREFILL_BACKEND" ]] && \
  VLLM_ENGINE_ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend="$VLLM_GDN_PREFILL_BACKEND")
[[ -n "$VLLM_LANGUAGE_MODEL_ONLY" ]] && \
  VLLM_ENGINE_ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only="$VLLM_LANGUAGE_MODEL_ONLY")

TRAIN_ENTRYPOINT="verl.trainer.main_ppo"
SEPARATION_ARGS=()
if [[ "$SEPARATE_ROLLOUT" == "1" ]]; then
  TRAIN_ENTRYPOINT="verl.experimental.one_step_off_policy.main_ppo"
  VERL_TRAINER_CONFIG="${VERL_TRAINER_CONFIG:-$($PYTHON -c \
    'from pathlib import Path; import verl; print(Path(verl.__file__).resolve().parent / "trainer" / "config")')}"
  SEPARATION_ARGS=(
    "hydra.searchpath=[file://$VERL_TRAINER_CONFIG]"
    data.return_raw_chat=True
    actor_rollout_ref.hybrid_engine=False
    rollout.nnodes=1
    rollout.n_gpus_per_node="$ROLLOUT_N_GPUS"
    ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS"
    +ray_kwargs.ray_init.include_dashboard="$RAY_INCLUDE_DASHBOARD"
    actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_BACKEND"
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$CHECKPOINT_BUCKET_MB"
    reward.custom_reward_function.path="$REWARD_FILE"
    reward.custom_reward_function.name=compute_score
  )
  [[ -n "$CHECKPOINT_CUSTOM_BACKEND_MODULE" ]] && \
    SEPARATION_ARGS+=(actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module="$CHECKPOINT_CUSTOM_BACKEND_MODULE")
  [[ -n "$RAY_WORKER_SETUP_HOOK" ]] && \
    SEPARATION_ARGS+=(+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook="$RAY_WORKER_SETUP_HOOK")

  if [[ "$MODEL_PATH" == *"Qwen3.5"* ]]; then
    : "${QWEN35_TRAIN_TRITON_OVERLAY:?set QWEN35_TRAIN_TRITON_OVERLAY for separated Qwen3.5 GRPO}"
    [[ -f "$QWEN35_TRAIN_TRITON_OVERLAY/triton/__init__.py" ]] || {
      echo "[train_grpo] invalid Qwen3.5 Triton overlay: $QWEN35_TRAIN_TRITON_OVERLAY" >&2
      exit 2
    }
  fi
fi

set -x
QWEN35_ENABLE_VERL_PATCHES="$QWEN35_ENABLE_VERL_PATCHES" "$PYTHON" -m "$TRAIN_ENTRYPOINT" \
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
  data.seed="$TRAINING_SEED" \
  custom_reward_function.path="$REWARD_FILE" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.data_loader_seed="$TRAINING_SEED" \
  actor_rollout_ref.actor.fsdp_config.seed="$TRAINING_SEED" \
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
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.seed="$TRAINING_SEED" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.critic_warmup=0 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  'trainer.logger=["console"]' \
  ${OUTPUT_DIR:+trainer.default_local_dir="$OUTPUT_DIR"} \
  "${LORA_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${VLLM_ENGINE_ARGS[@]}" \
  "${SEPARATION_ARGS[@]}" \
  "$@"
set +x

# Depending on the verl execution path, PPO may save either a ready-to-serve
# adapter or only an FSDP LoRA shard. Normalize both formats so the next cycle
# always receives a PEFT directory that vLLM can load.
if [[ "$LORA_RANK" -gt 0 && -n "$OUTPUT_DIR" ]]; then
  ADAPTER_PATH="$(find "$OUTPUT_DIR" -mindepth 3 -maxdepth 3 -type d -path '*/actor/lora_adapter' -print 2>/dev/null | sort -V | tail -1)"
  if [[ -z "$ADAPTER_PATH" ]]; then
    latest_actor="$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d -path '*/actor' -print 2>/dev/null | sort -V | tail -1)"
    if [[ -n "$latest_actor" && -s "$latest_actor/model_world_size_1_rank_0.pt" ]]; then
      ADAPTER_PATH="$OUTPUT_DIR/final_adapter"
      echo "[train_grpo] extracting LoRA adapter from $latest_actor"
      "$PYTHON" -m verl_code_rl.extract_sft_lora_adapter \
        --checkpoint-dir "$latest_actor" \
        --base-model "$MODEL_PATH" \
        --output "$ADAPTER_PATH" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --target-modules "$LORA_TARGET_MODULES"
    fi
  fi

  if [[ -z "$ADAPTER_PATH" || ! -s "$ADAPTER_PATH/adapter_config.json" || ! -s "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
    echo "[train_grpo] no valid LoRA adapter checkpoint under $OUTPUT_DIR" >&2
    exit 1
  fi
  printf '%s\n' "$ADAPTER_PATH" > "$OUTPUT_DIR/final_adapter_path.txt"
  echo "[train_grpo] final adapter: $ADAPTER_PATH"
fi
