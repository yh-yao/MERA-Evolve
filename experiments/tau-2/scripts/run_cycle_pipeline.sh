#!/usr/bin/env bash
set -euo pipefail

# 4-cycle tau2 closed loop, mirroring humaneval_mbpp's run_full_pipeline.sh:
# each cycle does collect(probe-only fallback) -> skillbook -> SFT -> [GRPO
# if ENABLE_GRPO=1] -> reload agent server -> held-out eval, threading the
# LoRA adapter forward cycle-to-cycle (SFT continues the previous cycle's
# adapter; GRPO, if enabled, continues training that same adapter further
# within the cycle).
#
# Env vars (all required to be set explicitly by the caller for this
# overnight run -- no interactive prompts):
#   RESULTS_DIR, AGENT_GPU, AGENT_PORT, USER_GPU, USER_PORT, TRAIN_GPU,
#   ENABLE_GRPO, N_CYCLES
# GRPO_GPU remains a backward-compatible alias for TRAIN_GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-/shared_home/yuhang.yao/router-skills-evolve}"
TAU2_PY="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
TRAIN_VENV="${TRAIN_VENV:-$ROOT/venv}"
MERA_PY="${MERA_PYTHON:-$TRAIN_VENV/bin/python3}"
VLLM_BIN="${VLLM_BIN:-$TRAIN_VENV/bin/vllm}"
export TAU2_WORKSPACE
export PYTHONPATH="$EXPERIMENT_DIR:$TRAIN_VENV/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "$ROOT"

RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR}"
case "$RESULTS_DIR" in
  /*) ;;
  *) echo "FATAL: RESULTS_DIR must be an absolute path (got: $RESULTS_DIR)" >&2; exit 2 ;;
esac
AGENT_GPU="${AGENT_GPU:?set AGENT_GPU}"
AGENT_PORT="${AGENT_PORT:?set AGENT_PORT}"
USER_GPU="${USER_GPU:?set USER_GPU}"
USER_PORT="${USER_PORT:?set USER_PORT}"
ENABLE_GRPO="${ENABLE_GRPO:-0}"
TRAIN_GPU="${TRAIN_GPU:-${GRPO_GPU:-}}"
TRAIN_GPU="${TRAIN_GPU:?set TRAIN_GPU (the GPU used for SFT and optional GRPO)}"
ROLLOUT_GPU="${ROLLOUT_GPU:-}"
N_CYCLES="${N_CYCLES:-4}"
START_CYCLE="${START_CYCLE:-0}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-}"
INITIAL_SKILLBOOK="${INITIAL_SKILLBOOK:-}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
USER_MODEL_PATH="${USER_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DISTILLER_MODEL="${DISTILLER_MODEL:-openai/gpt-5.5}"
DISTILLER_BASE_URL="${DISTILLER_BASE_URL:-https://api.commonstack.ai/v1}"
DISTILLER_API_KEY="${DISTILLER_API_KEY:-}"
FALLBACK_AGENT_MODEL="${FALLBACK_AGENT_MODEL:-openai/evol-llm-user}"
FALLBACK_AGENT_BASE_URL="${FALLBACK_AGENT_BASE_URL:-}"
FALLBACK_ATTEMPTS="${FALLBACK_ATTEMPTS:-1}"
COLLECT_MAX_STEPS="${COLLECT_MAX_STEPS:-40}"
COLLECT_WORKERS="${COLLECT_WORKERS:-96}"
EVAL_WORKERS="${EVAL_WORKERS:-$COLLECT_WORKERS}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-}"
AGENT_MAX_MODEL_LEN="${AGENT_MAX_MODEL_LEN:-16384}"
USER_MAX_MODEL_LEN="${USER_MAX_MODEL_LEN:-32768}"
GRPO_TOTAL_STEPS="${GRPO_TOTAL_STEPS:-10}"
GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-8}"
GRPO_N_GENERATIONS="${GRPO_N_GENERATIONS:-4}"
GRPO_ACTOR_LR="${GRPO_ACTOR_LR:-5e-6}"
GRPO_MAX_RESPONSE_LENGTH="${GRPO_MAX_RESPONSE_LENGTH:-2048}"
REUSE_EXISTING_ARTIFACTS="${REUSE_EXISTING_ARTIFACTS:-0}"
SFT_MAX_LENGTH_DEFAULT=12288

MODEL_ARGS=()
THINKING_ARGS=()
SFT_ARGS=()
GRPO_ARGS=()
AGENT_TOKEN_ARGS=()
[[ -n "$AGENT_MAX_TOKENS" ]] && AGENT_TOKEN_ARGS=(--agent-max-tokens "$AGENT_MAX_TOKENS")
if [[ "$BASE_MODEL" == *"Qwen3.5"* ]]; then
  SFT_MAX_LENGTH_DEFAULT=24576
  export PYTHONPATH="$EXPERIMENT_DIR/compat/qwen35_torch_fallback:$PYTHONPATH"
  export QWEN35_TRAIN_TRITON_OVERLAY="${QWEN35_TRAIN_TRITON_OVERLAY:-$ROOT/.deps/qwen35-triton33}"
  MODEL_ARGS=(--language-model-only --gdn-prefill-backend triton)
  THINKING_ARGS=(--no-agent-thinking --no-user-thinking)
  SFT_ARGS=(
    PYTHONPATH="$QWEN35_TRAIN_TRITON_OVERLAY:$PYTHONPATH"
    FLA_TILELANG=0
    SFT_ENABLE_THINKING=False
    SFT_ATTN_IMPLEMENTATION=sdpa
    SFT_USE_TORCH_COMPILE=False
    SFT_DATASET_PATH="$EXPERIMENT_DIR/tau2_evolve/sft_dataset.py"
    SFT_DATASET_NAME=Qwen35MultiTurnSFTDataset
    # Keep one long tau2 conversation per dynamic microbatch. Packing two
    # Qwen3.5 GDN sequences caused an intermittent Triton backward hang and
    # eventual CUDA launch failure after a reshuffle in the second epoch.
    SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-14000}"
    # Qwen3.5's GDN kernels can still fail asynchronously; retain enough state
    # to resume without discarding an entire multi-epoch SFT run.
    SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-3}"
  )
  GRPO_ARGS=(
    TRAIN_VENV="$TRAIN_VENV"
    AGENT_THINKING=False
    MODEL_ATTN_IMPLEMENTATION=sdpa
    USE_REMOVE_PADDING=False
    ROLLOUT_TEMPERATURE=1.0
    ROLLOUT_TOP_P=0.98
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.70
    VLLM_GDN_PREFILL_BACKEND=triton
    VLLM_LANGUAGE_MODEL_ONLY=True
  )
fi

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/pipeline.log"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
DISTILLER_API_KEY="${DISTILLER_API_KEY:-${COMMONSTACK_API_KEY:-}}"

if ! [[ "$START_CYCLE" =~ ^[0-9]+$ ]] || (( START_CYCLE >= N_CYCLES )); then
  log "FATAL: START_CYCLE must be an integer in [0, $((N_CYCLES - 1))] (got: $START_CYCLE)" >&2
  exit 2
fi
for worker_setting in COLLECT_WORKERS EVAL_WORKERS; do
  worker_count="${!worker_setting}"
  if ! [[ "$worker_count" =~ ^[1-9][0-9]*$ ]]; then
    log "FATAL: $worker_setting must be a positive integer (got: $worker_count)" >&2
    exit 2
  fi
done
[[ -z "$INITIAL_ADAPTER" || -s "$INITIAL_ADAPTER/adapter_model.safetensors" ]] || {
  log "FATAL: INITIAL_ADAPTER is not a loadable LoRA adapter: $INITIAL_ADAPTER" >&2
  exit 2
}
[[ -z "$INITIAL_SKILLBOOK" || -s "$INITIAL_SKILLBOOK" ]] || {
  log "FATAL: INITIAL_SKILLBOOK does not exist: $INITIAL_SKILLBOOK" >&2
  exit 2
}

log "=== starting: RESULTS_DIR=$RESULTS_DIR ENABLE_GRPO=$ENABLE_GRPO N_CYCLES=$N_CYCLES START_CYCLE=$START_CYCLE ==="

# ---- launch agent + user servers (idempotent: skip if already up) ----
# (separate agent/user variants rather than a generic extra-env parameter --
# `eval "$extra_env"` would only set a local shell var, not export it to the
# nohup'd child, so a generic env-string-passthrough silently wouldn't work)
start_agent_server() {
  local gpu="$1" port="$2"
  if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "evol-llm-agent"; then
    log "agent server already up on port $port"
    return 0
  fi
  log "starting agent server (base, no adapter) on GPU $gpu port $port"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
    nohup "$VLLM_BIN" serve "$BASE_MODEL" \
    --served-model-name evol-llm-agent \
    --enable-lora --max-lora-rank 16 \
    --port "$port" --gpu-memory-utilization 0.5 --max-model-len "$AGENT_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$AGENT_MAX_MODEL_LEN" --enable-prefix-caching \
    --dtype bfloat16 --trust-remote-code --enforce-eager \
    "${MODEL_ARGS[@]}" \
    --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-hermes}" \
    > "$RESULTS_DIR/evol-llm-agent_server.log" 2>&1 &
  disown
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "evol-llm-agent"; then
      log "agent server ready"
      return 0
    fi
    sleep 5
  done
  log "FATAL: agent server did not become ready" >&2
  exit 1
}

start_user_server() {
  local gpu="$1" port="$2"
  if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "evol-llm-user"; then
    log "user server already up on port $port"
    return 0
  fi
  log "starting user server on GPU $gpu port $port"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "$VLLM_BIN" serve "$USER_MODEL_PATH" \
    --served-model-name evol-llm-user \
    --port "$port" --gpu-memory-utilization 0.5 --max-model-len "$USER_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$USER_MAX_MODEL_LEN" --enable-prefix-caching \
    --dtype bfloat16 --trust-remote-code --enforce-eager \
    "${MODEL_ARGS[@]}" \
    --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-hermes}" \
    > "$RESULTS_DIR/evol-llm-user_server.log" 2>&1 &
  disown
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "evol-llm-user"; then
      log "user server ready"
      return 0
    fi
    sleep 5
  done
  log "FATAL: user server did not become ready" >&2
  exit 1
}

start_agent_server "$AGENT_GPU" "$AGENT_PORT"
start_user_server "$USER_GPU" "$USER_PORT"

AGENT_BASE_URL="http://127.0.0.1:$AGENT_PORT/v1"
USER_BASE_URL="http://127.0.0.1:$USER_PORT/v1"
DISTILLER_BASE_URL="${DISTILLER_BASE_URL:-$USER_BASE_URL}"
FALLBACK_AGENT_BASE_URL="${FALLBACK_AGENT_BASE_URL:-$USER_BASE_URL}"

CURRENT_ADAPTER="$INITIAL_ADAPTER"  # empty == base model, no adapter yet

# Always use vLLM's dynamic load/unload REST API (server started once, bare,
# with --enable-lora --max-lora-rank 16, no --lora-modules) rather than
# restarting the whole server per cycle/step -- mixing a startup-time
# --lora-modules registration with runtime LoRA updates
  # /v1/load_lora_adapter calls for the SAME served name risks a load/unload
  # conflict. The external server is used for collection/eval; verl owns its
  # separate training rollout engine and exports the final LoRA adapter, which
  # is loaded here through the dynamic API.
reload_agent() {
  local adapter_dir
  adapter_dir="$(cd "$1" && pwd)"  # vLLM needs an absolute path
  log "hot-swapping agent adapter: $adapter_dir"
  curl -sf -X POST "$AGENT_BASE_URL/unload_lora_adapter" \
    -H 'Content-Type: application/json' -d '{"lora_name": "evol-llm-agent"}' >/dev/null 2>&1 || true
  if ! curl -sf -X POST "$AGENT_BASE_URL/load_lora_adapter" \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"evol-llm-agent\", \"lora_path\": \"$adapter_dir\"}"; then
    log "FATAL: failed to hot-swap adapter $adapter_dir into the agent server" >&2
    exit 1
  fi
  log "agent adapter hot-swapped"
}

if [[ -n "$CURRENT_ADAPTER" ]]; then
  reload_agent "$CURRENT_ADAPTER"
fi

PREV_SKILLBOOK="$INITIAL_SKILLBOOK"

for ((cycle=START_CYCLE; cycle<N_CYCLES; cycle++)); do
  OUT="$RESULTS_DIR/cycle_$cycle"
  mkdir -p "$OUT"
  log "== cycle $cycle =="

  # 1. Collect TRAIN traces with probe-only fallback. By default, the larger
  # local user model serves both fallback roles; hosted fallback is opt-in.
  # Uses the PREVIOUS cycle's skillbook (like humaneval_mbpp's run_full_pipeline.sh) so the
  # compounding effect shows up in what gets collected, not just at final eval.
  SKILL_ARG=()
  [[ -n "$PREV_SKILLBOOK" ]] && SKILL_ARG=(--skillbook "$PREV_SKILLBOOK")
  if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -f "$OUT/train_traces.jsonl" && "$(wc -l < "$OUT/train_traces.jsonl")" -eq 97 ]]; then
    log "reusing complete TRAIN traces: $OUT/train_traces.jsonl"
  else
    log "collecting TRAIN traces (skillbook=${PREV_SKILLBOOK:-none})"
    "$TAU2_PY" -m tau2_evolve.collect_traces \
      --bucket TRAIN --workers "$COLLECT_WORKERS" --max-steps "$COLLECT_MAX_STEPS" --probe-only --resume \
      --agent-model "openai/evol-llm-agent" --agent-base-url "$AGENT_BASE_URL" \
      --user-model "openai/evol-llm-user" --user-base-url "$USER_BASE_URL" \
      --fallback-agent-model "$FALLBACK_AGENT_MODEL" \
      --fallback-agent-base-url "$FALLBACK_AGENT_BASE_URL" \
      --fallback-attempts "$FALLBACK_ATTEMPTS" \
      "${AGENT_TOKEN_ARGS[@]}" \
      "${THINKING_ARGS[@]}" \
      "${SKILL_ARG[@]}" \
      --out "$OUT/train_traces.jsonl" >> "$LOG" 2>&1
  fi

  # 2. skillbook
  if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -s "$OUT/skillbook.json" ]]; then
    log "reusing skillbook: $OUT/skillbook.json"
  else
    log "building skillbook"
    "$TAU2_PY" -m tau2_evolve.build_skillbook \
      --traces "$OUT/train_traces.jsonl" --output "$OUT/skillbook.json" \
      --distiller-model "$DISTILLER_MODEL" --distiller-base-url "$DISTILLER_BASE_URL" \
      --api-key "$DISTILLER_API_KEY" >> "$LOG" 2>&1
  fi

  # 3. SFT (LoRA continuation from previous cycle's adapter, if any)
  if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -s "$OUT/sft_pairs.parquet" ]]; then
    log "reusing SFT pairs: $OUT/sft_pairs.parquet"
  else
    log "building SFT pairs"
    "$MERA_PY" -m tau2_evolve.traces_to_sft \
      --traces "$OUT/train_traces.jsonl" --output "$OUT/sft_pairs.parquet" >> "$LOG" 2>&1
  fi

  SFT_ADAPTER="$OUT/sft_adapter"
  if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -s "$SFT_ADAPTER/adapter_model.safetensors" ]]; then
    log "reusing SFT adapter: $SFT_ADAPTER"
    CURRENT_ADAPTER="$SFT_ADAPTER"
  elif [[ -s "$OUT/sft_pairs.parquet" ]]; then
    log "training SFT (init_adapter=${CURRENT_ADAPTER:-none})"
    LORA_ARG=()
    [[ -n "$CURRENT_ADAPTER" ]] && LORA_ARG=(LORA_ADAPTER_PATH="$CURRENT_ADAPTER")
    # train_sft.sh does its own `cd "$(dirname "$0")/.."` to MERA-Evolve root,
    # so SFT_DATA/SFT_OUTPUT_DIR must be absolute (OUT already is, since
    # RESULTS_DIR is enforced absolute above) -- NOT re-prefixed with
    # experiments/tau-2/, which would double up into a broken path.
    # Must activate MERA-Evolve's own venv here -- without it, torchrun
    # silently resolves to system python (no `verl` installed), which
    # crashed both overnight pipelines identically on the first attempt
    # (ModuleNotFoundError: No module named 'verl', caught by set -e).
    ( cd "$ROOT" && source "$TRAIN_VENV/bin/activate" && \
      env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
      PYTHON="$MERA_PY" \
      SFT_DATA="$OUT/sft_pairs.parquet" \
      SFT_OUTPUT_DIR="$SFT_ADAPTER" \
      MODEL_PATH="$BASE_MODEL" \
      "${LORA_ARG[@]}" \
      SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}" \
      SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-1}" \
      SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}" \
      SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-$SFT_MAX_LENGTH_DEFAULT}" \
      "${SFT_ARGS[@]}" \
      SFT_PROJECT_NAME=tau2_4cycle SFT_EXPERIMENT_NAME="cycle${cycle}_$(basename "$RESULTS_DIR")" \
      bash scripts/train_sft.sh data.num_workers=0 ) >> "$LOG" 2>&1
    CURRENT_ADAPTER="$SFT_ADAPTER"
  else
    log "no SFT pairs this cycle (no successful trajectories) -- skipping SFT"
  fi

  # 4. GRPO (continues training CURRENT_ADAPTER further, only if enabled)
  if [[ "$ENABLE_GRPO" == "1" && -n "$CURRENT_ADAPTER" ]]; then
    GRPO_OUT="$OUT/verl_grpo"
    VERL_DATA="$OUT/verl_grpo.parquet"
    REUSED_GRPO_ADAPTER=""
    if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -s "$GRPO_OUT/final_adapter_path.txt" ]]; then
      REUSED_GRPO_ADAPTER="$(<"$GRPO_OUT/final_adapter_path.txt")"
      [[ -s "$REUSED_GRPO_ADAPTER/adapter_model.safetensors" ]] || REUSED_GRPO_ADAPTER=""
    fi
    if [[ -n "$REUSED_GRPO_ADAPTER" ]]; then
      CURRENT_ADAPTER="$REUSED_GRPO_ADAPTER"
      log "reusing GRPO adapter: $CURRENT_ADAPTER"
    else
      log "GRPO training (continuing from $CURRENT_ADAPTER)"
      reload_agent "$CURRENT_ADAPTER"  # keep the external eval server in sync
      "$MERA_PY" -m tau2_evolve.prepare_grpo_data \
        --traces "$OUT/train_traces.jsonl" --skillbook "$OUT/skillbook.json" \
        --user-base-url "$USER_BASE_URL" "${THINKING_ARGS[@]:1}" \
        --output "$VERL_DATA" >> "$LOG" 2>&1
      CUDA_DEVICES="$TRAIN_GPU"
      SEPARATION_ARGS=()
      if [[ -n "$ROLLOUT_GPU" ]]; then
        CUDA_DEVICES="$TRAIN_GPU,$ROLLOUT_GPU"
        SEPARATION_ARGS=(
          SEPARATE_ROLLOUT=1 ROLLOUT_N_GPUS=1 RAY_NUM_CPUS=24
          CHECKPOINT_BACKEND=nixl_tau2
          CHECKPOINT_CUSTOM_BACKEND_MODULE=tau2_evolve.nixl_checkpoint
        )
      fi
      ( cd "$ROOT" && \
        env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" TRAIN_FILE="$VERL_DATA" \
        OUTPUT_DIR="$GRPO_OUT" MODEL_PATH="$BASE_MODEL" \
        LORA_ADAPTER_PATH="$CURRENT_ADAPTER" TOTAL_STEPS="$GRPO_TOTAL_STEPS" \
        TRAIN_BATCH_SIZE="$GRPO_TRAIN_BATCH_SIZE" N_GENERATIONS="$GRPO_N_GENERATIONS" \
        ACTOR_LR="$GRPO_ACTOR_LR" MAX_RESPONSE_LENGTH="$GRPO_MAX_RESPONSE_LENGTH" \
        "${GRPO_ARGS[@]}" "${SEPARATION_ARGS[@]}" \
        EXPERIMENT_NAME="cycle${cycle}_$(basename "$RESULTS_DIR")" \
        bash experiments/tau-2/scripts/train_verl_grpo.sh data.shuffle=False ) >> "$LOG" 2>&1
      CURRENT_ADAPTER="$(<"$GRPO_OUT/final_adapter_path.txt")"
      if [[ ! -s "$CURRENT_ADAPTER/adapter_model.safetensors" ]]; then
        log "FATAL: verl GRPO adapter is missing: $CURRENT_ADAPTER"
        exit 1
      fi
    fi
  fi

  # 5. reload agent with this cycle's final adapter, then held-out eval
  if [[ -n "$CURRENT_ADAPTER" ]]; then
    reload_agent "$CURRENT_ADAPTER"
  fi
  if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -f "$OUT/eval.jsonl" && "$(wc -l < "$OUT/eval.jsonl")" -eq 35 ]]; then
    log "reusing complete held-out EVAL: $OUT/eval.jsonl"
  else
    log "held-out EVAL (35 tasks)"
    "$TAU2_PY" -m tau2_evolve.collect_traces \
      --bucket EVAL --workers "$EVAL_WORKERS" --max-steps 60 --resume \
      --agent-model "openai/evol-llm-agent" --agent-base-url "$AGENT_BASE_URL" \
      --user-model "openai/evol-llm-user" --user-base-url "$USER_BASE_URL" \
      "${AGENT_TOKEN_ARGS[@]}" \
      "${THINKING_ARGS[@]}" \
      --skillbook "$OUT/skillbook.json" \
      --out "$OUT/eval.jsonl" >> "$LOG" 2>&1
  fi

  "$MERA_PY" - "$OUT/eval.jsonl" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
by_domain = defaultdict(list)
for r in rows:
    by_domain[r["domain"]].append(r["passed"])
total = sum(r["passed"] for r in rows)
print(f"[cycle summary] total: {total}/{len(rows)} = {total/len(rows):.1%}")
for d, vals in sorted(by_domain.items()):
    print(f"[cycle summary]   {d}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.1%}")
PYEOF

  PREV_SKILLBOOK="$OUT/skillbook.json"
done

log "=== done: $RESULTS_DIR ==="
