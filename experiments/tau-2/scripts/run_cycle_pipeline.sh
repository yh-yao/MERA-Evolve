#!/usr/bin/env bash
set -euo pipefail

# 4-cycle tau2 closed loop. Each cycle collects unbiased student traces,
# obtains verified teacher suffixes by replaying failed decision states (OPD),
# builds skills, runs SFT and optional GRPO, then learns an escalation router
# from fresh post-training trajectories before plain and routed evaluation.
# The LoRA adapter is threaded forward cycle-to-cycle.
#
# Env vars (all required to be set explicitly by the caller for this
# overnight run -- no interactive prompts):
#   RESULTS_DIR, AGENT_GPU, AGENT_PORT, USER_GPU, USER_PORT, TRAIN_GPU,
#   ENABLE_GRPO, N_CYCLES
# GRPO_GPU remains a backward-compatible alias for TRAIN_GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
TAU2_PY="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
TRAIN_VENV="${TRAIN_VENV:-$ROOT/venv}"
MERA_PY="${MERA_PYTHON:-$TRAIN_VENV/bin/python3}"
VLLM_BIN="${VLLM_BIN:-$TRAIN_VENV/bin/vllm}"
if [[ ! -x "$TAU2_PY" ]]; then
  echo "FATAL: tau2 Python not found at $TAU2_PY; set TAU2_WORKSPACE or TAU2_PYTHON" >&2
  exit 2
fi
if [[ ! -x "$MERA_PY" || ! -x "$VLLM_BIN" ]]; then
  echo "FATAL: training environment is incomplete at $TRAIN_VENV; set TRAIN_VENV" >&2
  exit 2
fi
TRAIN_SITE_PACKAGES="${TRAIN_SITE_PACKAGES:-$($MERA_PY -c 'import site; print(site.getsitepackages()[0])')}"
export TAU2_WORKSPACE
export PYTHONPATH="$EXPERIMENT_DIR:$TRAIN_SITE_PACKAGES:${PYTHONPATH:-}"
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
ENABLE_OPD="${ENABLE_OPD:-1}"
ENABLE_ROUTER="${ENABLE_ROUTER:-1}"
# Tau2 calls LiteLLM, where the first `openai/` selects the provider and the
# remainder is forwarded to CommonStack. CommonStack's model ID itself also
# contains `openai/`; the direct OpenAI-client distiller needs only one prefix.
OPD_TEACHER_MODEL="${OPD_TEACHER_MODEL:-openai/openai/gpt-5.5}"
OPD_TEACHER_BASE_URL="${OPD_TEACHER_BASE_URL:-$DISTILLER_BASE_URL}"
OPD_TEACHER_API_KEY="${OPD_TEACHER_API_KEY:-$DISTILLER_API_KEY}"
OPD_WORKERS="${OPD_WORKERS:-8}"
OPD_BRANCH_ATTEMPTS="${OPD_BRANCH_ATTEMPTS:-3}"
OPD_TEACHER_MAX_TOKENS="${OPD_TEACHER_MAX_TOKENS:-1024}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.5}"
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
SFT_MIN_ROWS="${SFT_MIN_ROWS:-12}"
SFT_LR="${SFT_LR:-1e-6}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
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
  export CUDA_HOME="${QWEN35_CUDA_HOME:-$ROOT/.deps/cuda-12.8}"
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "FATAL: Qwen3.5 requires the CUDA 12.8 toolkit at $CUDA_HOME" >&2
    exit 2
  fi
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
  TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
  MODEL_ARGS=(--language-model-only --gdn-prefill-backend "${EXTERNAL_GDN_PREFILL_BACKEND:-flashinfer}")
  THINKING_ARGS=(--no-agent-thinking --no-user-thinking)
  SFT_ARGS=(
    PYTHONPATH="$QWEN35_TRAIN_TRITON_OVERLAY:$PYTHONPATH"
    FLA_TILELANG=0
    SFT_ENABLE_THINKING=False
    SFT_ATTN_IMPLEMENTATION=sdpa
    SFT_USE_TORCH_COMPILE=False
    SFT_DATASET_PATH="$EXPERIMENT_DIR/tau2_evolve/sft_dataset.py"
    SFT_DATASET_NAME=Qwen35MultiTurnSFTDataset
    # This budget must cover the longest individual sequence before dynamic
    # batching can split the batch, but must not pack two long GDN sequences:
    # cycle OPD trajectories reach about 16k and a 24k packed microbatch OOMs
    # inside FLA's backward autotuner on an 80 GB H100.
    SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-16384}"
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
OPD_TEACHER_API_KEY="${OPD_TEACHER_API_KEY:-$DISTILLER_API_KEY}"

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

# Persist the effective non-secret configuration so a run can be reproduced
# on another machine without recovering settings from process listings.
"$MERA_PY" - "$RESULTS_DIR/config_snapshot.json" \
  "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
  "base_model=$BASE_MODEL" "user_model_path=$USER_MODEL_PATH" \
  "train_venv=$TRAIN_VENV" "tau2_workspace=$TAU2_WORKSPACE" \
  "agent_gpu=$AGENT_GPU" "agent_port=$AGENT_PORT" \
  "user_gpu=$USER_GPU" "user_port=$USER_PORT" \
  "train_gpu=$TRAIN_GPU" "rollout_gpu=$ROLLOUT_GPU" \
  "n_cycles=$N_CYCLES" "start_cycle=$START_CYCLE" \
  "enable_sft=1" "enable_grpo=$ENABLE_GRPO" \
  "enable_opd=$ENABLE_OPD" "enable_router=$ENABLE_ROUTER" \
  "distiller_model=$DISTILLER_MODEL" "opd_teacher_model=$OPD_TEACHER_MODEL" \
  "collect_workers=$COLLECT_WORKERS" "eval_workers=$EVAL_WORKERS" \
  "opd_workers=$OPD_WORKERS" "opd_branch_attempts=$OPD_BRANCH_ATTEMPTS" \
  "sft_lr=$SFT_LR" "sft_total_epochs=$SFT_TOTAL_EPOCHS" \
  "grpo_total_steps=$GRPO_TOTAL_STEPS" \
  "grpo_train_batch_size=$GRPO_TRAIN_BATCH_SIZE" \
  "grpo_n_generations=$GRPO_N_GENERATIONS" "grpo_actor_lr=$GRPO_ACTOR_LR" <<'PY'
import json
import sys

output = sys.argv[1]
config = dict(item.split("=", 1) for item in sys.argv[2:])
with open(output, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

log "=== starting: RESULTS_DIR=$RESULTS_DIR ENABLE_GRPO=$ENABLE_GRPO N_CYCLES=$N_CYCLES START_CYCLE=$START_CYCLE ==="

# ---- launch agent + user servers (idempotent: skip if already up) ----
# (separate agent/user variants rather than a generic extra-env parameter --
# `eval "$extra_env"` would only set a local shell var, not export it to the
# nohup'd child, so a generic env-string-passthrough silently wouldn't work)
start_agent_server() {
  local gpu="$1" port="$2"
  local serve_model="$BASE_MODEL"
  local cache_dir="$HOME/.cache/huggingface/hub/models--${BASE_MODEL//\//--}/snapshots"
  if [[ ! -d "$serve_model" && -d "$cache_dir" ]]; then
    serve_model="$(find "$cache_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  fi
  if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "evol-llm-agent"; then
    log "agent server already up on port $port"
    return 0
  fi
  log "starting agent server (base, no adapter) on GPU $gpu port $port"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
    nohup "$VLLM_BIN" serve "$serve_model" \
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
  local serve_model="$USER_MODEL_PATH"
  local cache_dir="$HOME/.cache/huggingface/hub/models--${USER_MODEL_PATH//\//--}/snapshots"
  if [[ ! -d "$serve_model" && -d "$cache_dir" ]]; then
    serve_model="$(find "$cache_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  fi
  if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "evol-llm-user"; then
    log "user server already up on port $port"
    return 0
  fi
  log "starting user server on GPU $gpu port $port"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "$VLLM_BIN" serve "$serve_model" \
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

  # 1. Collect unbiased student-only TRAIN traces. Teacher correction happens
  # from replayed student states below; independent full-task fallback would
  # hide the student's failure distribution and is not OPD.
  # Uses the PREVIOUS cycle's skillbook (like humaneval_mbpp's run_full_pipeline.sh) so the
  # compounding effect shows up in what gets collected, not just at final eval.
  SKILL_ARG=()
  [[ -n "$PREV_SKILLBOOK" ]] && SKILL_ARG=(--skillbook "$PREV_SKILLBOOK")
  if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -f "$OUT/train_traces.jsonl" && "$(wc -l < "$OUT/train_traces.jsonl")" -eq 97 ]]; then
    log "reusing complete TRAIN traces: $OUT/train_traces.jsonl"
  else
    log "collecting TRAIN traces (skillbook=${PREV_SKILLBOOK:-none})"
    "$TAU2_PY" -m tau2_evolve.collect_traces \
      --bucket TRAIN --workers "$COLLECT_WORKERS" --max-steps "$COLLECT_MAX_STEPS" --resume \
      --agent-model "openai/evol-llm-agent" --agent-base-url "$AGENT_BASE_URL" \
      --user-model "openai/evol-llm-user" --user-base-url "$USER_BASE_URL" \
      "${AGENT_TOKEN_ARGS[@]}" \
      "${THINKING_ARGS[@]}" \
      "${SKILL_ARG[@]}" \
      --out "$OUT/train_traces.jsonl" >> "$LOG" 2>&1
  fi

  # 1b. Replay failed student prefixes and let the teacher generate a verified
  # suffix. Router labels are counterfactual: student success -> continue;
  # verified teacher rescue -> escalate.
  OPD_TRACES="$OUT/opd_traces.jsonl"
  ROUTER_DATA="$OUT/router_data_pretrain.jsonl"
  if [[ "$ENABLE_OPD" == "1" ]]; then
    if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -f "$OPD_TRACES" && -f "$ROUTER_DATA" ]]; then
      log "reusing OPD traces and router data"
    else
      log "collecting verified OPD teacher suffixes"
      "$TAU2_PY" -m tau2_evolve.collect_opd \
        --traces "$OUT/train_traces.jsonl" --output "$OPD_TRACES" --router-data "$ROUTER_DATA" \
        --teacher-model "$OPD_TEACHER_MODEL" --teacher-base-url "$OPD_TEACHER_BASE_URL" \
        --teacher-api-key "$OPD_TEACHER_API_KEY" --teacher-max-tokens "$OPD_TEACHER_MAX_TOKENS" \
        --user-base-url "$USER_BASE_URL" --max-steps "$COLLECT_MAX_STEPS" \
        --branch-attempts "$OPD_BRANCH_ATTEMPTS" --workers "$OPD_WORKERS" \
        "${SKILL_ARG[@]}" >> "$LOG" 2>&1
    fi
  else
    : > "$OPD_TRACES"
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
    SFT_TRACE_ARGS=(--traces "$OUT/train_traces.jsonl")
    [[ -s "$OPD_TRACES" ]] && SFT_TRACE_ARGS+=(--traces "$OPD_TRACES")
    if (( cycle > START_CYCLE )); then
      PREV_OUT="$RESULTS_DIR/cycle_$((cycle - 1))"
      [[ -s "$PREV_OUT/train_traces.jsonl" ]] && SFT_TRACE_ARGS+=(--traces "$PREV_OUT/train_traces.jsonl")
      [[ -s "$PREV_OUT/opd_traces.jsonl" ]] && SFT_TRACE_ARGS+=(--traces "$PREV_OUT/opd_traces.jsonl")
    fi
    "$MERA_PY" -m tau2_evolve.traces_to_sft \
      "${SFT_TRACE_ARGS[@]}" --balance-domains --output "$OUT/sft_pairs.parquet" >> "$LOG" 2>&1
  fi

  SFT_ROWS="$($MERA_PY - "$OUT/sft_pairs.parquet" <<'PYEOF'
import pandas as pd, sys
print(len(pd.read_parquet(sys.argv[1])))
PYEOF
)"

  SFT_ADAPTER="$OUT/sft_adapter"
  if [[ "$REUSE_EXISTING_ARTIFACTS" == "1" && -s "$SFT_ADAPTER/adapter_model.safetensors" ]]; then
    log "reusing SFT adapter: $SFT_ADAPTER"
    CURRENT_ADAPTER="$SFT_ADAPTER"
  elif [[ -s "$OUT/sft_pairs.parquet" && "$SFT_ROWS" -ge "$SFT_MIN_ROWS" ]]; then
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
      SFT_LR="$SFT_LR" \
      SFT_TOTAL_EPOCHS="$SFT_TOTAL_EPOCHS" \
      SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-$SFT_MAX_LENGTH_DEFAULT}" \
      "${SFT_ARGS[@]}" \
      SFT_PROJECT_NAME=tau2_4cycle SFT_EXPERIMENT_NAME="cycle${cycle}_$(basename "$RESULTS_DIR")" \
      bash scripts/train_sft.sh data.num_workers=0 ) >> "$LOG" 2>&1
    CURRENT_ADAPTER="$SFT_ADAPTER"
  else
    log "SFT quality gate: rows=$SFT_ROWS minimum=$SFT_MIN_ROWS -- skipping SFT"
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

  # Train the router only after the policy update, using fresh trajectories
  # from the current checkpoint so labels do not describe a stale model.
  ROUTER_MODEL="$OUT/router.json"
  if [[ "$ENABLE_ROUTER" == "1" ]]; then
    log "post-train student rollout for router labels"
    "$TAU2_PY" -m tau2_evolve.collect_traces \
      --bucket TRAIN --workers "$COLLECT_WORKERS" --max-steps "$COLLECT_MAX_STEPS" --resume \
      --agent-model "openai/evol-llm-agent" --agent-base-url "$AGENT_BASE_URL" \
      --user-model "openai/evol-llm-user" --user-base-url "$USER_BASE_URL" \
      "${AGENT_TOKEN_ARGS[@]}" "${THINKING_ARGS[@]}" --skillbook "$OUT/skillbook.json" \
      --out "$OUT/system_train.jsonl" >> "$LOG" 2>&1
    "$TAU2_PY" -m tau2_evolve.collect_opd \
      --traces "$OUT/system_train.jsonl" --output "$OUT/router_teacher_rescues.jsonl" \
      --router-data "$OUT/router_data_posttrain.jsonl" \
      --teacher-model "$OPD_TEACHER_MODEL" --teacher-base-url "$OPD_TEACHER_BASE_URL" \
      --teacher-api-key "$OPD_TEACHER_API_KEY" --teacher-max-tokens "$OPD_TEACHER_MAX_TOKENS" \
      --user-base-url "$USER_BASE_URL" --max-steps "$COLLECT_MAX_STEPS" \
      --branch-attempts "$OPD_BRANCH_ATTEMPTS" --workers "$OPD_WORKERS" \
      --skillbook "$OUT/skillbook.json" >> "$LOG" 2>&1
    ROUTER_LABELS="$($MERA_PY - "$OUT/router_data_posttrain.jsonl" <<'PYEOF'
import json, sys
labels = {json.loads(line)["label"] for line in open(sys.argv[1]) if line.strip()}
print(len(labels))
PYEOF
)"
    if [[ "$ROUTER_LABELS" -eq 2 ]]; then
      "$MERA_PY" -m tau2_evolve.train_router \
        --data "$OUT/router_data_posttrain.jsonl" --output "$ROUTER_MODEL" \
        --threshold "$ROUTER_THRESHOLD" >> "$LOG" 2>&1
    else
      log "router quality gate: need both labels, found $ROUTER_LABELS -- skipping router"
    fi
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

  if [[ -s "$ROUTER_MODEL" ]]; then
    log "held-out routed system EVAL (35 tasks)"
    "$TAU2_PY" -m tau2_evolve.collect_traces \
      --bucket EVAL --workers "$EVAL_WORKERS" --max-steps 60 --resume \
      --agent-model "openai/evol-llm-agent" --agent-base-url "$AGENT_BASE_URL" \
      --user-model "openai/evol-llm-user" --user-base-url "$USER_BASE_URL" \
      --router-model "$ROUTER_MODEL" --router-teacher-model "$OPD_TEACHER_MODEL" \
      --router-teacher-base-url "$OPD_TEACHER_BASE_URL" \
      --router-teacher-api-key "$OPD_TEACHER_API_KEY" \
      --router-teacher-max-tokens "$OPD_TEACHER_MAX_TOKENS" \
      "${AGENT_TOKEN_ARGS[@]}" "${THINKING_ARGS[@]}" \
      --skillbook "$OUT/skillbook.json" --out "$OUT/eval_routed.jsonl" >> "$LOG" 2>&1
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
