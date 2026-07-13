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
#   RESULTS_DIR, AGENT_GPU, AGENT_PORT, USER_GPU, USER_PORT, ENABLE_GRPO,
#   GRPO_GPU (only if ENABLE_GRPO=1), N_CYCLES

cd "$(dirname "$0")"
export PYTHONPATH="/shared_home/yuhang.yao/MERA-Evolve/venv/lib/python3.12/site-packages"
TAU2_PY="/shared_home/yuhang.yao/router-skills-evolve/.venv_tau2/bin/python3"
MERA_PY="/shared_home/yuhang.yao/MERA-Evolve/venv/bin/python3"

RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR}"
case "$RESULTS_DIR" in
  /*) ;;
  *) echo "FATAL: RESULTS_DIR must be an absolute path (got: $RESULTS_DIR) -- this script cd's into" \
          "experiments/tau-2/ first, so a relative path would land in the wrong place." >&2; exit 2 ;;
esac
AGENT_GPU="${AGENT_GPU:?set AGENT_GPU}"
AGENT_PORT="${AGENT_PORT:?set AGENT_PORT}"
USER_GPU="${USER_GPU:?set USER_GPU}"
USER_PORT="${USER_PORT:?set USER_PORT}"
ENABLE_GRPO="${ENABLE_GRPO:-0}"
GRPO_GPU="${GRPO_GPU:-}"
N_CYCLES="${N_CYCLES:-4}"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
USER_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
DISTILLER_MODEL="openai/gpt-5.5"
DISTILLER_BASE_URL="https://api.commonstack.ai/v1"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/pipeline.log"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

set -a; source /shared_home/yuhang.yao/MERA-Evolve/.env; set +a

log "=== starting: RESULTS_DIR=$RESULTS_DIR ENABLE_GRPO=$ENABLE_GRPO N_CYCLES=$N_CYCLES ==="

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
    nohup /shared_home/yuhang.yao/MERA-Evolve/venv/bin/vllm serve "$BASE_MODEL" \
    --served-model-name evol-llm-agent \
    --enable-lora --max-lora-rank 16 \
    --port "$port" --gpu-memory-utilization 0.5 --max-model-len 16384 \
    --dtype bfloat16 --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser hermes \
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
    nohup /shared_home/yuhang.yao/MERA-Evolve/venv/bin/vllm serve "$USER_MODEL_PATH" \
    --served-model-name evol-llm-user \
    --port "$port" --gpu-memory-utilization 0.5 --max-model-len 16384 \
    --dtype bfloat16 --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser hermes \
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

CURRENT_ADAPTER=""  # empty == base model, no adapter yet

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

PREV_SKILLBOOK=""

for ((cycle=0; cycle<N_CYCLES; cycle++)); do
  OUT="$RESULTS_DIR/cycle_$cycle"
  mkdir -p "$OUT"
  log "== cycle $cycle =="

  # 1. collect TRAIN traces with probe-only fallback (agent=current policy, user=3B; fallback=3B+3B).
  # Uses the PREVIOUS cycle's skillbook (like humaneval_mbpp's run_full_pipeline.sh) so the
  # compounding effect shows up in what gets collected, not just at final eval.
  SKILL_ARG=()
  [[ -n "$PREV_SKILLBOOK" ]] && SKILL_ARG=(--skillbook "$PREV_SKILLBOOK")
  log "collecting TRAIN traces (skillbook=${PREV_SKILLBOOK:-none})"
  "$TAU2_PY" collect_traces.py \
    --bucket TRAIN --workers 6 --max-steps 60 --probe-only \
    --agent-model "openai/evol-llm-agent" --agent-base-url "$AGENT_BASE_URL" \
    --user-model "openai/evol-llm-user" --user-base-url "$USER_BASE_URL" \
    "${SKILL_ARG[@]}" \
    --out "$OUT/train_traces.jsonl" >> "$LOG" 2>&1

  # 2. skillbook
  log "building skillbook"
  "$TAU2_PY" build_skillbook.py \
    --traces "$OUT/train_traces.jsonl" --output "$OUT/skillbook.json" \
    --distiller-model "$DISTILLER_MODEL" --distiller-base-url "$DISTILLER_BASE_URL" \
    --api-key "$COMMONSTACK_API_KEY" >> "$LOG" 2>&1

  # 3. SFT (LoRA continuation from previous cycle's adapter, if any)
  log "building SFT pairs"
  "$MERA_PY" traces_to_sft.py --traces "$OUT/train_traces.jsonl" --output "$OUT/sft_pairs.parquet" >> "$LOG" 2>&1

  SFT_ADAPTER="$OUT/sft_adapter"
  if [[ -s "$OUT/sft_pairs.parquet" ]]; then
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
    ( cd /shared_home/yuhang.yao/MERA-Evolve && source venv/bin/activate && \
      env CUDA_VISIBLE_DEVICES="${GRPO_GPU:-$USER_GPU}" \
      SFT_DATA="$OUT/sft_pairs.parquet" \
      SFT_OUTPUT_DIR="$SFT_ADAPTER" \
      MODEL_PATH="$BASE_MODEL" \
      "${LORA_ARG[@]}" \
      SFT_BATCH_SIZE=4 SFT_MICRO_BATCH_SIZE_PER_GPU=2 SFT_TOTAL_EPOCHS=5 \
      SFT_MAX_LENGTH=16384 \
      SFT_PROJECT_NAME=tau2_4cycle SFT_EXPERIMENT_NAME="cycle${cycle}_$(basename "$RESULTS_DIR")" \
      bash scripts/train_sft.sh data.max_token_len_per_gpu=16384 data.num_workers=0 ) >> "$LOG" 2>&1
    CURRENT_ADAPTER="$SFT_ADAPTER"
  else
    log "no SFT pairs this cycle (no successful trajectories) -- skipping SFT"
  fi

  # 4. GRPO (continues training CURRENT_ADAPTER further, only if enabled)
  if [[ "$ENABLE_GRPO" == "1" && -n "$CURRENT_ADAPTER" ]]; then
    log "GRPO training (continuing from $CURRENT_ADAPTER)"
    reload_agent "$CURRENT_ADAPTER"  # keep the external eval server in sync
    GRPO_OUT="$OUT/verl_grpo"
    VERL_DATA="$OUT/verl_grpo.parquet"
    "$MERA_PY" prepare_verl_grpo_data.py \
      --traces "$OUT/train_traces.jsonl" --skillbook "$OUT/skillbook.json" \
      --user-base-url "$USER_BASE_URL" --output "$VERL_DATA" >> "$LOG" 2>&1
    ( cd /shared_home/yuhang.yao/MERA-Evolve && \
      env CUDA_VISIBLE_DEVICES="$GRPO_GPU" TRAIN_FILE="$VERL_DATA" \
      OUTPUT_DIR="$GRPO_OUT" MODEL_PATH="$BASE_MODEL" \
      LORA_ADAPTER_PATH="$CURRENT_ADAPTER" TOTAL_STEPS=10 \
      EXPERIMENT_NAME="cycle${cycle}_$(basename "$RESULTS_DIR")" \
      bash experiments/tau-2/train_verl_grpo.sh ) >> "$LOG" 2>&1
    CURRENT_ADAPTER="$(<"$GRPO_OUT/final_adapter_path.txt")"
    if [[ ! -s "$CURRENT_ADAPTER/adapter_model.safetensors" ]]; then
      log "FATAL: verl GRPO adapter is missing: $CURRENT_ADAPTER"
      exit 1
    fi
  fi

  # 5. reload agent with this cycle's final adapter, then held-out eval
  if [[ -n "$CURRENT_ADAPTER" ]]; then
    reload_agent "$CURRENT_ADAPTER"
  fi
  log "held-out EVAL (35 tasks)"
  "$TAU2_PY" collect_traces.py \
    --bucket EVAL --workers 6 --max-steps 60 \
    --agent-model "openai/evol-llm-agent" --agent-base-url "$AGENT_BASE_URL" \
    --user-model "openai/evol-llm-user" --user-base-url "$USER_BASE_URL" \
    --skillbook "$OUT/skillbook.json" \
    --out "$OUT/eval.jsonl" >> "$LOG" 2>&1

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
