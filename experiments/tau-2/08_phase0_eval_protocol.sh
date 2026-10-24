#!/usr/bin/env bash
# Phase 0 evaluation protocol: base-model repeatability under pass^k.
#
# Runs the unadapted student twice (independent seed blocks) on the VAL and
# EVAL buckets of partition v2 with K trials per task, then reports pass^1,
# pass^K, the run-vs-run paired CI and the minimum detectable difference.
# Phase 0 exit criterion: the two runs agree (CI covers 0) and the MDD is
# <= 10 points; otherwise the eval must be enlarged before Phase 1.
#
# Protocol held fixed for every later comparison:
#   agent greedy (temperature 0), user simulator temperature USER_TEMPERATURE
#   with per-trial seeds, non-thinking, 1024-token agent turns, 60 steps.
#
# Phase 1 reuses the identical protocol for trained students: set
# AGENT_ADAPTER to a LoRA adapter directory (adapter_config.json +
# adapter_model.safetensors) and RUN_LABEL to name the output files.
#
# Required: RESULTS_DIR (absolute). Servers are started on AGENT_GPU/USER_GPU
# unless something already answers with the expected served name.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
TAU2_PY="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
VLLM_BIN="${VLLM_BIN:?set VLLM_BIN to a vLLM >= 0.18 executable that serves Qwen3.5}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR}"
[[ "$RESULTS_DIR" == /* ]] || { echo "RESULTS_DIR must be absolute" >&2; exit 2; }

BASE_MODEL="${BASE_MODEL:?set BASE_MODEL (student, e.g. a local Qwen3.5-2B path)}"
USER_MODEL_PATH="${USER_MODEL_PATH:?set USER_MODEL_PATH (user simulator, e.g. Qwen3.5-4B)}"
AGENT_GPU="${AGENT_GPU:?set AGENT_GPU}" AGENT_PORT="${AGENT_PORT:-8260}"
USER_GPU="${USER_GPU:?set USER_GPU}" USER_PORT="${USER_PORT:-8261}"
AGENT_ADAPTER="${AGENT_ADAPTER:-}"
RUN_LABEL="${RUN_LABEL:-base}"
TRIALS="${TRIALS:-4}"
USER_TEMPERATURE="${USER_TEMPERATURE:-0.7}"
RUN_SEEDS="${RUN_SEEDS:-300 1300}"
BUCKETS="${BUCKETS-VAL EVAL}"
WORKERS="${WORKERS:-64}"
MAX_STEPS="${MAX_STEPS:-60}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export TAU2_PARTITION_PATH="${TAU2_PARTITION_PATH:-$SCRIPT_DIR/data/partition_v2.json}"
export TAU2_WORKSPACE
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/compat:${PYTHONPATH:-}"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/phase0.log"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

"$TAU2_PY" - "$RESULTS_DIR/config_snapshot.json" <<PY
import json, sys
json.dump({k: v for k, v in {
    "BASE_MODEL": "$BASE_MODEL", "USER_MODEL_PATH": "$USER_MODEL_PATH",
    "TRIALS": $TRIALS, "USER_TEMPERATURE": $USER_TEMPERATURE, "AGENT_TEMPERATURE": 0.0,
    "RUN_SEEDS": "$RUN_SEEDS", "BUCKETS": "$BUCKETS", "MAX_STEPS": $MAX_STEPS,
    "AGENT_MAX_TOKENS": $AGENT_MAX_TOKENS, "PARTITION": "$TAU2_PARTITION_PATH",
    "VLLM_BIN": "$VLLM_BIN", "thinking": False,
    "AGENT_ADAPTER": "$AGENT_ADAPTER", "RUN_LABEL": "$RUN_LABEL",
}.items()}, open(sys.argv[1], "w"), indent=2)
PY

serve() {
  local name="$1" model="$2" gpu="$3" port="$4" adapter="${5:-}"
  local served=(--served-model-name "$name")
  if [[ -n "$adapter" ]]; then
    [[ -s "$adapter/adapter_model.safetensors" ]] || { log "FATAL: no adapter at $adapter"; exit 1; }
    # The base keeps a different name so requests can only reach the adapter.
    served=(--served-model-name "${name}-base" --enable-lora --max-lora-rank 64
            --lora-modules "$name=$adapter")
  fi
  if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\""; then
    log "$name already serving on :$port"; return 0
  fi
  log "starting $name ($model) on GPU $gpu port $port"
  # setsid: the servers get their own process group, so a timeout or hangup
  # delivered to this script's group does not take them down mid-run.
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid nohup "$VLLM_BIN" serve "$model" "${served[@]}" --port "$port" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN" \
      --enable-prefix-caching --dtype bfloat16 --trust-remote-code \
      --language-model-only --gdn-prefill-backend "${GDN_PREFILL_BACKEND:-triton}" \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_xml}" \
      > "$RESULTS_DIR/${name}_server.log" 2>&1 < /dev/null &
  echo $! > "$RESULTS_DIR/${name}_server.pid"
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\"" && { log "$name ready"; return 0; }
    kill -0 "$(cat "$RESULTS_DIR/${name}_server.pid")" 2>/dev/null || { tail -40 "$RESULTS_DIR/${name}_server.log" >&2; exit 1; }
    sleep 5
  done
  log "FATAL: $name not ready"; exit 1
}

serve evol-llm-agent "$BASE_MODEL" "$AGENT_GPU" "$AGENT_PORT" "$AGENT_ADAPTER"
serve evol-llm-user "$USER_MODEL_PATH" "$USER_GPU" "$USER_PORT"

for bucket in $BUCKETS; do
  for seed in $RUN_SEEDS; do
    out="$RESULTS_DIR/${RUN_LABEL}_${bucket}_seed${seed}.jsonl"
    log "collect bucket=$bucket seed=$seed trials=$TRIALS -> $out"
    # Errored rows are infrastructure failures; --resume retries them.
    for attempt in 1 2 3; do
      "$TAU2_PY" -m tau2_evolve.collect_traces \
        --bucket "$bucket" --trials "$TRIALS" --seed "$seed" \
        --user-temperature "$USER_TEMPERATURE" --agent-temperature 0.0 \
        --workers "$WORKERS" --max-steps "$MAX_STEPS" --resume \
        --agent-model openai/evol-llm-agent --agent-base-url "http://127.0.0.1:$AGENT_PORT/v1" \
        --user-model openai/evol-llm-user --user-base-url "http://127.0.0.1:$USER_PORT/v1" \
        --agent-max-tokens "$AGENT_MAX_TOKENS" --no-agent-thinking --no-user-thinking \
        --out "$out" >> "$LOG" 2>&1
      # Check the row-level field; message text often contains "error".
      "$TAU2_PY" -c 'import json,sys; sys.exit(any(json.loads(l).get("error") for l in open(sys.argv[1]) if l.strip()))' "$out" && break
      log "errored rows remain after attempt $attempt; retrying"
    done
  done
  set -- $RUN_SEEDS
  "$TAU2_PY" -m tau2_evolve.eval_stats summary "$RESULTS_DIR/${RUN_LABEL}_${bucket}_seed$1.jsonl" \
    "$RESULTS_DIR/${RUN_LABEL}_${bucket}_seed$2.jsonl" | tee -a "$LOG"
  "$TAU2_PY" -m tau2_evolve.eval_stats repeatability "$RESULTS_DIR/${RUN_LABEL}_${bucket}_seed$1.jsonl" \
    "$RESULTS_DIR/${RUN_LABEL}_${bucket}_seed$2.jsonl" --json "$RESULTS_DIR/repeatability_${RUN_LABEL}_${bucket}.json" | tee -a "$LOG"
done
log "phase0 done"
