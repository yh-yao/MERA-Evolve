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
# Extra collect_traces flags, e.g. a per-turn router:
#   EXTRA_COLLECT_ARGS="--turn-router oracle --turn-router-teacher-base-url http://HOST:8270/v1"
# chunked_collect.sh restarts servers by re-invoking this script, so every
# server setting -- above all AGENT_ADAPTER -- must survive into that call;
# otherwise a restart would silently swap the adapter for the base model.
export BASE_MODEL USER_MODEL_PATH AGENT_GPU AGENT_PORT USER_GPU USER_PORT AGENT_ADAPTER \
  VLLM_BIN MAX_MODEL_LEN GPU_MEMORY_UTILIZATION
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

serving_matches() {
  # True when /v1/models lists $name backed by $adapter (LoRA) or by $model
  # with no adapter.
  curl -sf "http://127.0.0.1:$1/v1/models" | "$TAU2_PY" -c '
import json, os, sys
name, model, adapter = sys.argv[1:4]
real = os.path.realpath
for m in json.load(sys.stdin)["data"]:
    if m["id"] != name:
        continue
    if adapter:
        sys.exit(0 if real(m.get("root") or "") == real(adapter) else 1)
    sys.exit(0 if m.get("parent") is None and real(m.get("root") or "") == real(model) else 1)
sys.exit(1)' "$2" "$3" "$4"
}

stop_port() {
  local pid
  pid="$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
  [[ -n "$pid" ]] || return 0
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -qE -- '--served-model-name evol-llm-(agent|user)' \
    || { log "FATAL: port $1 is held by a process that is not ours"; exit 1; }
  kill "$pid"; for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true; sleep 5
}

# VERL exports Qwen3.5 adapters with text-only keys (model.layers.N...).
# vLLM >= 0.29 loads Qwen3.5 as Qwen3_5ForConditionalGeneration and only
# matches model.language_model.layers.N...; unmatched LoRA weights are
# skipped silently, so the "adapter" server answers exactly like the base
# (verified 2026-10-25: a 1e4-amplified adapter left greedy logprobs
# bit-identical). Serve a key-converted copy instead.
served_adapter() {
  local src="$1" dst
  dst="$RESULTS_DIR/served_adapters/$(echo "$(realpath "$src")" | md5sum | cut -c1-12)"
  if [[ ! -s "$dst/adapter_model.safetensors" ]]; then
    PYTHONPATH="$ROOT:${PYTHONPATH:-}" "${TRAIN_PY:-$ROOT/.venv_qwen35/bin/python}" -m verl_code_rl.prepare_qwen35_training_adapter \
      --input "$src" --output "$dst" >&2
    echo "$(realpath "$src")" > "$dst/SOURCE_ADAPTER"
  fi
  echo "$dst"
}

# Refuse to evaluate an adapter server that behaves exactly like the base.
assert_adapter_active() {
  local port="$1" adapter_dir="$2"
  "${TRAIN_PY:-$ROOT/.venv_qwen35/bin/python}" - "$port" "$adapter_dir" <<'PY' | tee -a "$LOG"
import json, sys, urllib.request
from safetensors import safe_open
port, adapter = sys.argv[1], sys.argv[2]
with safe_open(adapter + "/adapter_model.safetensors", "pt") as f:
    b_norm = max(float(f.get_tensor(k).abs().max()) for k in f.keys() if "lora_B" in k)
def lp(model):
    body = {"model": model, "prompt": "Customer: I need to change my flight to tomorrow. Agent:",
            "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    rows = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["prompt_logprobs"]
    return [next(iter(r.values()))["logprob"] for r in rows[1:]]
delta = max(abs(a - b) for a, b in zip(lp("evol-llm-agent"), lp("evol-llm-agent-base")))
print(f"[adapter-check] max |lora_B|={b_norm:.3g} max |dlogprob| vs base={delta:.3g}")
if b_norm > 1e-6 and delta == 0.0:
    sys.exit("FATAL: adapter is not applied by the server (logprobs identical to base)")
PY
}

serve() {
  local name="$1" model="$2" gpu="$3" port="$4" adapter="${5:-}"
  [[ -n "$adapter" ]] && adapter="$(served_adapter "$adapter")"
  local served=(--served-model-name "$name")
  if [[ -n "$adapter" ]]; then
    [[ -s "$adapter/adapter_model.safetensors" ]] || { log "FATAL: no adapter at $adapter"; exit 1; }
    # The base keeps a different name so requests can only reach the adapter.
    served=(--served-model-name "${name}-base" --enable-lora --max-lora-rank 64
            --lora-modules "$name=$adapter")
  fi
  if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\""; then
    # A server under the right name may still hold the wrong weights (base
    # vs. adapter); evaluating it would silently measure the wrong model.
    if serving_matches "$port" "$name" "$model" "$adapter"; then
      log "$name already serving on :$port"; return 0
    fi
    log "$name on :$port serves different weights; restarting it"
    stop_port "$port"
  fi
  log "starting $name ($model${adapter:+ + adapter $adapter}) on GPU $gpu port $port"
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
    if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\""; then
      log "$name ready"
      [[ -n "$adapter" ]] && assert_adapter_active "$port" "$adapter"
      return 0
    fi
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
    # Chunked so student/user servers restart between chunks (Qwen3.5 GDN
    # servers stall after hours); errored rows are retried by --resume.
    SERVERS_DIR="$RESULTS_DIR" bash "$SCRIPT_DIR/scripts/chunked_collect.sh" \
      --bucket "$bucket" --trials "$TRIALS" --seed "$seed" \
      --user-temperature "$USER_TEMPERATURE" --agent-temperature 0.0 \
      --workers "$WORKERS" --max-steps "$MAX_STEPS" \
      --agent-model openai/evol-llm-agent --user-model openai/evol-llm-user \
      --agent-max-tokens "$AGENT_MAX_TOKENS" --no-agent-thinking --no-user-thinking \
      ${EXTRA_COLLECT_ARGS:-} --out "$out" >> "$LOG" 2>&1
  done
  runs=()
  for seed in $RUN_SEEDS; do runs+=("$RESULTS_DIR/${RUN_LABEL}_${bucket}_seed${seed}.jsonl"); done
  "$TAU2_PY" -m tau2_evolve.eval_stats summary "${runs[@]}" | tee -a "$LOG"
  if (( ${#runs[@]} == 2 )); then
    "$TAU2_PY" -m tau2_evolve.eval_stats repeatability "${runs[@]}" \
      --json "$RESULTS_DIR/repeatability_${RUN_LABEL}_${bucket}.json" | tee -a "$LOG"
  fi
done
log "phase0 done"
