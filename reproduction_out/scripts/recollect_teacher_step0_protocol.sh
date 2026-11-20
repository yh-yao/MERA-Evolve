#!/usr/bin/env bash
# Re-collect teacher Step 0 under the repository protocol after the audit found two
# deviations in the first pass:
#   1) --agent-max-tokens was omitted (08_phase0_eval_protocol.sh uses 1024, and the E0 base
#      VAL snapshot records AGENT_MAX_TOKENS=1024) -> unmatched generation budgets and
#      unbounded generation (root cause of a >45 min runaway).
#   2) collection was not chunked -> no protection against the Qwen3.5 GDN zero-throughput
#      stall that chunked_collect.sh documents.
#
# chunked_collect.sh itself cannot be reused here: its stop_servers only recognises
# --served-model-name evol-llm-(agent|user), and it restarts via 08, which hardcodes
# --no-agent-thinking and no --reasoning-parser. research_plan.md l.458/l.454 requires the
# teacher to run with thinking ON and --reasoning-parser qwen3, so the teacher servers must be
# started with that configuration. The chunk loop below is orchestration only; every
# collection call is the repository's tau2_evolve.collect_traces.
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
SERIAL=$D/logs/serial_progress.log
LOG=$D/logs/teacher_step0_protocol.log
PY="$TAU2_WORKSPACE/.venv_tau2/bin/python"
VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm
LEN=131072
CHUNK_JOBS="${CHUNK_JOBS:-128}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
MAX_CHUNKS="${MAX_CHUNKS:-40}"
mkdir -p "$D/logs"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

stop_port() {
  local pid
  pid="$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}

start_srv() { # name model gpu port thinking parser
  local name=$1 model=$2 gpu=$3 port=$4 thinking=$5 parser=$6
  local kwargs='{"enable_thinking": false}' extra=()
  [[ "$thinking" == true ]] && { kwargs='{"enable_thinking": true}'; extra=(--reasoning-parser qwen3); }
  CUDA_VISIBLE_DEVICES=$gpu OPENAI_API_KEY=EMPTY VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid nohup "$VLLM_BIN" serve "$model" --served-model-name "$name" --port "$port" \
      --gpu-memory-utilization 0.85 --max-model-len "$LEN" --enable-prefix-caching \
      --dtype bfloat16 --trust-remote-code --language-model-only --gdn-prefill-backend triton \
      --default-chat-template-kwargs "$kwargs" \
      --enable-auto-tool-choice --tool-call-parser "$parser" \
      ${extra[@]+"${extra[@]}"} \
      > "$D/results/servers_uniform/${name}_${LEN}.log" 2>&1 < /dev/null &
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\"" && { log "$name ready"; return 0; }
    sleep 5
  done
  log "FATAL $name did not start"; return 1
}

restart_servers() {
  stop_port 8270; stop_port 8271; sleep 8
  start_srv evol-llm-teacher "$MODELS_DIR/Qwen3.5-9B" 2 8270 true  qwen3_coder || return 1
  start_srv evol-llm-user2   "$MODELS_DIR/Qwen3.5-4B" 3 8271 false qwen3_xml   || return 1
}

chunked() { # outfile  collect_traces args...
  local out=$1; shift
  local chunk=0 pending=1 tmplog="$LOG.tmp"
  while (( chunk < MAX_CHUNKS )); do
    chunk=$((chunk+1))
    restart_servers || return 1
    "$PY" -m tau2_evolve.collect_traces "$@" --resume \
      --max-jobs "$CHUNK_JOBS" --request-timeout "$REQUEST_TIMEOUT" --out "$out" \
      > "$tmplog" 2>&1
    cat "$tmplog" >> "$LOG"
    pending=$(sed -n 's/.*remaining=\([0-9]*\).*/\1/p' "$tmplog" | tail -1)
    log "chunk $chunk rows=$(wc -l < "$out" 2>/dev/null || echo 0) pending=${pending:-?}"
    rm -f "$tmplog"
    [[ "${pending:-1}" == "0" ]] && return 0
  done
  return 1
}

export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json
cd "$REPO"

log "=== frontier 58x8 (seed 2000) with --agent-max-tokens 1024 ==="
TAU2_PARTITION_PATH="$REPO/experiments/tau-2/data/partition_frontier_only.json" \
chunked "$D/results/e6_teacher_val/teacher_frontier_thinkon.jsonl" \
  --bucket TRAIN --trials 8 --seed 2000 \
  --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
  --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
  --agent-temperature 1.0 --user-temperature 0.0 --agent-thinking --no-user-thinking \
  --agent-max-tokens 1024 --workers 32 --max-steps 40

for S in 300 1300; do
  log "=== VAL seed $S with --agent-max-tokens 1024 ==="
  chunked "$D/results/e6_teacher_val/teacher_VAL_seed${S}_thinkon.jsonl" \
    --bucket VAL --trials 4 --seed "$S" \
    --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
    --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
    --agent-temperature 0.0 --user-temperature 0.7 --agent-thinking --no-user-thinking \
    --agent-max-tokens 1024 --workers 24 --max-steps 60
done

nf=$(wc -l < "$D/results/e6_teacher_val/teacher_frontier_thinkon.jsonl" 2>/dev/null || echo 0)
n3=$(wc -l < "$D/results/e6_teacher_val/teacher_VAL_seed300_thinkon.jsonl" 2>/dev/null || echo 0)
n13=$(wc -l < "$D/results/e6_teacher_val/teacher_VAL_seed1300_thinkon.jsonl" 2>/dev/null || echo 0)
log "final: frontier=$nf seed300=$n3 seed1300=$n13 (target 464/268/268)"
if [[ "$nf" -ge 464 && "$n3" -ge 268 && "$n13" -ge 268 ]]; then
  echo "[$(date -Is)] TEACHER_STEP0_THINKON_DONE protocol=agent-max-tokens-1024 frontier=$nf s300=$n3 s1300=$n13" >> "$SERIAL"
  log "marker written"
else
  log "NOT writing marker: counts below target"
  exit 1
fi
