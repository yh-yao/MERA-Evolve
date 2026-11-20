#!/usr/bin/env bash
# Complete teacher Step 0 on top of the existing "unbounded" pass, instead of restarting it.
# Existing state: frontier 464 rows (67 error), VAL seed300 266 rows (6 error),
# VAL seed1300 42 rows. --resume drops errored rows and re-runs only what is missing,
# so this both fills seed1300 and repairs the error rows.
#
# Configuration is kept identical to the pass that produced the existing rows: thinking ON,
# --reasoning-parser qwen3, --tool-call-parser qwen3_coder, max-model-len 131072, and NO
# --agent-max-tokens (1024 was proven incompatible with thinking: 76% empty messages).
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
SERIAL=$D/logs/serial_progress.log
LOG=$D/logs/teacher_step0_complete.log
PY="$TAU2_WORKSPACE/.venv_tau2/bin/python"
VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm
LEN=131072
mkdir -p "$D/logs" "$D/results/servers_uniform"

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

stop_port 8270; stop_port 8271; sleep 8
start_srv evol-llm-teacher "$MODELS_DIR/Qwen3.5-9B" 2 8270 true  qwen3_coder || exit 1
start_srv evol-llm-user2   "$MODELS_DIR/Qwen3.5-4B" 3 8271 false qwen3_xml   || exit 1

export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
cd "$REPO"

val() { # seed
  local seed=$1 out="$D/results/e6_teacher_val/teacher_VAL_seed${1}_thinkon.jsonl"
  log "VAL seed=$seed --resume (rows now $(wc -l < "$out" 2>/dev/null || echo 0))"
  TAU2_PARTITION_PATH="$REPO/experiments/tau-2/data/partition_v2.json" \
  OPENAI_API_KEY=EMPTY timeout 7200 "$PY" -m tau2_evolve.collect_traces \
    --bucket VAL --trials 4 --seed "$seed" --resume \
    --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
    --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
    --agent-temperature 0.0 --user-temperature 0.7 --agent-thinking --no-user-thinking \
    --workers 24 --max-steps 60 --request-timeout 900 --out "$out" \
    >> "$D/logs/e6_teacher_val_seed${seed}_complete.log" 2>&1
  log "seed=$seed rc=$? rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
}

log "frontier --resume (repair 67 error rows)"
TAU2_PARTITION_PATH="$REPO/experiments/tau-2/data/partition_frontier_only.json" \
OPENAI_API_KEY=EMPTY timeout 7200 "$PY" -m tau2_evolve.collect_traces \
  --bucket TRAIN --trials 8 --seed 2000 --resume \
  --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
  --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
  --agent-temperature 1.0 --user-temperature 0.0 --agent-thinking --no-user-thinking \
  --workers 32 --max-steps 40 --request-timeout 900 \
  --out "$D/results/e6_teacher_val/teacher_frontier_thinkon.jsonl" \
  >> "$D/logs/e6_teacher_frontier_complete.log" 2>&1
log "frontier rc=$? rows=$(wc -l < "$D/results/e6_teacher_val/teacher_frontier_thinkon.jsonl")"

val 300
val 1300

nf=$(wc -l < "$D/results/e6_teacher_val/teacher_frontier_thinkon.jsonl" 2>/dev/null || echo 0)
n3=$(wc -l < "$D/results/e6_teacher_val/teacher_VAL_seed300_thinkon.jsonl" 2>/dev/null || echo 0)
n13=$(wc -l < "$D/results/e6_teacher_val/teacher_VAL_seed1300_thinkon.jsonl" 2>/dev/null || echo 0)
log "final: frontier=$nf seed300=$n3 seed1300=$n13 (target 464/268/268)"
if [[ "$nf" -ge 464 && "$n3" -ge 268 && "$n13" -ge 268 ]]; then
  echo "[$(date -Is)] TEACHER_STEP0_THINKON_DONE unbounded frontier=$nf s300=$n3 s1300=$n13" >> "$SERIAL"
  log "marker written"
else
  log "NOT writing marker: counts below target"; exit 1
fi
