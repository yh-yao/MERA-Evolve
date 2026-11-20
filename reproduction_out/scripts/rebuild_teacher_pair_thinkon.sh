#!/usr/bin/env bash
# Rebuild the TEACHER pair (8270 teacher / 8271 user2) with the configuration
# research_plan.md actually declares, then run Step 0 for the 9B teacher:
#   Step 0 item 1: "Evaluate 9B on the frontier (58 x 8) as a diagnostic, THEN compare 9B
#   and the 4B base on VAL seeds 300 + 1300"; l.458 "Teacher thinking stays on";
#   l.454-455 pinned parsers --reasoning-parser qwen3 / --tool-call-parser qwen3_coder.
#
# Runs CONCURRENTLY with the tail of E1: it only touches GPU 2/3 and ports 8270/8271, never
# the student pair 8260/8261 (GPU 0/1) that E1 collects through.
#
# Existing teacher_VAL_seed{300,1300}.jsonl were collected with teacher thinking OFF, so they
# are not Step-0 evidence. They are left untouched and every output goes to *_thinkon.jsonl,
# because CLAUDE.md forbids mixing artifacts from different configs inside one file.
# NB: teacher_val_collect.sh starts with `rm -f ... teacher_VAL_seed*.jsonl` -- never run it,
# it would delete that evidence.
set -euo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
LEN=131072
VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm
LOGDIR=$D/results/servers_uniform
mkdir -p "$LOGDIR" "$D/logs"

echo "[$(date -Is)] rebuilding teacher pair at len=$LEN, teacher thinking=on"

stop_port() {
  local pid
  pid="$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for port in 8270 8271; do stop_port "$port"; done
sleep 8

start() { # name model gpu port thinking parser
  local name=$1 model=$2 gpu=$3 port=$4 thinking=$5 parser=$6
  local kwargs='{"enable_thinking": false}'
  local extra=()
  if [[ "$thinking" == true ]]; then
    kwargs='{"enable_thinking": true}'
    extra=(--reasoning-parser qwen3)
  fi
  echo "[$(date -Is)] starting $name ($model) gpu=$gpu port=$port len=$LEN thinking=$thinking parser=$parser"
  CUDA_VISIBLE_DEVICES=$gpu OPENAI_API_KEY=EMPTY VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid nohup "$VLLM_BIN" serve "$model" --served-model-name "$name" --port "$port" \
      --gpu-memory-utilization 0.85 --max-model-len "$LEN" --enable-prefix-caching \
      --dtype bfloat16 --trust-remote-code --language-model-only --gdn-prefill-backend triton \
      --default-chat-template-kwargs "$kwargs" \
      --enable-auto-tool-choice --tool-call-parser "$parser" \
      ${extra[@]+"${extra[@]}"} \
      > "$LOGDIR/${name}_${LEN}.log" 2>&1 < /dev/null &
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\"" && { echo "$name ready"; return 0; }
    sleep 5
  done
  echo "FATAL $name did not come up"; tail -40 "$LOGDIR/${name}_${LEN}.log"; return 1
}

start evol-llm-teacher "$MODELS_DIR/Qwen3.5-9B" 2 8270 true  qwen3_coder || exit 1
start evol-llm-user2   "$MODELS_DIR/Qwen3.5-4B" 3 8271 false qwen3_xml || exit 1

echo "--- effective limits ---"
for pair in 8270:teacher 8271:user2; do
  port=${pair%%:*}; nm=${pair#*:}
  printf '%-8s :%-5s ' "$nm" "$port"
  grep -ho "Maximum concurrency for [0-9,]* tokens per request: [0-9.]*x" \
    "$LOGDIR/evol-llm-${nm}_${LEN}.log" 2>/dev/null | tail -1 || echo "NOT FOUND"
done

export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json
cd "$REPO"

# 1) frontier diagnostic first, in the order Step 0 item 1 gives it.
OUT=$D/results/e6_teacher_val/teacher_frontier_thinkon.jsonl
TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_frontier_only.json \
OPENAI_API_KEY=EMPTY "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.collect_traces \
  --bucket TRAIN --trials 8 --seed 2000 \
  --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
  --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
  --agent-temperature 1.0 --user-temperature 0.0 --agent-thinking --no-user-thinking \
  --workers 32 --max-steps 40 \
  --out "$OUT" > "$D/logs/e6_teacher_frontier_thinkon.log" 2>&1
echo "[$(date -Is)] TEACHER_FRONTIER_THINKON rows=$(wc -l < "$OUT")" >> "$D/logs/serial_progress.log"
echo "[$(date -Is)] TEACHER_FRONTIER_THINKON rows=$(wc -l < "$OUT")"

# 2) then the VAL comparison seeds, paired-trial identical to the 4B base runs.
for S in 300 1300; do
  OUT=$D/results/e6_teacher_val/teacher_VAL_seed${S}_thinkon.jsonl
  OPENAI_API_KEY=EMPTY "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.collect_traces \
    --bucket VAL --trials 4 --seed "$S" \
    --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
    --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
    --user-temperature 0.7 --agent-temperature 0.0 --agent-thinking --no-user-thinking \
    --workers 24 --max-steps 60 \
    --out "$OUT" > "$D/logs/e6_teacher_val_seed${S}_thinkon.log" 2>&1
  echo "[$(date -Is)] TEACHER_VAL_${S}_THINKON rows=$(wc -l < "$OUT")" >> "$D/logs/serial_progress.log"
  echo "[$(date -Is)] TEACHER_VAL_${S}_THINKON rows=$(wc -l < "$OUT")"
done
echo "TEACHER_STEP0_THINKON_DONE" >> "$D/logs/serial_progress.log"
