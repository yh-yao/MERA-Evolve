#!/usr/bin/env bash
# Serial step: once E1 finishes (BOTH buckets -> E1_ALLDONE), raise the STUDENT pair
# (8260 agent / 8261 user) to --max-model-len 131072 and re-collect only the rows that died
# with ContextWindowExceededError.
#
# Scope is deliberately limited to 8260/8261. The teacher pair (8270/8271) is rebuilt and
# exercised by rebuild_teacher_pair_thinkon.sh, which is allowed to run concurrently with
# the tail of E1; stopping 8270/8271 from here would kill that job, so they are NEVER
# touched from this script.
#
# Student and user settings are copied verbatim from what E1 is collecting with right now
# (thinking off, qwen3_xml) so the resumed rows stay comparable to the rows already written.
# Rows still over 131072 are recorded as errors, never truncated (research_plan.md:
# "Do not silently truncate ... provision the required context").
set -euo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
LEN=131072
VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm
PROG=$D/logs/e1_r1_progress.log
LOGDIR=$D/results/servers_uniform
mkdir -p "$LOGDIR" "$D/logs"

echo "[$(date -Is)] armed: waiting for E1_ALLDONE in $PROG"
for _ in $(seq 1 4320); do          # up to 12 h
  if grep -q '^E1_ALLDONE$' "$PROG" 2>/dev/null; then break; fi
  sleep 10
done
grep -q '^E1_ALLDONE$' "$PROG" 2>/dev/null || { echo "E1 did not finish in 12h; aborting untouched"; exit 1; }
echo "[$(date -Is)] E1 finished -> raising student pair to $LEN"

# 1) back up before anything is rewritten
for b in AREAL TRAIN; do
  f=$D/results/e1_r1/${b}_r1.jsonl
  if [[ -f "$f" ]]; then cp -n "$f" "$f.bak_pre${LEN}"; fi
done

# 2) stop ONLY the student pair (serve_pair.sh would short-circuit with "already serving")
stop_port() {
  local pid
  pid="$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for port in 8260 8261; do stop_port "$port"; done
sleep 8

start() { # name model gpu port thinking parser
  local name=$1 model=$2 gpu=$3 port=$4 thinking=$5 parser=$6
  local kwargs='{"enable_thinking": false}'
  local extra=()
  if [[ "$thinking" == true ]]; then
    kwargs='{"enable_thinking": true}'
    extra=(--reasoning-parser qwen3)          # research_plan.md l.454-455
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

start evol-llm-agent "$MODELS_DIR/Qwen3.5-4B" 0 8260 false qwen3_xml || exit 1
start evol-llm-user  "$MODELS_DIR/Qwen3.5-4B" 1 8261 false qwen3_xml || exit 1

# 3) prove the new limit actually took effect on each endpoint
echo "--- effective limits ---"
for pair in 8260:agent 8261:user; do
  port=${pair%%:*}; nm=${pair#*:}
  printf '%-8s :%-5s ' "$nm" "$port"
  grep -ho "Maximum concurrency for [0-9,]* tokens per request: [0-9.]*x" \
    "$LOGDIR/evol-llm-${nm}_${LEN}.log" 2>/dev/null | tail -1 || echo "NOT FOUND"
done

# 4) strip only the context-overflow rows so --resume re-runs exactly those
for b in AREAL TRAIN; do
  f=$D/results/e1_r1/${b}_r1.jsonl
  [[ -f "$f" ]] || continue
  python3 - "$f" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
def is_ctx_err(r):
    for k in ("error", "status", "exception"):
        v = r.get(k)
        if isinstance(v, str) and ("ContextWindowExceeded" in v or "maximum context length" in v):
            return True
    return False
keep = [r for r in rows if not is_ctx_err(r)]
open(p, "w").write("".join(json.dumps(r) + "\n" for r in keep))
print(f"{p}: kept {len(keep)} of {len(rows)} -> resume re-runs {len(rows)-len(keep)}")
PY
done

# 5) resume -- identical parameters to run_e1_r1.sh, only the errored trials are missing
export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json
cd "$REPO"
for b in AREAL TRAIN; do
  f=$D/results/e1_r1/${b}_r1.jsonl
  [[ -f "$f" ]] || continue
  OPENAI_API_KEY=EMPTY "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.collect_traces \
    --bucket "$b" --trials 4 --seed 4000 --resume \
    --agent-model openai/evol-llm-agent --agent-base-url http://127.0.0.1:8260/v1 \
    --user-model openai/evol-llm-user --user-base-url http://127.0.0.1:8261/v1 \
    --agent-temperature 1.0 --user-temperature 0.0 --no-agent-thinking --no-user-thinking \
    --workers 48 --max-steps 40 \
    --out "$f" > "$D/logs/e1_r1_${b}_fix131072.log" 2>&1
  echo "[$(date -Is)] RESUMED $b rows=$(wc -l < "$f")" >> "$D/logs/serial_progress.log"
  echo "[$(date -Is)] RESUMED $b rows=$(wc -l < "$f")"
done
echo "E1_131072_FIX_DONE" >> "$D/logs/serial_progress.log"
