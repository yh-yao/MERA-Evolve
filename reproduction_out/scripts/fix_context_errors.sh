#!/usr/bin/env bash
# Serial step 1: repair the ContextWindowExceededError rows in the teacher VAL runs.
# Diagnosis: all 18 errors across both runs are litellm.ContextWindowExceededError, and the
# errored rows have ZERO messages -> the very first request already exceeded --max-model-len
# (long tau2 domain-policy + tool definitions), not a mid-conversation overflow.
# Fix = provision more context, which research_plan.md explicitly requires instead of truncating.
# (raising the teacher 40960 -> 65536 previously cut its error rate from 28% to 0.7%)
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
LEN=131072

# 1) strip errored rows so --resume will re-run exactly those (domain,task_id,trial)
python3 - "$D/results/e6_teacher_val/teacher_VAL_seed1300.jsonl" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p)]
keep = [r for r in rows if not r.get("error")]
open(p, "w").write("".join(json.dumps(r) + "\n" for r in keep))
print("kept", len(keep), "of", len(rows), "-> resume will re-run", len(rows) - len(keep))
PY

# 2) restart both teacher-side servers with the larger context
for port in 8270 8271; do
  pid="$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] && { kill "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null; }
  sleep 5
done
start() { # name model gpu port
  CUDA_VISIBLE_DEVICES=$3 OPENAI_API_KEY=EMPTY VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid nohup /home/powerop/.conda/envs/fave-vllm-py311/bin/vllm serve "$2" \
      --served-model-name "$1" --port "$4" --gpu-memory-utilization 0.85 \
      --max-model-len $LEN --enable-prefix-caching --dtype bfloat16 --trust-remote-code \
      --language-model-only --gdn-prefill-backend triton \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      > "$D/results/servers_teacher/${1}_128k.log" 2>&1 < /dev/null &
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$4/v1/models" 2>/dev/null | grep -q "\"$1\"" && { echo "$1 ready"; return 0; }
    sleep 5
  done
  echo "$1 FAILED"; return 1
}
start evol-llm-teacher "$MODELS_DIR/Qwen3.5-9B" 2 8270 || exit 1
start evol-llm-user2   "$MODELS_DIR/Qwen3.5-4B" 3 8271 || exit 1

# 3) resume only the stripped rows
cd "$REPO"
PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat \
TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json \
OPENAI_API_KEY=EMPTY "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.collect_traces \
  --bucket VAL --trials 4 --seed 1300 --resume \
  --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
  --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
  --user-temperature 0.7 --agent-temperature 0.0 --no-agent-thinking --no-user-thinking \
  --workers 8 --max-steps 60 \
  --out "$D/results/e6_teacher_val/teacher_VAL_seed1300.jsonl" \
  > "$D/logs/fix_teacher_1300.log" 2>&1
echo "TEACHER1300_FIXED rows=$(wc -l < "$D/results/e6_teacher_val/teacher_VAL_seed1300.jsonl")" \
  >> "$D/logs/serial_progress.log"
