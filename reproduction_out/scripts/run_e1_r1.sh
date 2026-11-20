#!/usr/bin/env bash
# E1 R1 selection (research_plan.md section E1):
#   4B base samples each valid AReaL task (1837) and each TRAIN task (139) 4 times so tasks with
#   0 < p < 1 can be picked as GRPO prompts. Agent temperature 1.0, user temperature 0.0,
#   at most 40 steps, seed 4000.
# This stage is SERVING-only (tau2 simulator + the two vLLM servers); it does NOT need VERL.
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
mkdir -p "$D/results/e1_r1"
cd "$REPO"
export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
export TAU2_WORKSPACE=$TAU2_WORKSPACE
export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json
export MERA_AREAL_DIR=$MERA_AREAL_DIR

for bucket in AREAL TRAIN; do
  for i in $(seq 1 180); do
    curl -sf --max-time 4 http://127.0.0.1:8260/v1/models 2>/dev/null | grep -q evol-llm-agent && break
    sleep 10
  done
  OPENAI_API_KEY=EMPTY "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.collect_traces \
    --bucket "$bucket" --trials 4 --seed 4000 \
    --agent-model openai/evol-llm-agent --agent-base-url http://127.0.0.1:8260/v1 \
    --user-model openai/evol-llm-user --user-base-url http://127.0.0.1:8261/v1 \
    --agent-temperature 1.0 --user-temperature 0.0 --no-agent-thinking --no-user-thinking \
    --workers 48 --max-steps 40 \
    --out "$D/results/e1_r1/${bucket}_r1.jsonl" \
    > "$D/logs/e1_r1_${bucket}.log" 2>&1
  echo "BUCKET $bucket done rows=$(wc -l < "$D/results/e1_r1/${bucket}_r1.jsonl" 2>/dev/null)" \
    >> "$D/logs/e1_r1_progress.log"
done
echo E1_ALLDONE >> "$D/logs/e1_r1_progress.log"
