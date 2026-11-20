#!/usr/bin/env bash
# Collect E6 step 0 VAL (both seeds) against the already-running teacher/user2 servers.
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
mkdir -p "$D/results/e6_teacher_val" "$D/logs"
REPO=/home/powerop/work/yjh/MERA-Evolve

for i in $(seq 1 120); do
  a=$(curl -sf --max-time 4 http://127.0.0.1:8270/v1/models 2>/dev/null | grep -c evol-llm-teacher)
  b=$(curl -sf --max-time 4 http://127.0.0.1:8271/v1/models 2>/dev/null | grep -c evol-llm-user2)
  (( a && b )) && { echo "servers ready"; break; }
  sleep 10
done

rm -f "$D/results/e6_teacher_val/"teacher_VAL_seed*.jsonl
for S in 300 1300; do
  PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat \
  TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json \
  OPENAI_API_KEY=EMPTY "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.collect_traces \
    --bucket VAL --trials 4 --seed "$S" \
    --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
    --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
    --user-temperature 0.7 --agent-temperature 0.0 --no-agent-thinking --no-user-thinking \
    --workers 24 --max-steps 60 \
    --out "$D/results/e6_teacher_val/teacher_VAL_seed${S}.jsonl" \
    > "$D/logs/e6_teacher_val_seed${S}.log" 2>&1
  echo "SEED $S DONE rows=$(wc -l < "$D/results/e6_teacher_val/teacher_VAL_seed${S}.jsonl" 2>/dev/null)" \
    >> "$D/logs/e6_teacher_rerun.log"
done
echo ALLDONE >> "$D/logs/e6_teacher_rerun.log"
