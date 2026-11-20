#!/usr/bin/env bash
# Second-level chain: once the VAL seed-300 and seed-1300 runs finish, run the frontier
# diagnostic (58 tasks x 8 trials) for each policy on its own GPU pair.
# Frontier protocol follows research_plan.md section 1.1: agent temperature 1.0,
# user temperature 0.0, at most 40 steps, partition_frontier_only.json.
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
cd "$REPO"

wait_for() {
  for _ in $(seq 1 1440); do
    grep -q "$2" "$1" 2>/dev/null && return 0
    sleep 30
  done
  return 1
}

# ---- student frontier ----
(
  wait_for "$D/logs/e0_val1300.log" "E0VAL1300_EXIT=" || exit 0
  tmux kill-session -t e0frontier 2>/dev/null
  tmux new-session -d -s e0frontier -n frontier "
    export OPENAI_API_KEY=EMPTY
    export TAU2_WORKSPACE=$TAU2_WORKSPACE
    export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_frontier_only.json
    export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
    $TAU2_WORKSPACE/.venv_tau2/bin/python -m tau2_evolve.collect_traces \
      --bucket TRAIN --trials 8 --seed 2000 \
      --agent-model openai/evol-llm-agent --agent-base-url http://127.0.0.1:8260/v1 \
      --user-model openai/evol-llm-user --user-base-url http://127.0.0.1:8261/v1 \
      --agent-temperature 1.0 --user-temperature 0.0 --no-agent-thinking --no-user-thinking \
      --workers 32 --max-steps 40 \
      --out $D/results/e0_base_val/base_frontier.jsonl > $D/logs/e0_frontier.log 2>&1
    echo FRONTIER_EXIT=\$? >> $D/logs/e0_frontier.log; sleep 604800"
  echo "chained student frontier"
) &

# ---- teacher frontier (diagnostic) ----
(
  wait_for "$D/logs/e6_teacher_val1300.log" "EXIT=" || exit 0
  tmux kill-session -t teacherfrontier 2>/dev/null
  tmux new-session -d -s teacherfrontier -n frontier "
    export OPENAI_API_KEY=EMPTY
    export TAU2_WORKSPACE=$TAU2_WORKSPACE
    export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_frontier_only.json
    export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
    $TAU2_WORKSPACE/.venv_tau2/bin/python -m tau2_evolve.collect_traces \
      --bucket TRAIN --trials 8 --seed 2000 \
      --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
      --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
      --agent-temperature 1.0 --user-temperature 0.0 --no-user-thinking \
      --workers 32 --max-steps 40 \
      --out $D/results/e6_teacher_val/teacher_frontier.jsonl > $D/logs/e6_teacher_frontier.log 2>&1
    echo FRONTIER_EXIT=\$? >> $D/logs/e6_teacher_frontier.log; sleep 604800"
  echo "chained teacher frontier"
) &

wait
echo "frontier chain armed"
