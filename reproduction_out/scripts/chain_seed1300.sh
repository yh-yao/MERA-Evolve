#!/usr/bin/env bash
# Chain the second VAL seed (1300) for both policies, so the teacher-vs-student comparison
# has seeds 300+1300 as research_plan.md E6 step 0 requires (~7 pt minimum detectable difference).
# Waits for the seed-300 runs to finish, then launches seed 1300 in its own tmux session.
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
cd "$REPO"

wait_for() { # marker-file-or-log-grep
  for _ in $(seq 1 720); do
    grep -q "$2" "$1" 2>/dev/null && return 0
    sleep 30
  done
  return 1
}

# ---- student VAL seed 1300 ----
if wait_for "$D/logs/e0_val300.log" "E0VAL_EXIT=" ; then
  tmux kill-session -t e0val1300 2>/dev/null
  tmux new-session -d -s e0val1300 -n val "
    export OPENAI_API_KEY=EMPTY
    export VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm
    export BASE_MODEL=$MODELS_DIR/Qwen3.5-4B
    export USER_MODEL_PATH=$MODELS_DIR/Qwen3.5-4B
    export AGENT_GPU=0 AGENT_PORT=8260 USER_GPU=1 USER_PORT=8261
    export RESULTS_DIR=$D/results/e0_base_val
    export BUCKETS=VAL RUN_SEEDS=1300 TRIALS=4 WORKERS=64
    export TAU2_WORKSPACE=$TAU2_WORKSPACE
    bash experiments/tau-2/08_phase0_eval_protocol.sh > $D/logs/e0_val1300.log 2>&1
    echo E0VAL1300_EXIT=\$? >> $D/logs/e0_val1300.log; sleep 604800"
  echo "chained student seed 1300"
fi

# ---- teacher VAL seed 1300 ----
if wait_for "$D/logs/e6_teacher_val.log" "EXIT=" ; then
  tmux kill-session -t teacherval1300 2>/dev/null
  tmux new-session -d -s teacherval1300 -n val "
    export OPENAI_API_KEY=EMPTY
    export TAU2_WORKSPACE=$TAU2_WORKSPACE
    export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json
    export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
    $TAU2_WORKSPACE/.venv_tau2/bin/python -m tau2_evolve.collect_traces \
      --bucket VAL --trials 4 --seed 1300 \
      --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
      --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
      --user-temperature 0.7 --agent-temperature 0.0 --no-user-thinking \
      --workers 32 --max-steps 60 \
      --out $D/results/e6_teacher_val/teacher_VAL_seed1300.jsonl > $D/logs/e6_teacher_val1300.log 2>&1
    echo EXIT=\$? >> $D/logs/e6_teacher_val1300.log; sleep 604800"
  echo "chained teacher seed 1300"
fi
echo "chain launched"
