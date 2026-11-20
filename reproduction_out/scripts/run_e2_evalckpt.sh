#!/usr/bin/env bash
# Per research_plan.md E2, every checkpoint must be evaluated on the 58-task frontier
# (464 conversations) and compared with the base; checkpoints passing the bar then go to VAL.
# This runner waits for E2 arm 1 training to finish, then delegates both stages to the
# repository pipeline (10_4b_rl_pipeline.sh stages: evalckpt val). No evaluation logic is
# re-implemented here.
set -uo pipefail

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
CONFIG=$REPO/reproduction_out/config/pipeline_e2_arm1.env
RUN_DIR=$D/results/e2_arm1_lr1e-5
LOG=$D/logs/e2_arm1_evalckpt.log
mkdir -p "$D/logs" "$RUN_DIR/base"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# The pipeline's evalckpt stage requires base/frontier.jsonl; the base frontier is the E0
# artifact (unbounded pass), which is what the teacher/base comparison also used.
if [[ ! -s "$RUN_DIR/base/frontier.jsonl" ]]; then
  cp "$D/results/e0_base_val/base_frontier_max1024.jsonl" "$RUN_DIR/base/frontier.jsonl"
fi
log "armed: waiting for E2 arm 1 training to produce final_adapter_path.txt"

for _ in $(seq 1 5760); do          # up to 16 h
  [[ -s "$RUN_DIR/train/group_O/verl/final_adapter_path.txt" ]] && break
  if [[ $(pgrep -f 'one_step_off[_]policy' | wc -l) == 0 ]] && \
     grep -q 'E2_ARM1 EXIT rc=[1-9]' "$D/logs/serial_progress.log" 2>/dev/null; then
    log "training exited without final_adapter; not evaluating"; exit 1
  fi
  sleep 20
done
[[ -s "$RUN_DIR/train/group_O/verl/final_adapter_path.txt" ]] || { log "timeout waiting for training"; exit 1; }
log "training finished; waiting for training processes to release the GPUs"
for _ in $(seq 1 120); do
  [[ $(pgrep -f 'one_step_off[_]policy' | wc -l) == 0 ]] && break
  sleep 10
done
sleep 20
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while read -r q; do log "  $q"; done

log "delegating to repository pipeline: evalckpt val (PASS_DELTA=0.05, VAL_SEEDS='300 1300')"
cd "$REPO"
start=$(date +%s)
SMOKE=0 bash experiments/tau-2/10_4b_rl_pipeline.sh "$CONFIG" evalckpt val >> "$LOG" 2>&1
rc=$?
end=$(date +%s)
log "E2_ARM1_EVAL_EXIT rc=$rc elapsed=$((end-start))s"
echo "E2_ARM1_EVAL EXIT rc=$rc elapsed=$((end-start))s" >> "$D/logs/serial_progress.log"
exit "$rc"
