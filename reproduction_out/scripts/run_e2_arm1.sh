#!/usr/bin/env bash
# Launch E2 arm 1 (LR=1e-5, 200 steps, 2 actor + 1 rollout) once the Section I
# 2-step smoke has passed. All training work is delegated to the repository pipeline
# experiments/tau-2/10_4b_rl_pipeline.sh (research_plan.md E2 settings).
set -uo pipefail

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
SERIAL=$D/logs/serial_progress.log
LOG=$D/logs/e2_arm1.log
CONFIG=$REPO/reproduction_out/config/pipeline_e2_arm1.env
RUN_DIR=$D/results/e2_arm1_lr1e-5
mkdir -p "$D/logs" "$RUN_DIR/r1"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# Gate 1: the latest smoke result must be rc=0 (it may already have happened).
while true; do
  LAST=$(grep 'SECTION_I_PIPELINE EXIT' "$SERIAL" 2>/dev/null | tail -1)
  [[ "$LAST" == *"rc=0"* ]] && break
  if [[ "$LAST" == *"rc="* && "$LAST" != *"rc=0"* ]]; then
    log "latest smoke result is not rc=0: $LAST"; exit 1
  fi
  sleep 10
done
log "smoke passed: $LAST"

# Gate 2: teacher Step 0 marker must exist (it needs GPU2/3, which E2 also uses).
while true; do
  NEW_T=$(grep 'TEACHER_STEP0_THINKON_DONE' "$SERIAL" 2>/dev/null | tail -1)
  [[ -n "$NEW_T" ]] && break
  if grep -q 'TEACHER_COMPLETE EXIT rc=[1-9]' "$SERIAL" 2>/dev/null; then
    log "teacher completion failed"; exit 1
  fi
  sleep 10
done
log "teacher Step 0 done: $NEW_T"
log "smoke passed; starting E2 arm 1 (LR=1e-5, 200 steps, 2 actor + 1 rollout)"

ln -sf "$D/results/e1_r1/AREAL_r1.jsonl" "$RUN_DIR/r1/AREAL_e1.jsonl"
ln -sf "$D/results/e1_r1/TRAIN_r1.jsonl" "$RUN_DIR/r1/TRAIN_e1.jsonl"

cd "$REPO"
start=$(date +%s)
SMOKE=0 bash experiments/tau-2/10_4b_rl_pipeline.sh "$CONFIG" select train >> "$LOG" 2>&1
rc=$?
end=$(date +%s)
log "E2_ARM1_EXIT rc=$rc elapsed=$((end-start))s log=$LOG"
echo "E2_ARM1 EXIT rc=$rc elapsed=$((end-start))s" >> "$SERIAL"
exit "$rc"
