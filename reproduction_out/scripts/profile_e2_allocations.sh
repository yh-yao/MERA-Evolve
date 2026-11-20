#!/usr/bin/env bash
# Runs the research_plan.md Section I warmed-up profile (10-20 steps) in TWO GPU allocations
# so the real step-time can be compared on this 4-card node. Delegates every stage to the
# repository pipeline 10_4b_rl_pipeline.sh (select + train); adds no pipeline logic.
#
# Gates on the 2-step smoke having exited rc=0, so a broken smoke never wastes profile time.
# Each allocation gets its own RUN_DIR, which also avoids the pipeline's `.done_<stage>`
# markers silently skipping select/train.
set -uo pipefail

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
SERIAL=$D/logs/serial_progress.log
LOG=$D/logs/profile_e2_allocations.log
mkdir -p "$D/logs"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "armed: waiting for the Section I smoke to exit rc=0"
for _ in $(seq 1 4320); do          # up to 12 h
  if grep -q 'SECTION_I_PIPELINE EXIT rc=0' "$SERIAL" 2>/dev/null; then break; fi
  if grep -q 'SECTION_I_PIPELINE EXIT rc=[1-9]' "$SERIAL" 2>/dev/null; then
    log "smoke failed (rc!=0); not profiling"; exit 1
  fi
  sleep 10
done
grep -q 'SECTION_I_PIPELINE EXIT rc=0' "$SERIAL" 2>/dev/null || { log "timeout waiting for smoke"; exit 1; }
log "smoke passed; starting 12-step profiles"

run_alloc() { # tag config
  local tag=$1 cfg=$2
  local rundir
  rundir=$(grep -E '^RUN_DIR=' "$cfg" | cut -d= -f2-)
  mkdir -p "$rundir/r1"
  ln -sf "$D/results/e1_r1/AREAL_r1.jsonl" "$rundir/r1/AREAL_e1.jsonl"
  ln -sf "$D/results/e1_r1/TRAIN_r1.jsonl" "$rundir/r1/TRAIN_e1.jsonl"
  log "=== $tag: $(grep -E '^SERVE_GPU=|^TRAIN_GPUS=|^ROLLOUT_GPUS=' "$cfg" | tr '\n' ' ')==="
  local t0 t1 rc
  t0=$(date +%s)
  ( cd "$REPO" && SMOKE=0 bash experiments/tau-2/10_4b_rl_pipeline.sh "$cfg" select train ) >> "$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  local tl="$rundir/train/group_O/train.log"
  local clip steps_per_s
  clip=$(grep -oE "response_length/clip_ratio:[0-9.]+" "$tl" 2>/dev/null | awk -F: '{s+=$2;n++} END{if(n)printf "%.3f",s/n}')
  log "PROFILE_${tag} rc=$rc elapsed=$((t1-t0))s clip_ratio=${clip:-n/a} log=$LOG"
  echo "PROFILE_${tag} EXIT rc=$rc elapsed=$((t1-t0))s clip=${clip:-n/a}" >> "$SERIAL"
}

run_alloc A "$REPO/reproduction_out/config/pipeline_profileA.env"
run_alloc B "$REPO/reproduction_out/config/pipeline_profileB.env"
log "PROFILE_ALLDONE"
echo "PROFILE_ALLDONE" >> "$SERIAL"
