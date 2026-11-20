#!/usr/bin/env bash
# Thin gate before the repository's Section I smoke. Delegates all work to
# experiments/tau-2/10_4b_rl_pipeline.sh (SMOKE=1, stages: select train).
#
# Gate: E1 collection finished + E1 repair finished + no collector running.
# Teacher Step 0 is deliberately NOT part of this gate: research_plan.md E2 is
# "student update without a teacher", so E2 does not depend on teacher qualification.
set -uo pipefail

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
E1PROG=$D/logs/e1_r1_progress.log
SERIAL=$D/logs/serial_progress.log
CONFIG=$REPO/reproduction_out/config/pipeline_smoke.env
RUN_DIR=$D/results/section_i_pipeline
LOG=$D/logs/section_i_pipeline.log
mkdir -p "$D/logs" "$RUN_DIR/r1"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
log "armed: waiting for E1 completion + idle collectors (teacher Step 0 not required)"

for _ in $(seq 1 2880); do          # up to 8 h
  ok1=0; ok2=0
  grep -q '^E1_ALLDONE$' "$E1PROG" 2>/dev/null && ok1=1
  grep -q 'E1_131072_FIX_DONE' "$SERIAL" 2>/dev/null && ok2=1
  alive=$(pgrep -f 'tau2_evolve[.]collect' | wc -l)
  if [[ $ok1 == 1 && $ok2 == 1 && $alive == 0 ]]; then
    log "gates passed (E1=$ok1 FIX=$ok2 collectors=$alive)"
    break
  fi
  sleep 10
done

if ! (grep -q '^E1_ALLDONE$' "$E1PROG" 2>/dev/null && grep -q 'E1_131072_FIX_DONE' "$SERIAL" 2>/dev/null); then
  log "timed out; aborting without touching anything"; exit 1
fi

stop_port() {
  local pid
  pid="$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
log "freeing all GPUs (stop 8260/8261/8270/8271)"
for port in 8260 8261 8270 8271; do stop_port "$port"; done
sleep 15
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while read -r q; do log "  $q"; done

ln -sf "$D/results/e1_r1/AREAL_r1.jsonl" "$RUN_DIR/r1/AREAL_e1.jsonl"
ln -sf "$D/results/e1_r1/TRAIN_r1.jsonl" "$RUN_DIR/r1/TRAIN_e1.jsonl"

log "delegating to repository pipeline: SMOKE=1 10_4b_rl_pipeline.sh select train"
cd "$REPO"
start=$(date +%s)
SMOKE=1 bash experiments/tau-2/10_4b_rl_pipeline.sh "$CONFIG" select train >> "$LOG" 2>&1
rc=$?
end=$(date +%s)
log "SECTION_I_PIPELINE EXIT rc=$rc elapsed=$((end-start))s"
echo "SECTION_I_PIPELINE EXIT rc=$rc elapsed=$((end-start))s" >> "$SERIAL"
exit "$rc"
