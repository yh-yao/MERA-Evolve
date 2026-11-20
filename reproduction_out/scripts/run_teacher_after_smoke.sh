#!/usr/bin/env bash
# After the Section I smoke passes, complete teacher Step 0 (fill seed1300 + repair error rows).
# Teacher needs GPU2 (9B) + GPU3 (4B), which the smoke's rollout also uses, so it must run
# strictly after the smoke exits rc=0.
set -uo pipefail

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
SERIAL=$D/logs/serial_progress.log
LOG=$D/logs/teacher_after_smoke.log
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

BASE=$(grep -c 'SECTION_I_PIPELINE EXIT' "$SERIAL" 2>/dev/null || echo 0)
log "armed: waiting for a NEW rc=0 smoke exit (baseline lines=$BASE)"
for _ in $(seq 1 4320); do
  NEW=$(grep 'SECTION_I_PIPELINE EXIT' "$SERIAL" 2>/dev/null | tail -n +$((BASE + 1)) | tail -1)
  if [[ -n "$NEW" ]]; then
    if [[ "$NEW" == *"rc=0"* ]]; then break; fi
    log "smoke failed: $NEW"; exit 1
  fi
  sleep 10
done
[[ "$NEW" == *"rc=0"* ]] || { log "timeout"; exit 1; }

# Free the smoke's servers before the teacher pair binds.
for port in 8590 8591 8260 8261 8270 8271; do
  pid="$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
done
sleep 15

log "smoke passed; completing teacher Step 0"
bash "$REPO/reproduction_out/scripts/complete_teacher_step0.sh" >> "$LOG" 2>&1
rc=$?
log "TEACHER_COMPLETE_EXIT rc=$rc"
echo "TEACHER_COMPLETE EXIT rc=$rc" >> "$SERIAL"
exit "$rc"
