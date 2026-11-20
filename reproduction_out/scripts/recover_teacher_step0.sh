#!/usr/bin/env bash
# Recover teacher Step 0 after a runaway generation stalled seed300 at 267/268 for >45 min.
# Keeps all completed work: kills only the stuck collector, re-runs seed300 with --resume
# (1 missing trial), then runs seed1300, then writes the completion marker.
# Every collection call is the repository's own tau2_evolve.collect_traces with the SAME
# parameters the original run used (so the three Step-0 artifacts stay comparable).
# A per-seed `timeout` is added so a second runaway cannot stall the queue indefinitely;
# partial rows persist and can be resumed again.
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
SERIAL=$D/logs/serial_progress.log
LOG=$D/logs/teacher_step0_recovery.log
PY="$TAU2_WORKSPACE/.venv_tau2/bin/python"
mkdir -p "$D/logs"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "killing stuck collector (seed300 airline/44 trial)"
pkill -f 'tau2_evolve[.]collect' 2>/dev/null || true
sleep 10
log "collectors alive after kill: $(pgrep -f 'tau2_evolve[.]collect' | wc -l)"
for _ in $(seq 1 30); do pgrep -f 'rebuild_teacher_pair[_]thinkon' >/dev/null || break; sleep 2; done
pkill -f 'rebuild_teacher_pair[_]thinkon' 2>/dev/null || true
log "teacher watcher alive: $(pgrep -f 'rebuild_teacher_pair[_]thinkon' | wc -l)"

if ! curl -sf --max-time 5 http://127.0.0.1:8270/v1/models 2>/dev/null | grep -q evol-llm-teacher; then
  log "ABORT: teacher server 8270 is down"; exit 1
fi
if ! curl -sf --max-time 5 http://127.0.0.1:8271/v1/models 2>/dev/null | grep -q evol-llm-user2; then
  log "ABORT: user2 server 8271 is down"; exit 1
fi

export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json
cd "$REPO"

run_val() { # seed outfile extra_args...
  local seed=$1 out=$2; shift 2
  log "collect VAL seed=$seed -> $out $*"
  OPENAI_API_KEY=EMPTY timeout 5400 "$PY" -m tau2_evolve.collect_traces \
    --bucket VAL --trials 4 --seed "$seed" \
    --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
    --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
    --user-temperature 0.7 --agent-temperature 0.0 --agent-thinking --no-user-thinking \
    --workers 24 --max-steps 60 --out "$out" "$@" \
    >> "$D/logs/e6_teacher_val_seed${seed}_thinkon.log" 2>&1
  local rc=$?
  log "seed=$seed rc=$rc rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
  return $rc
}

run_val 300  "$D/results/e6_teacher_val/teacher_VAL_seed300_thinkon.jsonl"  --resume
run_val 1300 "$D/results/e6_teacher_val/teacher_VAL_seed1300_thinkon.jsonl" --resume

n300=$(wc -l < "$D/results/e6_teacher_val/teacher_VAL_seed300_thinkon.jsonl" 2>/dev/null || echo 0)
n1300=$(wc -l < "$D/results/e6_teacher_val/teacher_VAL_seed1300_thinkon.jsonl" 2>/dev/null || echo 0)
log "final rows: seed300=$n300 seed1300=$n1300 (target 268 each)"
if [[ "$n300" -ge 268 && "$n1300" -ge 268 ]]; then
  echo "[$(date -Is)] TEACHER_STEP0_THINKON_DONE rows=464+$n300+$n1300" >> "$SERIAL"
  log "marker written: TEACHER_STEP0_THINKON_DONE"
else
  log "NOT writing marker: row targets not met"
  exit 1
fi
