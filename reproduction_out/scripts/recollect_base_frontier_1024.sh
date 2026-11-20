#!/usr/bin/env bash
# Option A of the Step-0 budget audit: keep the teacher frontier at --agent-max-tokens 1024
# and re-collect the E0 base frontier with the SAME cap, so both sides satisfy
# research_plan.md Step 0's "matched generation budgets".
#
# Delegates entirely to the repository's own scripts:
#   experiments/tau-2/scripts/chunked_collect.sh  (chunks + server restarts, GDN-stall guard)
#   experiments/tau-2/08_phase0_eval_protocol.sh  (restarts agent/user servers via BUCKETS="")
#
# MAX_MODEL_LEN must be passed explicitly: 08 defaults to 40960, which would silently
# revert the 131072 serving profile.
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
LOG=$D/logs/base_frontier_1024.log
OUT=$D/results/e0_base_val/base_frontier_max1024.jsonl
SERVERS_DIR=$D/results/servers_frontier1024
mkdir -p "$D/logs" "$SERVERS_DIR"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "base frontier re-run with --agent-max-tokens 1024 (student pair, GPU0/1)"
cd "$REPO"
SERVERS_DIR="$SERVERS_DIR" \
BASE_MODEL="$MODELS_DIR/Qwen3.5-4B" \
USER_MODEL_PATH="$MODELS_DIR/Qwen3.5-4B" \
VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm \
AGENT_GPU=0 USER_GPU=1 AGENT_PORT=8260 USER_PORT=8261 \
MAX_MODEL_LEN=131072 GPU_MEMORY_UTILIZATION=0.85 \
TAU2_PARTITION_PATH="$REPO/experiments/tau-2/data/partition_frontier_only.json" \
TAU2_WORKSPACE="$TAU2_WORKSPACE" \
CHUNK_JOBS=128 REQUEST_TIMEOUT=900 \
  bash experiments/tau-2/scripts/chunked_collect.sh \
    --out "$OUT" \
    --bucket TRAIN --trials 8 --seed 2000 \
    --agent-temperature 1.0 --user-temperature 0.0 \
    --agent-max-tokens 1024 --no-agent-thinking --no-user-thinking \
    --workers 32 --max-steps 40 >> "$LOG" 2>&1
rc=$?
n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
log "base frontier rc=$rc rows=$n (target 464)"
if [[ "$rc" == 0 && "$n" -ge 464 ]]; then
  echo "[$(date -Is)] BASE_FRONTIER_1024_DONE rows=$n" >> "$D/logs/serial_progress.log"
  log "marker written"
else
  log "NOT writing marker"; exit 1
fi
