#!/usr/bin/env bash
# VAL evaluation for the SFT-only adapter (research_plan.md: conclusions come from VAL, and an
# adapter must be reloaded with an active-adapter assertion). Uses the repository's 08_phase0
# protocol with its defaults (agent turns 1024, VAL user temperature 0.7, 4 trials, max-steps 60)
# so the baseline is results/e0_base_val, collected under the same protocol. 08 converts the VERL
# LoRA key names for vLLM and asserts that log-probs differ from base.
#
# 2026-11-19: the two VAL seeds are independent runs, so run them as two 08 instances in
# parallel (agent and user each on their own GPU, exactly as in E0) instead of sequentially:
#   seed 300  -> agent GPU0, user GPU1, ports 8282/8283, results in eval/seed300
#   seed 1300 -> agent GPU2, user GPU3, ports 8284/8285, results in eval/seed1300
# Each instance gets its own RESULTS_DIR, so the served-adapter conversion cannot race.
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
R=$D/results/sft_only
LOG=$D/logs/sft_eval.log
VLLM_BIN=${VLLM_BIN:-/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm}
mkdir -p "$R/eval" "$D/logs"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[[ -s "$R/sft_adapter/adapter_model.safetensors" ]] || { log "no SFT adapter at $R/sft_adapter"; exit 1; }

run_seed() { # seed agent_gpu user_gpu port out_dir
  local seed=$1 ag=$2 ug=$3 port=$4 out=$5
  mkdir -p "$out"
  log "seed $seed: agent GPU$ag :$port, user GPU$ug :$((port + 1)) -> $out"
  ( cd "$REPO" && \
    RESULTS_DIR="$out" BUCKETS=VAL RUN_SEEDS="$seed" RUN_LABEL=sft TRIALS=4 \
    BASE_MODEL="$MODELS_DIR/Qwen3.5-4B" USER_MODEL_PATH="$MODELS_DIR/Qwen3.5-4B" \
    AGENT_ADAPTER="$R/sft_adapter" AGENT_GPU="$ag" USER_GPU="$ug" \
    AGENT_PORT="$port" USER_PORT=$((port + 1)) \
    VLLM_BIN="$VLLM_BIN" \
      bash experiments/tau-2/08_phase0_eval_protocol.sh ) >> "$LOG" 2>&1
}

log "SFT VAL eval (2 seeds in parallel): adapter=$R/sft_adapter"
run_seed 300  0 1 8282 "$R/eval/seed300"  & p1=$!
run_seed 1300 2 3 8284 "$R/eval/seed1300" & p2=$!
wait "$p1"; rc1=$?
wait "$p2"; rc2=$?
log "SFT_EVAL seed300 rc=$rc1 seed1300 rc=$rc2"
echo "SFT_EVAL EXIT rc1=$rc1 rc2=$rc2" >> "$D/logs/serial_progress.log"

# 08 leaves its setsid servers running; release the GPUs.
for port in 8282 8283 8284 8285; do
  pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
done

c1="$R/eval/seed300/sft_VAL_seed300.jsonl"
c2="$R/eval/seed1300/sft_VAL_seed1300.jsonl"
if [[ -s "$c1" && -s "$c2" ]]; then
  export PYTHONPATH="$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat"
  "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.eval_stats compare \
    --baseline "$D/results/e0_base_val/base_VAL_seed300.jsonl" "$D/results/e0_base_val/base_VAL_seed1300.jsonl" \
    --candidate "$c1" "$c2" \
    --json "$R/eval/sft_vs_base.json" 2>&1 | sed 's/^/SFT vs E0 base: /' | tee -a "$LOG"
else
  log "missing candidate rows (seed300=$(wc -l < "$c1" 2>/dev/null || echo 0), seed1300=$(wc -l < "$c2" 2>/dev/null || echo 0)); compare skipped"
fi
