#!/usr/bin/env bash
# research_plan.md §7 (Q4): final EVAL report for the 4B base and the best available model
# (here: the SFT-only adapter), under the same frozen Qwen3.5-4B user simulator and the same
# 08 decoding protocol as VAL (agent greedy, 1024-token turns, user temperature 0.7, 4 trials,
# seeds 300 + 1300, max-steps 60). EVAL is reported once and is never used for selection.
#
# Each policy runs its two seeds as two parallel 08 instances, agent and user on separate GPUs
# (the E0 layout), with separate RESULTS_DIRs so the served-adapter conversion cannot race.
#   base: GPU0/1 (seed 300, ports 8290/8291) + GPU2/3 (seed 1300, ports 8292/8293)
#   sft : same layout, ports 8294/8295 and 8296/8297
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
OUT=$D/results/final_eval
LOG=$D/logs/final_eval.log
SFT_ADAPTER=$D/results/sft_only/sft_adapter
VLLM_BIN=${VLLM_BIN:-/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm}
mkdir -p "$OUT" "$D/logs"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

run_seed() { # label adapter seed agent_gpu user_gpu port out_dir
  local label=$1 adapter=$2 seed=$3 ag=$4 ug=$5 port=$6 out=$7
  mkdir -p "$out"
  log "$label seed $seed: agent GPU$ag :$port, user GPU$ug :$((port + 1)) -> $out"
  ( cd "$REPO" && \
    RESULTS_DIR="$out" BUCKETS=EVAL RUN_SEEDS="$seed" RUN_LABEL="$label" TRIALS=4 \
    BASE_MODEL="$MODELS_DIR/Qwen3.5-4B" USER_MODEL_PATH="$MODELS_DIR/Qwen3.5-4B" \
    AGENT_ADAPTER="$adapter" AGENT_GPU="$ag" USER_GPU="$ug" \
    AGENT_PORT="$port" USER_PORT=$((port + 1)) \
    VLLM_BIN="$VLLM_BIN" \
      bash experiments/tau-2/08_phase0_eval_protocol.sh ) >> "$LOG" 2>&1
}

stop_ports() { for port in "$@"; do
  pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
done; sleep 10; }

run_policy() { # label adapter p300 p1300
  local label=$1 adapter=$2 p300=$3 p1300=$4
  log "=== EVAL policy=$label (adapter=${adapter:-none}) ==="
  local t0; t0=$(date +%s)
  run_seed "$label" "$adapter" 300  0 1 "$p300"  "$OUT/$label/seed300"  & local p1=$!
  run_seed "$label" "$adapter" 1300 2 3 "$p1300" "$OUT/$label/seed1300" & local p2=$!
  wait "$p1"; local rc1=$?
  wait "$p2"; local rc2=$?
  stop_ports "$p300" $((p300 + 1)) "$p1300" $((p1300 + 1))
  log "EVAL $label EXIT rc1=$rc1 rc2=$rc2 elapsed=$(( $(date +%s) - t0 ))s"
  echo "FINAL_EVAL_${label^^} EXIT rc1=$rc1 rc2=$rc2" >> "$D/logs/serial_progress.log"
}

run_policy base "" 8290 8292
if [[ -s "$SFT_ADAPTER/adapter_model.safetensors" ]]; then
  run_policy sft "$SFT_ADAPTER" 8294 8296
else
  log "no SFT adapter; skipping the sft EVAL policy"
fi

# ---- analysis: pooled pass^1/pass^4 per policy + paired comparison base vs sft
export PYTHONPATH="$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat"
"$TAU2_WORKSPACE/.venv_tau2/bin/python" - "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/powerop/work/yjh/MERA-Evolve/experiments/tau-2")
from tau2_evolve.eval_stats import load_runs, summarize, compare

OUT = Path(sys.argv[1])
def files(label):
    return [OUT / label / f"seed{s}" / f"{label}_EVAL_seed{s}.jsonl" for s in (300, 1300)]
def have(fs): return all(f.exists() and f.stat().st_size > 0 for f in fs)

runs = {}
for label in ("base", "sft"):
    fs = files(label)
    if not have(fs):
        print(f"{label}: missing {[str(f) for f in fs if not f.exists()]}")
        continue
    runs[label] = fs
    s = summarize(load_runs(fs), k=4)
    o = s["overall"]
    print(f"{label}: EVAL n={o['n_tasks']} pass^1={100*o['pass^1']:.1f}% pass^4={100*o['pass^4']:.1f}%")
    for d, b in s["domains"].items():
        print(f"   {d:8s} n={b['n_tasks']:2d} pass^1={100*b['pass^1']:.1f}% pass^4={100*b['pass^4']:.1f}%")

if len(runs) == 2:
    c = compare(load_runs(runs["base"]), load_runs(runs["sft"]))
    print(f"paired base -> sft: delta={100*c['delta_pass^1']:+.1f}pt 95% CI "
          f"[{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}]  W/L/T {c['wins']}/{c['losses']}/{c['ties']}")
    json.dump(c, (OUT / "base_vs_sft_EVAL.json").open("w"), indent=1)
PY

log "final EVAL finished"
