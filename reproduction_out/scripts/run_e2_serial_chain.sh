#!/usr/bin/env bash
# One serial chain, no manual waiting between stages:
#   E2 arm 1 (already running) -> evalckpt+val -> SFT-only (student successes)
#   -> IF arm 1 passed the plan's frontier bar AND RUN_ARM2=1: E2 arm 2 (LR=3e-5) -> evalckpt+val
# This encodes research_plan.md's E2 decision rule ("one run each at 1e-5 and 3e-5";
# a checkpoint passes when frontier delta >= +5 pt and CI low > 0). All stages are the
# repository pipeline 10_4b_rl_pipeline.sh and the repository SFT runner; this file only
# sequences them.
#
# 2026-11-18: GRPO arms are paused by decision (SFT-only phase). The arm-2 branch is now
# gated on RUN_ARM2, default 0; set RUN_ARM2=1 to re-enable research_plan.md's second LR arm.
set -uo pipefail
RUN_ARM2="${RUN_ARM2:-0}"

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
SERIAL=$D/logs/serial_progress.log
LOG=$D/logs/e2_serial_chain.log
CFG1=$REPO/reproduction_out/config/pipeline_e2_arm1.env
CFG2=$REPO/reproduction_out/config/pipeline_e2_arm2.env
RUN1=$D/results/e2_arm1_lr1e-5
RUN2=$D/results/e2_arm2_lr3e-5
mkdir -p "$D/logs" "$RUN1/base" "$RUN2/base"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

wait_training() { # final_adapter_path
  for _ in $(seq 1 8640); do
    [[ -s "$1/train/group_O/verl/final_adapter_path.txt" ]] && return 0
    sleep 20
  done
  return 1
}

wait_gpus_idle() {
  for _ in $(seq 1 180); do
    [[ $(pgrep -f 'one_step_off[_]policy|03_grpo' | wc -l) == 0 ]] && break
    sleep 10
  done
  sleep 20
  # The training stage starts its own agent/user vLLM pair on GPU0 (ports 8890/8891, 0.42 each,
  # ~84 GiB together) and does not stop them when verl exits. They are idle leftovers after
  # training; free GPU0 so the eval stages can use it.
  for port in 8890 8891; do
    pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
    if [[ -n "$pid" ]]; then
      log "stopping leftover training server on :$port (pid $pid)"
      kill "$pid" 2>/dev/null
    fi
  done
  sleep 15
}

pick_eval_cfg() { # config -> config to use for this eval
  # GPU0 normally belongs to someone else on this shared node. evalckpt drops a checkpoint after
  # two failed attempts, so never point it at a GPU that cannot host a 0.42-utilization server.
  local cfg=$1 free
  free=$(python3 -c 'import torch; f,_=torch.cuda.mem_get_info(0); print(int(f/2**30))' 2>/dev/null || echo 0)
  if [[ "${free:-0}" -ge 40 ]]; then
    echo "$cfg"
  else
    local alt="${cfg%.env}_eval123.env"
    sed 's/^EVAL_GPUS=.*/EVAL_GPUS=1,2,3/' "$cfg" > "$alt"
    log "GPU0 has ${free} GiB free (<40); eval on GPUs 1,2,3 via $alt"
    echo "$alt"
  fi
}

eval_stage() { # tag config run_dir
  local tag=$1 cfg=$2 run=$3 t0 t1 rc
  cfg=$(pick_eval_cfg "$cfg")
  # Matched baseline: run the repository's own `base` stage in this RUN_DIR so that BOTH the
  # frontier comparison and the VAL comparison use identical generation arguments to evalckpt
  # (research_plan.md E2 requires matched budgets; the E0 artifact was produced by 08 with a
  # 1024 agent token cap, which would make the VAL comparison asymmetric).
  if [[ ! -s "$run/base/val_seed300.jsonl" || ! -s "$run/base/frontier.jsonl" ]]; then
    log "=== $tag: base stage (matched baseline) ==="
    t0=$(date +%s)
    ( cd "$REPO" && SMOKE=0 bash experiments/tau-2/10_4b_rl_pipeline.sh "$cfg" base ) >> "$LOG" 2>&1
    rc=$?
    log "E2_${tag}_BASE_EXIT rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
    echo "E2_${tag}_BASE EXIT rc=$rc" >> "$SERIAL"
  fi
  if [[ ! -s "$run/base/frontier.jsonl" ]]; then
    # Must be the SAME --agent-max-tokens 1024 re-run that evalckpt uses; the original
    # base_frontier.jsonl is unbounded and would silently change the comparison.
    fb="$D/results/e0_base_val/base_frontier_max1024.jsonl"
    if [[ -s "$fb" && "$(wc -l < "$fb")" -eq 464 ]]; then
      log "base stage produced no frontier; falling back to $fb (record as deviation)"
      cp "$fb" "$run/base/frontier.jsonl"
    else
      log "base stage produced no frontier and the 1024 E0 artifact is incomplete ($(wc -l < "$fb" 2>/dev/null || echo 0)/464 rows); NOT substituting"
    fi
  fi
  log "=== $tag: evalckpt val ==="
  t0=$(date +%s)
  ( cd "$REPO" && SMOKE=0 bash experiments/tau-2/10_4b_rl_pipeline.sh "$cfg" evalckpt val ) >> "$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  log "E2_${tag}_EVAL_EXIT rc=$rc elapsed=$((t1-t0))s"
  echo "E2_${tag}_EVAL EXIT rc=$rc elapsed=$((t1-t0))s" >> "$SERIAL"
  return "$rc"
}

count_passing() { # run_dir -> number of checkpoints with passes=true
  python3 - "$1" <<'PY'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/eval/step_*/compare.json"):
    try:
        if json.load(open(f)).get("passes"):
            n += 1
    except Exception:
        pass
print(n)
PY
}

log "armed: waiting for E2 arm 1 training to finish"
wait_training "$RUN1" || { log "arm 1 training timeout"; exit 1; }
wait_gpus_idle
log "arm 1 training done"

eval_stage ARM1 "$CFG1" "$RUN1" || log "arm 1 eval exited non-zero (continuing)"
PASS1=$(count_passing "$RUN1")
log "arm 1 checkpoints passing the frontier bar: $PASS1"
echo "E2_ARM1_PASS_COUNT=$PASS1" >> "$SERIAL"

# Matched-protocol base VAL (2026-11-18 decision). Teacher Step 0 compared the 9B against a
# base run whose agent turns were capped at 1024 tokens, while the teacher had no cap; the plan
# requires matched generation budgets. Re-collect the 4B base on VAL seeds 300 + 1300 with the
# teacher's budget (08 takes it from AGENT_MAX_TOKENS, no code change) and re-run the pooled
# verdict on matched settings. ~1 h on 2 GPUs; SFT uses the same two GPUs afterwards.
if [[ ! -s "$D/results/e6_teacher_val/matched_base/base_VAL_seed1300.jsonl" ]]; then
  log "=== matched-protocol base VAL (AGENT_MAX_TOKENS=131072) ==="
  t0=$(date +%s)
  ( set +u; source "$REPO/reproduction_out/env.sh"; set -u
    cd "$REPO"
    RESULTS_DIR="$D/results/e6_teacher_val/matched_base" BUCKETS=VAL RUN_SEEDS="300 1300" \
    BASE_MODEL="$MODELS_DIR/Qwen3.5-4B" USER_MODEL_PATH="$MODELS_DIR/Qwen3.5-4B" \
    AGENT_GPU=1 USER_GPU=2 AGENT_PORT=8280 USER_PORT=8281 \
    AGENT_MAX_TOKENS=131072 MAX_MODEL_LEN=131072 TRIALS=4 \
    VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm \
    bash experiments/tau-2/08_phase0_eval_protocol.sh ) >> "$LOG" 2>&1
  rc=$?
  log "MATCHED_BASE EXIT rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
  echo "MATCHED_BASE EXIT rc=$rc" >> "$SERIAL"
  # 08 leaves its setsid servers running; release GPUs 1-2 before SFT.
  for port in 8280 8281; do
    pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
  done
  sleep 15
  if [[ "$rc" == 0 ]]; then
    python3 "$REPO/reproduction_out/scripts/analyze_teacher_step0_pooled.py" \
      --base-dir "$D/results/e6_teacher_val/matched_base" --tag matched >> "$LOG" 2>&1
    log "matched-base pooled verdict: $(grep -h 'errors as FAILURES' \
      "$D/results/e6_teacher_val/STEP0_VERDICT_AMENDED_matched.txt" 2>/dev/null)"
  fi
else
  log "matched-protocol base VAL already collected; skipping"
fi

# SFT-only side track (independent of the pass/fail decision). run_sft_only.sh waits for
# E2_ARM1_EVAL which eval_stage just wrote, so it proceeds immediately.
log "=== SFT-only (student successes) ==="
bash "$REPO/reproduction_out/scripts/run_sft_only.sh" >> "$LOG" 2>&1
SFT_RC=$?
log "SFT stage finished rc=$SFT_RC"

# VAL for the SFT adapter. research_plan.md: conclusions come from VAL, and every evaluation
# must reload the adapter and pass the adapter-active assertion. 08 with the E0 protocol
# (agent turns 1024, VAL user temperature 0.7, trials 4, seeds 300 + 1300) keeps the baseline
# matched with results/e0_base_val; 08 asserts the adapter is active before collecting.
R=$D/results/sft_only
if [[ "$SFT_RC" == 0 && -s "$R/sft_adapter/adapter_model.safetensors" ]]; then
  log "=== SFT VAL eval (08, adapter=$R/sft_adapter) ==="
  t0=$(date +%s)
  ( set +u; source "$REPO/reproduction_out/env.sh"; set -u
    cd "$REPO"
    RESULTS_DIR="$R/eval" BUCKETS=VAL RUN_SEEDS="300 1300" RUN_LABEL=sft TRIALS=4 \
    BASE_MODEL="$MODELS_DIR/Qwen3.5-4B" USER_MODEL_PATH="$MODELS_DIR/Qwen3.5-4B" \
    AGENT_ADAPTER="$R/sft_adapter" AGENT_GPU=1 USER_GPU=2 AGENT_PORT=8282 USER_PORT=8283 \
    VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm \
    bash experiments/tau-2/08_phase0_eval_protocol.sh ) >> "$LOG" 2>&1
  rc=$?
  log "SFT_EVAL EXIT rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
  echo "SFT_EVAL EXIT rc=$rc" >> "$SERIAL"
  for port in 8282 8283; do
    pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
  done
  if [[ "$rc" == 0 ]]; then
    ( set +u; source "$REPO/reproduction_out/env.sh"; set -u
      export PYTHONPATH="$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat"
      "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.eval_stats compare \
        --baseline "$D/results/e0_base_val/base_VAL_seed300.jsonl" "$D/results/e0_base_val/base_VAL_seed1300.jsonl" \
        --candidate "$R/eval/sft_VAL_seed300.jsonl" "$R/eval/sft_VAL_seed1300.jsonl" \
        --json "$R/eval/sft_vs_base.json" ) 2>&1 | sed 's/^/SFT vs E0 base: /' | tee -a "$LOG"
  fi
else
  log "SFT adapter missing or SFT failed (rc=$SFT_RC); VAL eval skipped"
fi

if [[ "$RUN_ARM2" == 1 && "$PASS1" -gt 0 ]]; then
  log "=== arm 1 passed -> starting E2 arm 2 (LR=3e-5) ==="
  mkdir -p "$RUN2/r1"
  ln -sf "$D/results/e1_r1/AREAL_r1.jsonl" "$RUN2/r1/AREAL_e1.jsonl"
  ln -sf "$D/results/e1_r1/TRAIN_r1.jsonl" "$RUN2/r1/TRAIN_e1.jsonl"
  ( cd "$REPO" && SMOKE=0 bash experiments/tau-2/10_4b_rl_pipeline.sh "$CFG2" select train ) >> "$LOG" 2>&1
  rc=$?
  log "E2_ARM2_TRAIN_EXIT rc=$rc"
  echo "E2_ARM2_TRAIN EXIT rc=$rc" >> "$SERIAL"
  if [[ "$rc" == 0 ]]; then
    wait_gpus_idle
    eval_stage ARM2 "$CFG2" "$RUN2" || log "arm 2 eval exited non-zero"
    log "arm 2 checkpoints passing: $(count_passing "$RUN2")"
  fi
elif [[ "$RUN_ARM2" != 1 ]]; then
  log "RUN_ARM2=0 -> GRPO paused by decision (SFT-only phase); arm 2 not started (arm 1 passing checkpoints: $PASS1)"
else
  log "arm 1 did not pass the frontier bar -> per research_plan.md E2 decision rule, go to E4; arm 2 not started"
fi

log "E2_SERIAL_CHAIN_DONE"
echo "E2_SERIAL_CHAIN_DONE" >> "$SERIAL"
