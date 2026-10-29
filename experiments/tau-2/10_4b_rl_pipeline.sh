#!/usr/bin/env bash
# TAU-2 4B RL pipeline (research_plan.md, E0-E2): one config file, resumable stages.
#
#   bash experiments/tau-2/10_4b_rl_pipeline.sh <config.env> [stage ...]
#
# Stages (default: all, in this order; each writes $RUN_DIR/.done_<stage> and is skipped next time):
#   check     environment, models, data, GPUs; builds the AReaL valid-task list if missing
#   base      student base on VAL (Phase 0 protocol) and on the 58 frontier TRAIN tasks
#   r1        student base, R1_TRIALS samples per AReaL / TRAIN task (sharded over EVAL_GPUS)
#   select    tasks with 0 < p < 1 -> GRPO prompts (DATA_MIX=areal+train | train)
#   train     GRPO, official tau2 reward only (09_phase1_opd_ablation.sh group O)
#   evalckpt  every checkpoint -> LoRA adapter -> frontier eval -> paired comparison with base
#             (WATCH=1 keeps polling until training has finished; run it next to `train`)
#   val       VAL for checkpoints that passed the frontier bar (or VAL_CKPTS="N M")
# Re-run a finished stage with FORCE=1. Template config: experiments/tau-2/config/pipeline_4b_rl.env.example
# SMOKE=1 shrinks every stage (a handful of tasks, 2 GRPO steps) to check the plumbing end to end.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:?usage: 10_4b_rl_pipeline.sh <config.env> [stage ...]}"; shift
# shellcheck disable=SC1090
source "$CONFIG"

# ---------------------------------------------------------------- defaults
: "${RUN_DIR:?RUN_DIR (absolute) must be set in the config}"
: "${STUDENT_MODEL:?}" "${USER_MODEL:?}" "${VLLM_BIN:?}" "${TAU2_WORKSPACE:?}"
TRAIN_VENV="${TRAIN_VENV:-$ROOT/.venv_qwen35}"
TAU2_PY="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
USE_AREAL="${USE_AREAL:-1}"; MERA_AREAL_DIR="${MERA_AREAL_DIR:-}"
DATA_MIX="${DATA_MIX:-areal+train}"
SERVE_GPU="${SERVE_GPU:?}" TRAIN_GPUS="${TRAIN_GPUS:?}" ROLLOUT_GPUS="${ROLLOUT_GPUS:?}" EVAL_GPUS="${EVAL_GPUS:?}"
PORT_BASE="${PORT_BASE:-8500}"
R1_TRIALS="${R1_TRIALS:-4}"
ACTOR_LR="${ACTOR_LR:-1e-5}" TOTAL_STEPS="${TOTAL_STEPS:-200}" SAVE_EVERY="${SAVE_EVERY:-25}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" N_GENERATIONS="${N_GENERATIONS:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}" MAX_TURNS="${MAX_TURNS:-30}"
LORA_RANK="${LORA_RANK:-16}" LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}"     # bf16 on 32 GB cards
CHECKPOINT_BUCKET_MB="${CHECKPOINT_BUCKET_MB:-2560}"  # must exceed the largest tensor (4B bf16 embedding ~1.27 GB)
ENABLE_TIS="${ENABLE_TIS:-1}" TIS_THRESHOLD="${TIS_THRESHOLD:-2.0}"
PASS_DELTA="${PASS_DELTA:-0.05}" VAL_SEEDS="${VAL_SEEDS:-300 1300}"
SMOKE="${SMOKE:-0}" FORCE="${FORCE:-0}" WATCH="${WATCH:-0}"
if [[ "$SMOKE" == 1 ]]; then
  LIMIT="--limit ${SMOKE_TASKS:-4}"; R1_TRIALS=2; FRONTIER_TRIALS=2; VAL_TRIALS=2
  TOTAL_STEPS=2; SAVE_EVERY=1; TRAIN_BATCH_SIZE=4; N_GENERATIONS=4; VAL_SEEDS=300  # batch x n must be a multiple of the 16 agent-loop workers
else
  LIMIT=""; FRONTIER_TRIALS=8; VAL_TRIALS=4
fi
FRONTIER_PARTITION="${FRONTIER_PARTITION:-$SCRIPT_DIR/data/partition_frontier_only.json}"
export TAU2_WORKSPACE MERA_AREAL_DIR VLLM_BIN
export TAU2_PARTITION_PATH="$SCRIPT_DIR/data/partition_v2.json"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/compat:$ROOT:${PYTHONPATH:-}"
export TAU2_SITE_PACKAGES="${TAU2_SITE_PACKAGES:-$(ls -d "$TAU2_WORKSPACE"/.venv_tau2/lib/python3*/site-packages 2>/dev/null | head -1)}"
mkdir -p "$RUN_DIR"; LOG="$RUN_DIR/pipeline.log"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
die() { log "FATAL $*"; exit 1; }
read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS//,/ }"
stage_done() { [[ "$FORCE" != 1 && -f "$RUN_DIR/.done_$1" ]]; }
mark_done() { touch "$RUN_DIR/.done_$1"; log "stage $1 done"; }
cp -n "$CONFIG" "$RUN_DIR/config.env" 2>/dev/null || true

stop_port() {
  local pid; pid=$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
}

# collect <out.jsonl> <gpu> <port> <adapter|-> <partition> -- <collect_traces args>
# Student and user simulator (both served by 08 through chunked_collect) share one GPU.
collect() {
  local out=$1 gpu=$2 port=$3 adapter=$4 partition=$5; shift 6
  local sdir="${out%.jsonl}_servers"; mkdir -p "$sdir"
  SERVERS_DIR="$sdir" TAU2_PARTITION_PATH="$partition" BASE_MODEL="$STUDENT_MODEL" USER_MODEL_PATH="$USER_MODEL" \
  AGENT_ADAPTER="${adapter#-}" AGENT_GPU="$gpu" USER_GPU="$gpu" AGENT_PORT="$port" USER_PORT=$((port + 1)) \
  GPU_MEMORY_UTILIZATION=0.42 VLLM_MAX_NUM_SEQS=128 CHUNK_JOBS="${CHUNK_JOBS:-128}" REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}" \
    bash "$SCRIPT_DIR/scripts/chunked_collect.sh" "$@" $LIMIT \
      --agent-model openai/evol-llm-agent --user-model openai/evol-llm-user \
      --agent-max-tokens 1024 --no-agent-thinking --no-user-thinking --out "$out" > "${out%.jsonl}.log" 2>&1
  local rc=$?
  stop_port "$port"; stop_port $((port + 1))
  return $rc
}
val_args()      { echo --bucket VAL --trials "$VAL_TRIALS" --seed "$1" --agent-temperature 0.0 --user-temperature 0.7 --workers 64 --max-steps 60; }
frontier_args() { echo --bucket TRAIN --trials "$FRONTIER_TRIALS" --seed 2000 --agent-temperature 1.0 --user-temperature 0.0 --workers 64 --max-steps 40; }

# ---------------------------------------------------------------- stages
stage_check() {
  log "check: driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), GPUs $(nvidia-smi -L | wc -l)"
  for f in "$STUDENT_MODEL/config.json" "$USER_MODEL/config.json" "$VLLM_BIN" "$TAU2_PY" "$TRAIN_VENV/bin/python"; do
    [[ -e "$f" ]] || die "missing $f"
  done
  "$TAU2_PY" -c "import tau2" > /dev/null 2>&1 || die "tau2 not importable from $TAU2_PY"
  "$TRAIN_VENV/bin/python" -c "import torch, verl, vllm, fla; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tail -1 | tee -a "$LOG"
  "$TRAIN_VENV/bin/python" -c "import causal_conv1d" 2>/dev/null \
    || log "WARNING causal_conv1d missing in TRAIN_VENV: Qwen3.5 GDN falls back to the slow torch path"
  if [[ "$USE_AREAL" == 1 ]]; then
    [[ -s "$MERA_AREAL_DIR/tau2_rl_train.jsonl" ]] || die "MERA_AREAL_DIR has no tau2_rl_train.jsonl"
    if [[ ! -s "$MERA_AREAL_DIR/valid_task_ids.json" ]]; then
      log "check: replaying AReaL gold actions to build valid_task_ids.json (CPU, ~10 min)"
      "$TAU2_PY" "$SCRIPT_DIR/scripts/check_areal_tasks.py" --write-valid "$MERA_AREAL_DIR/valid_task_ids.json" \
        > "$RUN_DIR/check_areal.log" 2>&1 || die "check_areal_tasks failed (see check_areal.log)"
    fi
    log "check: $("$TAU2_PY" -c "import json; print(len(json.load(open('$MERA_AREAL_DIR/valid_task_ids.json'))), 'valid AReaL tasks')")"
  fi
}

stage_base() {
  local d="$RUN_DIR/base"; mkdir -p "$d"
  local g1=${EVAL_GPU_LIST[0]} g2=${EVAL_GPU_LIST[1]:-${EVAL_GPU_LIST[0]}}
  log "base: VAL on GPU $g1, frontier on GPU $g2"
  collect "$d/val_seed300.jsonl" "$g1" "$PORT_BASE" - "$TAU2_PARTITION_PATH" -- $(val_args 300) &
  local p1=$!
  if [[ "$g2" == "$g1" ]]; then wait $p1; fi
  collect "$d/frontier.jsonl" "$g2" $((PORT_BASE + 2)) - "$FRONTIER_PARTITION" -- $(frontier_args)
  wait $p1
  "$TAU2_PY" -m tau2_evolve.eval_stats summary "$d/val_seed300.jsonl" 2>&1 | tee -a "$LOG"
  [[ -s "$d/frontier.jsonl" ]] || die "base frontier produced no rows"
  log "base frontier pass^1: $("$TAU2_PY" -c "import json; r=[json.loads(l) for l in open('$d/frontier.jsonl')]; print(round(sum(x['passed'] for x in r)/len(r),3), len(r), 'rows')")"
}

stage_r1() {
  local d="$RUN_DIR/r1"; mkdir -p "$d"
  local n=${#EVAL_GPU_LIST[@]} jobs=() i
  if [[ "$USE_AREAL" == 1 ]]; then for ((i = 0; i < n; i++)); do jobs+=("AREAL:$i/$n"); done; fi
  jobs+=("TRAIN:0/1")
  log "r1: ${#jobs[@]} shards over GPUs ${EVAL_GPU_LIST[*]}"
  local pids=()
  for ((i = 0; i < n; i++)); do
    (
      for ((j = i; j < ${#jobs[@]}; j += n)); do
        local b=${jobs[$j]%%:*} s=${jobs[$j]#*:}
        collect "$d/${b}_${s//\//of}.jsonl" "${EVAL_GPU_LIST[$i]}" $((PORT_BASE + 10 + 2 * i)) - "$TAU2_PARTITION_PATH" -- \
          --bucket "$b" --shard "$s" --trials "$R1_TRIALS" --seed 4000 --agent-temperature 1.0 --user-temperature 0.0 \
          --workers 64 --max-steps 40
      done
    ) & pids+=($!)
  done
  wait "${pids[@]}"
  log "r1: $(cat "$d"/*.jsonl 2>/dev/null | wc -l) rollouts"
}

stage_select() {
  local d="$RUN_DIR/data"; mkdir -p "$d"
  python3 "$SCRIPT_DIR/scripts/select_frontier.py" --traces "$RUN_DIR/r1/TRAIN_*.jsonl" --out "$d/train_frontier.jsonl" \
    --min-trials "$R1_TRIALS" | tee -a "$LOG"
  : > "$d/areal_frontier.jsonl"
  if [[ "$USE_AREAL" == 1 && "$DATA_MIX" == "areal+train" ]]; then
    python3 "$SCRIPT_DIR/scripts/select_frontier.py" --traces "$RUN_DIR/r1/AREAL_*.jsonl" --out "$d/areal_frontier.jsonl" \
      --min-trials "$R1_TRIALS" | tee -a "$LOG"
  fi
  cat "$d/areal_frontier.jsonl" "$d/train_frontier.jsonl" | python3 -c "
import random, sys; rows = sys.stdin.readlines(); random.Random(1).shuffle(rows); sys.stdout.writelines(rows)" > "$d/grpo_traces.jsonl"
  if [[ "$SMOKE" == 1 && $(wc -l < "$d/grpo_traces.jsonl") -lt 4 ]]; then
    log "select: SMOKE with < 4 frontier tasks; using the first R1 rollouts regardless of p"
    cat "$RUN_DIR"/r1/*.jsonl | head -8 > "$d/grpo_traces.jsonl"
  fi
  local n; n=$(wc -l < "$d/grpo_traces.jsonl"); [[ "$n" -gt 0 ]] || die "no GRPO prompts selected"
  log "select: $n GRPO prompts -> $d/grpo_traces.jsonl"
}

stage_train() {
  local workers="${AGENT_LOOP_WORKERS:-16}"
  (( TRAIN_BATCH_SIZE * N_GENERATIONS % workers == 0 )) \
    || die "TRAIN_BATCH_SIZE x N_GENERATIONS ($TRAIN_BATCH_SIZE x $N_GENERATIONS) must be a multiple of AGENT_LOOP_WORKERS ($workers)"
  local extra="actor_rollout_ref.actor.fsdp_config.model_dtype=$ACTOR_MODEL_DTYPE"
  # VERL's one-step off-policy trainer defaults to bypass_mode=True: old_log_prob := rollout log-prob and the
  # PPO ratio pi_theta / pi_rollout is clipped, so rollout_is weights are ignored. TIS needs decoupled mode.
  [[ "$ENABLE_TIS" == 1 ]] && extra+=" algorithm.rollout_correction.bypass_mode=False algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=$TIS_THRESHOLD actor_rollout_ref.rollout.calculate_log_probs=True"
  # The parquet embeds the user-simulator URL, so build it here from the current PORT_BASE.
  "$TRAIN_VENV/bin/python" -m tau2_evolve.prepare_grpo_data --traces "$RUN_DIR/data/grpo_traces.jsonl" \
    --output "$RUN_DIR/data/grpo.parquet" --user-base-url "http://127.0.0.1:$((PORT_BASE + 91))/v1" \
    --no-user-thinking --max-steps 40 --no-balance-domains 2>&1 | tail -1 | tee -a "$LOG"
  [[ -s "$RUN_DIR/data/grpo.parquet" ]] || die "prepare_grpo_data failed"
  log "train: lr $ACTOR_LR, $TOTAL_STEPS steps x $TRAIN_BATCH_SIZE prompts x $N_GENERATIONS, response cap $MAX_RESPONSE_LENGTH, TIS $ENABLE_TIS, train GPUs $TRAIN_GPUS, rollout GPUs $ROLLOUT_GPUS"
  env RESULTS_DIR="$RUN_DIR/train" PHASE1_GROUPS=O SKIP_EVAL=1 GRPO_PARQUET="$RUN_DIR/data/grpo.parquet" \
    ACTOR_LR="$ACTOR_LR" TOTAL_STEPS="$TOTAL_STEPS" SAVE_EVERY="$SAVE_EVERY" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    N_GENERATIONS="$N_GENERATIONS" MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    MAX_TURNS="$MAX_TURNS" LORA_RANK="$LORA_RANK" LORA_ALPHA="$LORA_ALPHA" LORA_TARGET_MODULES="$LORA_TARGET_MODULES" \
    CHECKPOINT_BUCKET_MB="$CHECKPOINT_BUCKET_MB" EXTRA_TRAIN_ARGS="$extra" \
    TRAIN_GPU="$TRAIN_GPUS" ROLLOUT_GPU="$ROLLOUT_GPUS" AGENT_GPU="$SERVE_GPU" USER_GPU="$SERVE_GPU" \
    AGENT_PORT=$((PORT_BASE + 90)) USER_PORT=$((PORT_BASE + 91)) GPU_MEMORY_UTILIZATION=0.42 VLLM_MAX_NUM_SEQS=128 \
    BASE_MODEL="$STUDENT_MODEL" USER_MODEL_PATH="$USER_MODEL" TRAIN_VENV="$TRAIN_VENV" \
    bash "$SCRIPT_DIR/09_phase1_opd_ablation.sh" > "$RUN_DIR/train.driver.log" 2>&1
  local rc=$?
  stop_port $((PORT_BASE + 90)); stop_port $((PORT_BASE + 91))
  local tl="$RUN_DIR/train/group_O/train.log"
  [[ -s "$RUN_DIR/train/group_O/verl/final_adapter_path.txt" ]] || die "training ended without a final adapter (rc $rc; see $tl)"
  local clip; clip=$(grep -oE "response_length/clip_ratio:[0-9.]+" "$tl" | awk -F: '{s+=$2; n++} END {if (n) printf "%.3f", s/n}')
  log "train: mean truncation (response_length/clip_ratio) = ${clip:-n/a}"
  awk -v c="${clip:-0}" 'BEGIN {exit !(c > 0.05)}' && log "WARNING truncation above 5%: raise MAX_RESPONSE_LENGTH (research_plan.md E2)"
  return 0
}

eval_one_ckpt() { # step gpu port
  local s=$1 gpu=$2 port=$3 d="$RUN_DIR/eval/step_$1"
  [[ -f "$d/compare.json" ]] && return 0
  mkdir -p "$d"
  if [[ ! -s "$d/adapter/adapter_model.safetensors" ]]; then
    "$TRAIN_VENV/bin/python" -m verl_code_rl.extract_sft_lora_adapter \
      --checkpoint-dir "$RUN_DIR/train/group_O/verl/global_step_$s/actor" --base-model "$STUDENT_MODEL" --output "$d/adapter" \
      --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --target-modules "$LORA_TARGET_MODULES" > "$d/extract.log" 2>&1 \
      || { log "evalckpt: step $s adapter extraction failed"; echo x >> "$d/.failures"; return 1; }
  fi
  collect "$d/frontier.jsonl" "$gpu" "$port" "$d/adapter" "$FRONTIER_PARTITION" -- $(frontier_args) \
    || { log "evalckpt: step $s frontier collection failed (see $d/frontier.log)"; echo x >> "$d/.failures"; return 1; }
  python3 "$SCRIPT_DIR/scripts/compare_tasks.py" --base "$RUN_DIR/base/frontier.jsonl" --cand "$d/frontier.jsonl" \
    --min-delta "$PASS_DELTA" --json "$d/compare.json" | sed "s/^/evalckpt: step $s frontier vs base: /" | tee -a "$LOG"
}

stage_evalckpt() {
  [[ -s "$RUN_DIR/base/frontier.jsonl" ]] || die "run stage base first"
  while true; do
    local pending=() s
    for ck in $(ls -d "$RUN_DIR"/train/group_O/verl/global_step_* 2>/dev/null | sort -V); do
      s=${ck##*_}
      [[ -f "$RUN_DIR/eval/step_$s/compare.json" ]] && continue
      [[ $(cat "$RUN_DIR/eval/step_$s/.failures" 2>/dev/null | wc -l) -ge 2 ]] && continue  # gave up after two failures
      pending+=("$s")
    done
    local i=0 pids=()
    for s in "${pending[@]:0:${#EVAL_GPU_LIST[@]}}"; do   # one checkpoint per eval GPU at a time
      eval_one_ckpt "$s" "${EVAL_GPU_LIST[$i]}" $((PORT_BASE + 40 + 2 * i)) & pids+=($!)
      i=$((i + 1))
    done
    if [[ ${#pids[@]} -gt 0 ]]; then wait "${pids[@]}"; continue; fi
    # nothing left to evaluate: stop, unless WATCH=1 and training is still writing checkpoints
    [[ "$WATCH" == 1 && ! -s "$RUN_DIR/train/group_O/verl/final_adapter_path.txt" ]] || break
    sleep 300
  done
  for f in "$RUN_DIR"/eval/step_*/compare.json; do
    [[ -f "$f" ]] || continue
    python3 -c "import json; r=json.load(open('$f')); print('$(basename "$(dirname "$f")")', '%+.3f' % r['delta'], 'CI[%+.3f,%+.3f]' % (r['ci_low'], r['ci_high']), 'PASS' if r['passes'] else '')"
  done | sort -V | sed 's/^/evalckpt summary: /' | tee -a "$LOG"
}

stage_val() {
  local ckpts="${VAL_CKPTS:-}"
  if [[ -z "$ckpts" ]]; then
    ckpts=$(for f in "$RUN_DIR"/eval/step_*/compare.json; do
      [[ -f "$f" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$f'))['passes'] else 1)" && basename "$(dirname "$f")" | cut -d_ -f2
    done | sort -n | tr '\n' ' ')
  fi
  [[ -n "$ckpts" ]] || { log "val: no checkpoint passed the frontier bar (delta >= $PASS_DELTA, CI low > 0); nothing to run"; return 0; }
  log "val: checkpoints $ckpts, seeds $VAL_SEEDS"
  for s in $ckpts; do
    for seed in $VAL_SEEDS; do
      local base_val="$RUN_DIR/base/val_seed$seed.jsonl"
      [[ -s "$base_val" ]] || collect "$base_val" "${EVAL_GPU_LIST[0]}" $((PORT_BASE + 60)) - "$TAU2_PARTITION_PATH" -- $(val_args "$seed")
      collect "$RUN_DIR/eval/step_$s/val_seed$seed.jsonl" "${EVAL_GPU_LIST[0]}" $((PORT_BASE + 62)) \
        "$RUN_DIR/eval/step_$s/adapter" "$TAU2_PARTITION_PATH" -- $(val_args "$seed")
    done
    local b=() c=()
    for seed in $VAL_SEEDS; do b+=("$RUN_DIR/base/val_seed$seed.jsonl"); c+=("$RUN_DIR/eval/step_$s/val_seed$seed.jsonl"); done
    "$TAU2_PY" -m tau2_evolve.eval_stats compare --baseline "${b[@]}" --candidate "${c[@]}" 2>&1 | sed "s/^/val step $s: /" | tee -a "$LOG"
  done
}

# ---------------------------------------------------------------- driver
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(check base r1 select train evalckpt val)
log "pipeline start: stages ${STAGES[*]} (SMOKE=$SMOKE) config $CONFIG"
for st in "${STAGES[@]}"; do
  declare -F "stage_$st" > /dev/null || die "unknown stage $st"
  if stage_done "$st"; then log "stage $st already done (FORCE=1 to rerun)"; continue; fi
  log "stage $st start"
  "stage_$st" || die "stage $st failed"
  [[ "$st" == evalckpt && "$WATCH" != 1 ]] || mark_done "$st"
done
log "pipeline finished"
