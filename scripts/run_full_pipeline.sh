#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Auto-load .env (COMMONSTACK_API_KEY etc.) if present -- never checked in,
# .gitignore'd, safe to source into this run's environment only.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON="${PYTHON:-python}"
DATA="${DATA:-data/raw/he_mbpp.jsonl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mera_evolve_$(date -u +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-results/$EXPERIMENT_NAME}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
SMALL_SERVED_MODEL="${SMALL_SERVED_MODEL:-$SMALL_MODEL}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SMALL_BASE_URL="${SMALL_BASE_URL:-http://127.0.0.1:8000/v1}"
LARGE_BASE_URL="${LARGE_BASE_URL:-http://127.0.0.1:8001/v1}"
API_KEY="${API_KEY:-EMPTY}"
export API_KEY
N_CYCLES="${N_CYCLES:-1}"
LIMIT="${LIMIT:--1}"
WORKERS="${WORKERS:-16}"
SPLIT="${SPLIT:-train}"
DATASET="${DATASET:-all}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
# Independent of SKIP_TRAIN: skip GRPO specifically while still running SFT
# (when ENABLE_SFT=1) each cycle -- e.g. to isolate SFT+skills' own effect.
SKIP_GRPO="${SKIP_GRPO:-0}"
MOCK="${MOCK:-0}"
PROBE_ONLY="${PROBE_ONLY:-0}"
SMALL_SAMPLES="${SMALL_SAMPLES:-1}"
MAX_REPAIR_TURNS="${MAX_REPAIR_TURNS:-0}"
TRAINING_SEED="${TRAINING_SEED:-1}"
LARGE_TEMPERATURE="${LARGE_TEMPERATURE:-0.0}"
SMALL_MAX_TOKENS="${SMALL_MAX_TOKENS:-768}"
LARGE_MAX_TOKENS="${LARGE_MAX_TOKENS:-768}"
LARGE_CACHE="${LARGE_CACHE:-}"
ROUTER_TASK_MODULO="${ROUTER_TASK_MODULO:-5}"
ROUTER_TRAIN_REMAINDER="${ROUTER_TRAIN_REMAINDER:-0}"
ROUTER_TRAIN_REMAINDERS="${ROUTER_TRAIN_REMAINDERS:-}"
ROUTER_CALIBRATION_REMAINDER="${ROUTER_CALIBRATION_REMAINDER:-1}"
ROUTER_TARGET_PASS_RATE="${ROUTER_TARGET_PASS_RATE:-}"
SFT_LR_SCHEDULE="${SFT_LR_SCHEDULE:-${SFT_LR:-1e-4}}"
SFT_EPOCHS_SCHEDULE="${SFT_EPOCHS_SCHEDULE:-${SFT_TOTAL_EPOCHS:-3}}"
GRPO_LR_SCHEDULE="${GRPO_LR_SCHEDULE:-${LR:-1e-6}}"
DISTILLER_MODEL="${DISTILLER_MODEL:-}"
# CommonStack (OpenAI-compatible aggregator) -- matches router-skills-evolve's
# src/config.py. Only reached when DISTILLER_MODEL is set (e.g.
# openai/gpt-5.5); local small/large model calls never touch this.
DISTILLER_BASE_URL="${DISTILLER_BASE_URL:-https://api.commonstack.ai/v1}"
DISTILLER_API_KEY="${DISTILLER_API_KEY:-${COMMONSTACK_API_KEY:-}}"
export DISTILLER_API_KEY
ENABLE_SFT="${ENABLE_SFT:-0}"
RUN_POST_TRAIN_EVAL="${RUN_POST_TRAIN_EVAL:-1}"
RUN_PRETRAIN_EVAL="${RUN_PRETRAIN_EVAL:-0}"
SMALL_RELOAD_CMD="${SMALL_RELOAD_CMD:-}"
SMALL_STOP_CMD="${SMALL_STOP_CMD:-}"
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE:-0}"
COLLECT_CHUNK_TIMEOUT="${COLLECT_CHUNK_TIMEOUT:-180}"
COLLECT_CHUNK_RETRIES="${COLLECT_CHUNK_RETRIES:-3}"
MERGE_ADAPTER_FOR_SERVING="${MERGE_ADAPTER_FOR_SERVING:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
START_CYCLE="${START_CYCLE:-0}"

usage() {
  cat <<'EOF'
MERA evolve loop.

Options:
  --n-cycles N
  --limit N
  --mock
  --skip-train
  --probe-only
  --small-model MODEL
  --large-model MODEL
  --small-base-url URL
  --large-base-url URL
  --data PATH
  --experiment-name NAME
  --small-samples K
  --max-repair-turns N
  --skip-post-train-eval

Real runs expect small/large OpenAI-compatible vLLM servers to already be up.
Use SCALING_MOCK=1 or --mock for orchestration smoke tests with no servers.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-cycles) N_CYCLES="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --mock) MOCK=1; SKIP_TRAIN=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --probe-only) PROBE_ONLY=1; shift ;;
    --small-model) SMALL_MODEL="$2"; shift 2 ;;
    --large-model) LARGE_MODEL="$2"; shift 2 ;;
    --small-base-url) SMALL_BASE_URL="$2"; shift 2 ;;
    --large-base-url) LARGE_BASE_URL="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; RESULTS_DIR="results/$2"; shift 2 ;;
    --small-samples) SMALL_SAMPLES="$2"; shift 2 ;;
    --max-repair-turns) MAX_REPAIR_TURNS="$2"; shift 2 ;;
    --skip-post-train-eval) RUN_POST_TRAIN_EVAL=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[pipeline] unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RESULTS_DIR"

if [[ "$MOCK" == "1" ]]; then
  export SCALING_MOCK=1
fi

router_threshold_from() {
  local router_dir="$1"
  "$PYTHON" - "$router_dir/router_meta.json" <<'PY'
import json
import sys
try:
    print(json.load(open(sys.argv[1])).get("threshold_default", 0.5))
except Exception:
    print(0.5)
PY
}

schedule_value() {
  local schedule="$1" cycle="$2"
  "$PYTHON" - "$schedule" "$cycle" <<'PY'
import sys
values = [value.strip() for value in sys.argv[1].split(",") if value.strip()]
if not values:
    raise SystemExit("empty cycle schedule")
print(values[min(int(sys.argv[2]), len(values) - 1)])
PY
}

reload_small_server() {
  local model_path="$1"
  if [[ "$MERGE_ADAPTER_FOR_SERVING" == "1" && -f "$model_path/adapter_config.json" ]]; then
    local merged_path="${model_path}_merged"
    echo "[pipeline] merging adapter for serving: $model_path -> $merged_path"
    "$PYTHON" -m verl_code_rl.merge_lora --adapter "$model_path" --output "$merged_path"
    model_path="$merged_path"
  fi
  if [[ -z "$SMALL_RELOAD_CMD" ]]; then
    echo "[pipeline] ERROR: a new checkpoint was produced, but SMALL_RELOAD_CMD is unset." >&2
    echo "[pipeline] Set it to the command that restarts your small vLLM server with MODEL_PATH and waits for /v1/models." >&2
    return 2
  fi
  echo "[pipeline] reloading small vLLM server for $model_path"
  local port
  port="$($PYTHON - "$SMALL_BASE_URL" <<'PY'
from urllib.parse import urlparse
import sys
print(urlparse(sys.argv[1]).port or 8000)
PY
)"
  MODEL_PATH="$model_path" PORT="$port" SMALL_BASE_URL="$SMALL_BASE_URL" eval "$SMALL_RELOAD_CMD"
}

collect_traces() {
  local output="$1"
  shift
  local args=("$@")
  if (( COLLECT_CHUNK_SIZE <= 0 )); then
    "$PYTHON" "${args[@]}" --limit "$LIMIT" --out "$output"
    return
  fi

  local offset=0 part_count chunk_limit attempt part existing_count aligned_count
  if [[ "$RESUME_EXISTING" == "1" && -s "$output" ]]; then
    existing_count="$(wc -l < "$output")"
    aligned_count=$((existing_count / COLLECT_CHUNK_SIZE * COLLECT_CHUNK_SIZE))
    if (( aligned_count != existing_count )); then
      echo "[pipeline] discarding incomplete trailing chunk: $output ($existing_count -> $aligned_count rows)"
      head -n "$aligned_count" "$output" > "${output}.resume_tmp"
      mv "${output}.resume_tmp" "$output"
    fi
    offset="$aligned_count"
    echo "[pipeline] resuming collection at offset=$offset: $output"
  else
    : > "$output"
  fi
  rm -f "${output}.part_"*
  while (( LIMIT < 0 || offset < LIMIT )); do
    chunk_limit="$COLLECT_CHUNK_SIZE"
    if (( LIMIT >= 0 && offset + chunk_limit > LIMIT )); then
      chunk_limit=$((LIMIT - offset))
    fi
    part="${output}.part_${offset}"
    for ((attempt=1; attempt<=COLLECT_CHUNK_RETRIES; attempt++)); do
      echo "[pipeline] collect chunk offset=$offset limit=$chunk_limit attempt=$attempt"
      rm -f "$part"
      if timeout -k 10s "${COLLECT_CHUNK_TIMEOUT}s" \
          "$PYTHON" "${args[@]}" --task-offset "$offset" --limit "$chunk_limit" --out "$part"; then
        break
      fi
      if (( attempt == COLLECT_CHUNK_RETRIES )); then
        echo "[pipeline] collection chunk failed after $attempt attempts: $part" >&2
        return 1
      fi
      reload_small_server "$CURRENT_MODEL"
    done
    part_count="$(wc -l < "$part")"
    cat "$part" >> "$output"
    rm -f "$part"
    offset=$((offset + part_count))
    if (( part_count < chunk_limit )); then
      break
    fi
    if (( LIMIT < 0 || offset < LIMIT )); then
      reload_small_server "$CURRENT_MODEL"
    fi
  done
}

cat > "$RESULTS_DIR/config_snapshot.json" <<EOF
{
  "data": "$DATA",
  "small_model": "$SMALL_MODEL",
  "large_model": "$LARGE_MODEL",
  "small_base_url": "$SMALL_BASE_URL",
  "large_base_url": "$LARGE_BASE_URL",
  "n_cycles": $N_CYCLES,
  "start_cycle": $START_CYCLE,
  "limit": $LIMIT,
  "split": "$SPLIT",
  "dataset": "$DATASET",
  "small_samples": $SMALL_SAMPLES,
  "max_repair_turns": $MAX_REPAIR_TURNS,
  "training_seed": $TRAINING_SEED,
  "large_temperature": $LARGE_TEMPERATURE,
  "small_max_tokens": $SMALL_MAX_TOKENS,
  "large_max_tokens": $LARGE_MAX_TOKENS,
  "large_cache": "$LARGE_CACHE",
  "large_use_skills": false,
  "distiller_model": "$DISTILLER_MODEL",
  "sft_lr_schedule": "$SFT_LR_SCHEDULE",
  "sft_epochs_schedule": "$SFT_EPOCHS_SCHEDULE",
  "grpo_lr_schedule": "$GRPO_LR_SCHEDULE",
  "router_train_remainders": "${ROUTER_TRAIN_REMAINDERS:-$ROUTER_TRAIN_REMAINDER}",
  "router_calibration_remainder": $ROUTER_CALIBRATION_REMAINDER,
  "enable_sft": $ENABLE_SFT,
  "run_pretrain_eval": $RUN_PRETRAIN_EVAL,
  "skip_grpo": $SKIP_GRPO,
  "probe_only": $PROBE_ONLY,
  "skip_train": $SKIP_TRAIN,
  "mock": $MOCK
}
EOF

echo "[pipeline] results=$RESULTS_DIR cycles=$N_CYCLES mock=$MOCK skip_train=$SKIP_TRAIN"

CURRENT_MODEL="$SMALL_MODEL"
# verl's model.path must always be a loadable full HF checkpoint; once LoRA
# training starts, CURRENT_MODEL becomes an adapter-only dir (no base weights),
# so it can never itself be fed back in as model.path. BASE_SMALL_MODEL stays
# pinned to the original checkpoint across all cycles for that purpose.
BASE_SMALL_MODEL="$SMALL_MODEL"
if (( START_CYCLE < 0 || START_CYCLE >= N_CYCLES )); then
  echo "[pipeline] START_CYCLE must satisfy 0 <= START_CYCLE < N_CYCLES" >&2
  exit 2
fi
if (( START_CYCLE > 0 )); then
  previous_adapter_file="$RESULTS_DIR/cycle_$((START_CYCLE-1))/verl_checkpoints/final_adapter_path.txt"
  if [[ ! -s "$previous_adapter_file" ]]; then
    echo "[pipeline] cannot resume cycle $START_CYCLE: missing $previous_adapter_file" >&2
    exit 1
  fi
  CURRENT_MODEL="$(<"$previous_adapter_file")"
  if [[ ! -s "$CURRENT_MODEL/adapter_config.json" || ! -s "$CURRENT_MODEL/adapter_model.safetensors" ]]; then
    echo "[pipeline] cannot resume cycle $START_CYCLE: invalid adapter $CURRENT_MODEL" >&2
    exit 1
  fi
  echo "[pipeline] resuming at cycle $START_CYCLE with adapter $CURRENT_MODEL"
  reload_small_server "$CURRENT_MODEL"
fi
large_cache_args=(
  --large-temperature "$LARGE_TEMPERATURE"
  --large-max-tokens "$LARGE_MAX_TOKENS"
)
if [[ -n "$LARGE_CACHE" ]]; then
  large_cache_args+=(--large-cache "$LARGE_CACHE" --require-large-cache)
fi
export LARGE_USE_SKILLS=0
if [[ "$RUN_PRETRAIN_EVAL" == "1" && "$START_CYCLE" == "0" ]]; then
  PRETRAIN_DIR="$RESULTS_DIR/pretrain_baseline"
  mkdir -p "$PRETRAIN_DIR"
  echo
  echo "== Pre-training held-out evaluation =="
  pretrain_eval_args=(
    -m verl_code_rl.collect_traces
    --data "$DATA" --split eval --dataset "$DATASET" --cycle -1
    --small-model "$SMALL_SERVED_MODEL" --large-model "$LARGE_MODEL"
    --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL"
    --workers "$WORKERS" --small-samples "$SMALL_SAMPLES"
    --max-repair-turns "$MAX_REPAIR_TURNS" --max-tokens "$SMALL_MAX_TOKENS"
    --probe-only "${large_cache_args[@]}"
  )
  collect_traces "$PRETRAIN_DIR/eval_traces.jsonl" "${pretrain_eval_args[@]}"
  "$PYTHON" - "$PRETRAIN_DIR/eval_traces.jsonl" "$PRETRAIN_DIR/summary.json" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
n = len(rows)
summary = {
    "n_eval": n,
    "small_direct_pass": sum(bool(row.get("small_success")) for row in rows) / n,
    "large_only_pass": sum(bool(row.get("large_success")) for row in rows) / n,
    "verifier_cascade_pass": sum(bool(row.get("final_success")) for row in rows) / n,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n")
print(f"[pretrain-eval] {summary}")
PY
fi
for ((cycle=START_CYCLE; cycle<N_CYCLES; cycle++)); do
  OUT="$RESULTS_DIR/cycle_$cycle"
  mkdir -p "$OUT"
  CYCLE_SFT_LR="$(schedule_value "$SFT_LR_SCHEDULE" "$cycle")"
  CYCLE_SFT_EPOCHS="$(schedule_value "$SFT_EPOCHS_SCHEDULE" "$cycle")"
  CYCLE_GRPO_LR="$(schedule_value "$GRPO_LR_SCHEDULE" "$cycle")"
  echo
  echo "== Cycle $cycle: sft_lr=$CYCLE_SFT_LR sft_epochs=$CYCLE_SFT_EPOCHS grpo_lr=$CYCLE_GRPO_LR =="

  PREV_SKILLBOOK=""
  PREV_ROUTER=""
  if (( cycle > 0 )); then
    [[ -f "$RESULTS_DIR/cycle_$((cycle-1))/skillbook/skill_statistics.json" ]] && PREV_SKILLBOOK="$RESULTS_DIR/cycle_$((cycle-1))/skillbook"
    [[ -f "$RESULTS_DIR/cycle_$((cycle-1))/router/router.joblib" ]] && PREV_ROUTER="$RESULTS_DIR/cycle_$((cycle-1))/router"
  fi

  ROUTER_THRESHOLD="0.5"
  [[ -n "$PREV_ROUTER" ]] && ROUTER_THRESHOLD="$(router_threshold_from "$PREV_ROUTER")"

  PREPARED=0
  if [[ "$RESUME_EXISTING" == "1" \
        && -s "$OUT/traces.jsonl" \
        && -s "$OUT/skillbook/skill_statistics.json" \
        && -s "$OUT/sft_pairs.parquet" \
        && -s "$OUT/processed/train.parquet" \
        && -s "$OUT/processed/val.parquet" ]]; then
    PREPARED=1
    echo "[pipeline] reusing completed collection/preparation for cycle $cycle"
  fi

  collect_args=(
    -m verl_code_rl.collect_traces
    --data "$DATA"
    --split "$SPLIT"
    --dataset "$DATASET"
    --cycle "$cycle"
    --small-model "$SMALL_SERVED_MODEL"
    --large-model "$LARGE_MODEL"
    --small-base-url "$SMALL_BASE_URL"
    --large-base-url "$LARGE_BASE_URL"
    --workers "$WORKERS"
    --small-samples "$SMALL_SAMPLES"
    --max-repair-turns "$MAX_REPAIR_TURNS"
    --max-tokens "$SMALL_MAX_TOKENS"
    "${large_cache_args[@]}"
    --router-threshold "$ROUTER_THRESHOLD"
  )
  [[ -n "$PREV_SKILLBOOK" ]] && collect_args+=(--skillbook "$PREV_SKILLBOOK")
  [[ -n "$PREV_ROUTER" ]] && collect_args+=(--router "$PREV_ROUTER")
  [[ "$PROBE_ONLY" == "1" ]] && collect_args+=(--probe-only)
  if [[ "$PREPARED" == "0" ]]; then
    collect_traces "$OUT/traces.jsonl" "${collect_args[@]}"

    build_args=(
      -m verl_code_rl.build_skillbook
      --traces "$OUT/traces.jsonl"
      --output "$OUT/skillbook"
      --small-model "$CURRENT_MODEL"
      --large-model "$LARGE_MODEL"
    )
    [[ -n "$PREV_SKILLBOOK" ]] && build_args+=(--previous "$PREV_SKILLBOOK")
    [[ -n "$DISTILLER_MODEL" ]] && build_args+=(--distiller-model "$DISTILLER_MODEL" --distiller-base-url "$DISTILLER_BASE_URL")
    "$PYTHON" "${build_args[@]}"

    "$PYTHON" -m verl_code_rl.trace_diagnostics \
      --traces "$OUT/traces.jsonl" \
      --output "$OUT/trace_diagnostics.json"

    "$PYTHON" -m verl_code_rl.traces_to_sft \
      --traces "$OUT/traces.jsonl" \
      --skillbook "$OUT/skillbook" \
      --output "$OUT/sft_pairs.parquet"

    "$PYTHON" -m verl_code_rl.prepare_data \
      --traces "$OUT/traces.jsonl" \
      --input "$DATA" \
      --skillbook "$OUT/skillbook" \
      --out-dir "$OUT/processed"
  fi

  UPDATED_MODEL=0
  if [[ "$SKIP_TRAIN" != "1" ]]; then
    if [[ -n "$SMALL_STOP_CMD" && ( "$ENABLE_SFT" == "1" || "$SKIP_GRPO" != "1" ) ]]; then
      echo "[pipeline] stopping small server before training"
      eval "$SMALL_STOP_CMD"
    fi
    if [[ "$ENABLE_SFT" == "1" && -s "$OUT/sft_pairs.parquet" ]]; then
      # Same reasoning as the GRPO block below: verl's model.path must be a
      # loadable full checkpoint, so CURRENT_MODEL (which may already be a
      # bare adapter dir from a prior cycle) is never passed as MODEL_PATH --
      # it's forwarded as LORA_ADAPTER_PATH for continuation instead.
      PREV_SFT_LORA_ADAPTER=""
      [[ -f "$CURRENT_MODEL/adapter_config.json" ]] && PREV_SFT_LORA_ADAPTER="$CURRENT_MODEL"

      SFT_DATA="$OUT/sft_pairs.parquet" \
      SFT_OUTPUT_DIR="$OUT/sft_adapter" \
      MODEL_PATH="$BASE_SMALL_MODEL" \
      LORA_ADAPTER_PATH="$PREV_SFT_LORA_ADAPTER" \
      SFT_LR="$CYCLE_SFT_LR" \
      SFT_TOTAL_EPOCHS="$CYCLE_SFT_EPOCHS" \
      TRAINING_SEED="$TRAINING_SEED" \
        scripts/train_sft.sh
      if [[ -d "$OUT/sft_adapter" ]]; then
        CURRENT_MODEL="$OUT/sft_adapter"
        UPDATED_MODEL=1
      fi
    fi

    if [[ "$SKIP_GRPO" != "1" ]]; then
      # If CURRENT_MODEL is already a LoRA adapter (a prior cycle's output),
      # continue training it via verl's lora_adapter_path, loaded on top of the
      # pinned base -- do NOT pass the adapter dir itself as MODEL_PATH.
      PREV_LORA_ADAPTER=""
      [[ -f "$CURRENT_MODEL/adapter_config.json" ]] && PREV_LORA_ADAPTER="$CURRENT_MODEL"

      TRAIN_FILE="$OUT/processed/train.parquet" \
      VAL_FILE="$OUT/processed/val.parquet" \
      MODEL_PATH="$BASE_SMALL_MODEL" \
      LORA_ADAPTER_PATH="$PREV_LORA_ADAPTER" \
      PROJECT_NAME="mera_evolve" \
      EXPERIMENT_NAME="${EXPERIMENT_NAME}_cycle_${cycle}" \
      OUTPUT_DIR="$OUT/verl_checkpoints" \
      LR="$CYCLE_GRPO_LR" \
      TRAINING_SEED="$TRAINING_SEED" \
        scripts/train_grpo.sh

      adapter_path_file="$OUT/verl_checkpoints/final_adapter_path.txt"
      if [[ -s "$adapter_path_file" ]]; then
        CURRENT_MODEL="$(<"$adapter_path_file")"
        if [[ ! -s "$CURRENT_MODEL/adapter_config.json" || ! -s "$CURRENT_MODEL/adapter_model.safetensors" ]]; then
          echo "[pipeline] invalid GRPO adapter recorded in $adapter_path_file: $CURRENT_MODEL" >&2
          exit 1
        fi
        UPDATED_MODEL=1
        echo "[pipeline] next cycle small model: $CURRENT_MODEL"
      else
        echo "[pipeline] GRPO completed without a loadable adapter: $OUT/verl_checkpoints" >&2
        exit 1
      fi
    fi

    if [[ "$RUN_POST_TRAIN_EVAL" == "1" && "$UPDATED_MODEL" == "1" ]]; then
      reload_small_server "$CURRENT_MODEL"
    fi
  fi

  # Train/calibrate the router on the *current* small model, using distinct
  # task-id shards.  These traces are never used for GRPO updates.
  router_trace_common=(
    -m verl_code_rl.collect_traces
    --data "$DATA" --split train --dataset "$DATASET" --cycle "$cycle"
    --small-model "$SMALL_SERVED_MODEL" --large-model "$LARGE_MODEL"
    --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL"
    --workers "$WORKERS" --small-samples "$SMALL_SAMPLES" --max-repair-turns "$MAX_REPAIR_TURNS"
    --max-tokens "$SMALL_MAX_TOKENS"
    --task-modulo "$ROUTER_TASK_MODULO" --skillbook "$OUT/skillbook" --probe-only
    "${large_cache_args[@]}"
  )
  router_train_shard_args=(--task-remainder "$ROUTER_TRAIN_REMAINDER")
  if [[ -n "$ROUTER_TRAIN_REMAINDERS" ]]; then
    router_train_shard_args=(--task-remainders "$ROUTER_TRAIN_REMAINDERS")
  fi
  collect_traces "$OUT/router_train_traces.jsonl" \
    "${router_trace_common[@]}" "${router_train_shard_args[@]}"
  collect_traces "$OUT/router_calibration_traces.jsonl" \
    "${router_trace_common[@]}" --task-remainder "$ROUTER_CALIBRATION_REMAINDER"

  router_train_args=(
    -m verl_code_rl.train_router
    --traces "$OUT/router_train_traces.jsonl"
    --calibration-traces "$OUT/router_calibration_traces.jsonl"
    --output-dir "$OUT/router"
  )
  [[ -n "$ROUTER_TARGET_PASS_RATE" ]] && router_train_args+=(--target-pass-rate "$ROUTER_TARGET_PASS_RATE")
  "$PYTHON" "${router_train_args[@]}"
  ROUTER_THRESHOLD="$(router_threshold_from "$OUT/router")"

  # This is the only ablation source: held-out eval tasks, evaluated with the
  # freshly trained/checkpointed small model and the calibrated router.
  eval_collect_args=(
    -m verl_code_rl.collect_traces
    --data "$DATA" --split eval --dataset "$DATASET" --cycle "$cycle"
    --small-model "$SMALL_SERVED_MODEL" --large-model "$LARGE_MODEL" \
    --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL" \
    --workers "$WORKERS" --small-samples "$SMALL_SAMPLES" --max-repair-turns "$MAX_REPAIR_TURNS" \
    --max-tokens "$SMALL_MAX_TOKENS" \
    --router "$OUT/router" --router-threshold "$ROUTER_THRESHOLD" --skillbook "$OUT/skillbook" \
    --probe-only "${large_cache_args[@]}"
  )
  collect_traces "$OUT/eval_traces.jsonl" "${eval_collect_args[@]}"

  "$PYTHON" -m verl_code_rl.run_ablation \
    --traces "$OUT/eval_traces.jsonl" \
    --router-dir "$OUT/router" --router-threshold "$ROUTER_THRESHOLD" \
    --output "$OUT/e2e_ablation_summary.json" \
    --markdown-output "$OUT/e2e_ablation_summary.md"
done

echo
echo "[pipeline] done: $RESULTS_DIR"
