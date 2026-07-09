#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA="${DATA:-data/raw/he_mbpp.jsonl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mera_evolve_$(date -u +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-results/$EXPERIMENT_NAME}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SMALL_BASE_URL="${SMALL_BASE_URL:-http://127.0.0.1:8000/v1}"
LARGE_BASE_URL="${LARGE_BASE_URL:-http://127.0.0.1:8001/v1}"
API_KEY="${API_KEY:-EMPTY}"
N_CYCLES="${N_CYCLES:-1}"
LIMIT="${LIMIT:--1}"
WORKERS="${WORKERS:-16}"
SPLIT="${SPLIT:-train}"
DATASET="${DATASET:-all}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
MOCK="${MOCK:-0}"
PROBE_ONLY="${PROBE_ONLY:-0}"
SMALL_SAMPLES="${SMALL_SAMPLES:-1}"
MAX_REPAIR_TURNS="${MAX_REPAIR_TURNS:-0}"
ROUTER_TASK_MODULO="${ROUTER_TASK_MODULO:-5}"
ROUTER_TRAIN_REMAINDER="${ROUTER_TRAIN_REMAINDER:-0}"
ROUTER_CALIBRATION_REMAINDER="${ROUTER_CALIBRATION_REMAINDER:-1}"
ROUTER_TARGET_PASS_RATE="${ROUTER_TARGET_PASS_RATE:-}"
DISTILLER_MODEL="${DISTILLER_MODEL:-}"
ENABLE_SFT="${ENABLE_SFT:-0}"
RUN_POST_TRAIN_EVAL="${RUN_POST_TRAIN_EVAL:-1}"
SMALL_RELOAD_CMD="${SMALL_RELOAD_CMD:-}"

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

reload_small_server() {
  local model_path="$1"
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

cat > "$RESULTS_DIR/config_snapshot.json" <<EOF
{
  "data": "$DATA",
  "small_model": "$SMALL_MODEL",
  "large_model": "$LARGE_MODEL",
  "small_base_url": "$SMALL_BASE_URL",
  "large_base_url": "$LARGE_BASE_URL",
  "n_cycles": $N_CYCLES,
  "limit": $LIMIT,
  "split": "$SPLIT",
  "dataset": "$DATASET",
  "small_samples": $SMALL_SAMPLES,
  "max_repair_turns": $MAX_REPAIR_TURNS,
  "skip_train": $SKIP_TRAIN,
  "mock": $MOCK
}
EOF

echo "[pipeline] results=$RESULTS_DIR cycles=$N_CYCLES mock=$MOCK skip_train=$SKIP_TRAIN"

CURRENT_MODEL="$SMALL_MODEL"
for ((cycle=0; cycle<N_CYCLES; cycle++)); do
  OUT="$RESULTS_DIR/cycle_$cycle"
  mkdir -p "$OUT"
  echo
  echo "== Cycle $cycle =="

  PREV_SKILLBOOK=""
  PREV_ROUTER=""
  if (( cycle > 0 )); then
    [[ -f "$RESULTS_DIR/cycle_$((cycle-1))/skillbook.json" ]] && PREV_SKILLBOOK="$RESULTS_DIR/cycle_$((cycle-1))/skillbook.json"
    [[ -f "$RESULTS_DIR/cycle_$((cycle-1))/router/router.joblib" ]] && PREV_ROUTER="$RESULTS_DIR/cycle_$((cycle-1))/router"
  fi

  ROUTER_THRESHOLD="0.5"
  [[ -n "$PREV_ROUTER" ]] && ROUTER_THRESHOLD="$(router_threshold_from "$PREV_ROUTER")"

  collect_args=(
    -m verl_code_rl.collect_traces
    --data "$DATA"
    --split "$SPLIT"
    --dataset "$DATASET"
    --limit "$LIMIT"
    --cycle "$cycle"
    --small-model "$CURRENT_MODEL"
    --large-model "$LARGE_MODEL"
    --small-base-url "$SMALL_BASE_URL"
    --large-base-url "$LARGE_BASE_URL"
    --api-key "$API_KEY"
    --workers "$WORKERS"
    --small-samples "$SMALL_SAMPLES"
    --max-repair-turns "$MAX_REPAIR_TURNS"
    --router-threshold "$ROUTER_THRESHOLD"
    --out "$OUT/traces.jsonl"
  )
  [[ -n "$PREV_SKILLBOOK" ]] && collect_args+=(--skillbook "$PREV_SKILLBOOK")
  [[ -n "$PREV_ROUTER" ]] && collect_args+=(--router "$PREV_ROUTER")
  [[ "$PROBE_ONLY" == "1" ]] && collect_args+=(--probe-only)
  "$PYTHON" "${collect_args[@]}"

  build_args=(
    -m verl_code_rl.build_skillbook
    --traces "$OUT/traces.jsonl"
    --output "$OUT/skillbook.json"
    --small-model "$CURRENT_MODEL"
    --large-model "$LARGE_MODEL"
  )
  [[ -n "$PREV_SKILLBOOK" ]] && build_args+=(--previous "$PREV_SKILLBOOK")
  [[ -n "$DISTILLER_MODEL" ]] && build_args+=(--distiller-model "$DISTILLER_MODEL" --distiller-base-url "$LARGE_BASE_URL" --api-key "$API_KEY")
  "$PYTHON" "${build_args[@]}"

  "$PYTHON" -m verl_code_rl.trace_diagnostics \
    --traces "$OUT/traces.jsonl" \
    --output "$OUT/trace_diagnostics.json"

  "$PYTHON" -m verl_code_rl.traces_to_sft \
    --traces "$OUT/traces.jsonl" \
    --skillbook "$OUT/skillbook.json" \
    --output "$OUT/sft_pairs.jsonl"

  "$PYTHON" -m verl_code_rl.prepare_data \
    --traces "$OUT/traces.jsonl" \
    --input "$DATA" \
    --skillbook "$OUT/skillbook.json" \
    --out-dir "$OUT/processed"

  UPDATED_MODEL=0
  if [[ "$SKIP_TRAIN" != "1" ]]; then
    if [[ "$ENABLE_SFT" == "1" && -s "$OUT/sft_pairs.jsonl" ]]; then
      SFT_DATA="$OUT/sft_pairs.jsonl" \
      SFT_OUTPUT_DIR="$OUT/sft_adapter" \
      MODEL_PATH="$CURRENT_MODEL" \
        scripts/train_sft.sh
      if [[ -d "$OUT/sft_adapter" ]]; then
        CURRENT_MODEL="$OUT/sft_adapter"
        UPDATED_MODEL=1
      fi
    fi

    TRAIN_FILE="$OUT/processed/train.parquet" \
    VAL_FILE="$OUT/processed/val.parquet" \
    MODEL_PATH="$CURRENT_MODEL" \
    PROJECT_NAME="mera_evolve" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_cycle_${cycle}" \
    OUTPUT_DIR="$OUT/verl_checkpoints" \
      scripts/train_grpo.sh

    latest="$(find "$OUT/verl_checkpoints" -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "$latest" ]]; then
      if [[ -d "$latest/actor" ]]; then
        CURRENT_MODEL="$latest/actor"
      else
        CURRENT_MODEL="$latest"
      fi
      UPDATED_MODEL=1
      echo "[pipeline] next cycle small model: $CURRENT_MODEL"
    fi

    if [[ "$RUN_POST_TRAIN_EVAL" == "1" && "$UPDATED_MODEL" == "1" ]]; then
      reload_small_server "$CURRENT_MODEL"
    fi
  fi

  # Train/calibrate the router on the *current* small model, using distinct
  # task-id shards.  These traces are never used for GRPO updates.
  router_trace_common=(
    -m verl_code_rl.collect_traces
    --data "$DATA" --split train --dataset "$DATASET" --limit "$LIMIT" --cycle "$cycle"
    --small-model "$CURRENT_MODEL" --large-model "$LARGE_MODEL"
    --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL" --api-key "$API_KEY"
    --workers "$WORKERS" --small-samples "$SMALL_SAMPLES" --max-repair-turns "$MAX_REPAIR_TURNS"
    --task-modulo "$ROUTER_TASK_MODULO" --skillbook "$OUT/skillbook.json"
  )
  "$PYTHON" "${router_trace_common[@]}" --task-remainder "$ROUTER_TRAIN_REMAINDER" \
    --out "$OUT/router_train_traces.jsonl"
  "$PYTHON" "${router_trace_common[@]}" --task-remainder "$ROUTER_CALIBRATION_REMAINDER" \
    --out "$OUT/router_calibration_traces.jsonl"

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
  "$PYTHON" -m verl_code_rl.collect_traces \
    --data "$DATA" --split eval --dataset "$DATASET" --limit "$LIMIT" --cycle "$cycle" \
    --small-model "$CURRENT_MODEL" --large-model "$LARGE_MODEL" \
    --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL" --api-key "$API_KEY" \
    --workers "$WORKERS" --small-samples "$SMALL_SAMPLES" --max-repair-turns "$MAX_REPAIR_TURNS" \
    --router "$OUT/router" --router-threshold "$ROUTER_THRESHOLD" --skillbook "$OUT/skillbook.json" \
    --out "$OUT/eval_traces.jsonl"

  "$PYTHON" -m verl_code_rl.run_ablation \
    --traces "$OUT/eval_traces.jsonl" \
    --router-dir "$OUT/router" --router-threshold "$ROUTER_THRESHOLD" \
    --output "$OUT/e2e_ablation_summary.json" \
    --markdown-output "$OUT/e2e_ablation_summary.md"
done

echo
echo "[pipeline] done: $RESULTS_DIR"
