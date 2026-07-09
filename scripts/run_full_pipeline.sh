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
    -h|--help) usage; exit 0 ;;
    *) echo "[pipeline] unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RESULTS_DIR"

if [[ "$MOCK" == "1" ]]; then
  export SCALING_MOCK=1
fi

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
  "$PYTHON" "${build_args[@]}"

  "$PYTHON" -m verl_code_rl.prepare_data \
    --traces "$OUT/traces.jsonl" \
    --input "$DATA" \
    --skillbook "$OUT/skillbook.json" \
    --out-dir "$OUT/processed"

  "$PYTHON" -m verl_code_rl.train_router \
    --traces "$OUT/traces.jsonl" \
    --output-dir "$OUT/router"

  "$PYTHON" -m verl_code_rl.run_ablation \
    --traces "$OUT/traces.jsonl" \
    --router-dir "$OUT/router" \
    --output "$OUT/e2e_ablation_summary.json" \
    --markdown-output "$OUT/e2e_ablation_summary.md"

  if [[ "$SKIP_TRAIN" != "1" ]]; then
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
      echo "[pipeline] next cycle small model: $CURRENT_MODEL"
    fi
  fi
done

echo
echo "[pipeline] done: $RESULTS_DIR"
