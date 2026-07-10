#!/usr/bin/env bash
set -euo pipefail

# Isolates SkillBook-only improvement from GRPO-training-only improvement,
# both starting from the same initial oracle traces (small model first,
# fallback to large only on failure, no router).
#
#   1. collect initial_traces.jsonl: small-first, fallback-on-failure, no router.
#   2. baseline: eval the untouched small model, no skill.
#   3. skills arm:  build a skillbook from initial_traces.jsonl only (no training),
#                   eval the SAME small model checkpoint WITH the skill prefix.
#   4. train arm:   prepare GRPO data from the SAME initial_traces.jsonl (no skill
#                   injection), GRPO-train (LoRA), eval the TRAINED checkpoint
#                   with no skill.
#   5. print baseline vs skills vs trained pass rates on the same held-out eval split.

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA="${DATA:-data/raw/he_mbpp.jsonl}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SMALL_BASE_URL="${SMALL_BASE_URL:-http://127.0.0.1:8012/v1}"
LARGE_BASE_URL="${LARGE_BASE_URL:-http://127.0.0.1:8013/v1}"
API_KEY="${API_KEY:-EMPTY}"
TRAIN_LIMIT="${TRAIN_LIMIT:-64}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
WORKERS="${WORKERS:-16}"
TRAIN_GPU="${TRAIN_GPU:-5}"
TRAIN_PORT="${TRAIN_PORT:-8015}"
TRAINED_SERVED_NAME="${TRAINED_SERVED_NAME:-trained-adapter}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
RESULTS_DIR="${RESULTS_DIR:-results/skill_vs_train_$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$RESULTS_DIR"
echo "[exp] results dir: $RESULTS_DIR"

echo
echo "[exp] step 1: collect initial traces (small-first, fallback on failure, no router)"
"$PYTHON" -m verl_code_rl.collect_traces \
  --data "$DATA" --split train --limit "$TRAIN_LIMIT" \
  --small-model "$SMALL_MODEL" --large-model "$LARGE_MODEL" \
  --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL" --api-key "$API_KEY" \
  --workers "$WORKERS" --probe-only \
  --out "$RESULTS_DIR/initial_traces.jsonl"

echo
echo "[exp] step 2: baseline eval (untouched small model, no skill)"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "$SMALL_BASE_URL" --api-key "$API_KEY" \
  --model "$SMALL_MODEL" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --out "$RESULTS_DIR/baseline_eval.jsonl"

echo
echo "[exp] step 3a: build skillbook from initial traces only (no training)"
"$PYTHON" -m verl_code_rl.build_skillbook \
  --traces "$RESULTS_DIR/initial_traces.jsonl" \
  --output "$RESULTS_DIR/skillbook" \
  --small-model "$SMALL_MODEL" --large-model "$LARGE_MODEL"

echo
echo "[exp] step 3b: eval SAME small model checkpoint WITH skill prefix"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "$SMALL_BASE_URL" --api-key "$API_KEY" \
  --model "$SMALL_MODEL" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --skillbook "$RESULTS_DIR/skillbook" \
  --out "$RESULTS_DIR/skills_eval.jsonl"

echo
echo "[exp] step 4a: prepare GRPO data from the SAME initial traces (no skill injection)"
"$PYTHON" -m verl_code_rl.prepare_data \
  --traces "$RESULTS_DIR/initial_traces.jsonl" \
  --input "$DATA" \
  --out-dir "$RESULTS_DIR/processed"

echo
echo "[exp] step 4b: GRPO training (LoRA, default) on GPU $TRAIN_GPU"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" VLLM_USE_FLASHINFER_SAMPLER=0 \
TRAIN_FILE="$RESULTS_DIR/processed/train.parquet" \
VAL_FILE="$RESULTS_DIR/processed/val.parquet" \
MODEL_PATH="$SMALL_MODEL" \
TRAIN_BATCH_SIZE="$TRAIN_LIMIT" \
PPO_MINI_BATCH_SIZE=16 \
TOTAL_EPOCHS="$TOTAL_EPOCHS" \
TEST_FREQ=100 \
SAVE_FREQ=1 \
PROJECT_NAME="skill_vs_train_exp" \
EXPERIMENT_NAME="exp_$(basename "$RESULTS_DIR")" \
OUTPUT_DIR="$RESULTS_DIR/verl_checkpoints" \
  scripts/train_grpo.sh

latest="$(find "$RESULTS_DIR/verl_checkpoints" -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1)"
if [[ -z "$latest" ]]; then
  echo "[exp] ERROR: no checkpoint produced by training" >&2
  exit 1
fi
if [[ -f "$latest/actor/lora_adapter/adapter_config.json" ]]; then
  TRAINED_MODEL="$latest/actor/lora_adapter"
else
  TRAINED_MODEL="$latest/actor"
fi
echo "[exp] trained model: $TRAINED_MODEL"

echo
echo "[exp] step 4c: serve trained checkpoint on GPU $TRAIN_GPU port $TRAIN_PORT"
VLLM_USE_FLASHINFER_SAMPLER=0 MODEL_PATH="$TRAINED_MODEL" PORT="$TRAIN_PORT" GPU="$TRAIN_GPU" \
  SERVED_NAME="$TRAINED_SERVED_NAME" scripts/serve_vllm.sh

echo
echo "[exp] step 4d: eval TRAINED checkpoint (no skill)"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "http://127.0.0.1:$TRAIN_PORT/v1" --api-key "$API_KEY" \
  --model "$TRAINED_SERVED_NAME" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --out "$RESULTS_DIR/trained_eval.jsonl"

PORT="$TRAIN_PORT" scripts/stop_vllm.sh

echo
"$PYTHON" - "$RESULTS_DIR" <<'PY'
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])


def rate(path):
    rows = [json.loads(line) for line in path.open()]
    return (sum(r["passed"] for r in rows) / len(rows) if rows else 0.0), len(rows)


baseline, n0 = rate(d / "baseline_eval.jsonl")
skills, n1 = rate(d / "skills_eval.jsonl")
trained, n2 = rate(d / "trained_eval.jsonl")

summary = {
    "baseline_pass_rate": baseline, "n_baseline": n0,
    "skills_pass_rate": skills, "n_skills": n1, "skills_delta": skills - baseline,
    "trained_pass_rate": trained, "n_trained": n2, "trained_delta": trained - baseline,
}
print(json.dumps(summary, indent=2))
(d / "summary.json").write_text(json.dumps(summary, indent=2))
PY

echo
echo "[exp] done: $RESULTS_DIR"
