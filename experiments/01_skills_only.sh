#!/usr/bin/env bash
set -euo pipefail

# Isolates the SkillBook's own effect: no training of any kind. A single
# probe-only oracle collection builds the skillbook, then the SAME untouched
# small model checkpoint is evaluated twice on the same held-out split --
# once bare, once with the distilled skill text prepended to every prompt.
#
# Prerequisites: small + large OpenAI-compatible vLLM servers already running
# at SMALL_BASE_URL / LARGE_BASE_URL (see scripts/serve_vllm.sh).
#
# Usage: bash experiments/01_skills_only.sh

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA="${DATA:-data/raw/he_mbpp.jsonl}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SMALL_BASE_URL="${SMALL_BASE_URL:-http://127.0.0.1:8000/v1}"
LARGE_BASE_URL="${LARGE_BASE_URL:-http://127.0.0.1:8001/v1}"
API_KEY="${API_KEY:-EMPTY}"
TRAIN_LIMIT="${TRAIN_LIMIT:--1}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"
WORKERS="${WORKERS:-32}"
DISTILLER_MODEL="${DISTILLER_MODEL:-}"
DISTILLER_BASE_URL="${DISTILLER_BASE_URL:-https://api.commonstack.ai/v1}"
DISTILLER_API_KEY="${DISTILLER_API_KEY:-${COMMONSTACK_API_KEY:-}}"
RESULTS_DIR="${RESULTS_DIR:-results/exp_skills_only_$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$RESULTS_DIR"
echo "[skills-only] results dir: $RESULTS_DIR"

echo
echo "[skills-only] step 1: collect oracle traces (small-first, fallback on failure, no router)"
"$PYTHON" -m verl_code_rl.collect_traces \
  --data "$DATA" --split train --limit "$TRAIN_LIMIT" \
  --small-model "$SMALL_MODEL" --large-model "$LARGE_MODEL" \
  --small-base-url "$SMALL_BASE_URL" --large-base-url "$LARGE_BASE_URL" --api-key "$API_KEY" \
  --workers "$WORKERS" --probe-only \
  --out "$RESULTS_DIR/traces.jsonl"

echo
echo "[skills-only] step 2: build skillbook from those traces only (no training)"
build_args=(
  -m verl_code_rl.build_skillbook
  --traces "$RESULTS_DIR/traces.jsonl"
  --output "$RESULTS_DIR/skillbook"
  --small-model "$SMALL_MODEL" --large-model "$LARGE_MODEL"
)
[[ -n "$DISTILLER_MODEL" ]] && build_args+=(--distiller-model "$DISTILLER_MODEL" --distiller-base-url "$DISTILLER_BASE_URL" --api-key "$DISTILLER_API_KEY")
"$PYTHON" "${build_args[@]}"

echo
echo "[skills-only] step 3: baseline eval (untouched small model, no skill)"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "$SMALL_BASE_URL" --api-key "$API_KEY" \
  --model "$SMALL_MODEL" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --out "$RESULTS_DIR/baseline_eval.jsonl"

echo
echo "[skills-only] step 4: skills eval (SAME checkpoint, skill text prepended)"
"$PYTHON" -m verl_code_rl.eval_vllm \
  --data "$DATA" --base-url "$SMALL_BASE_URL" --api-key "$API_KEY" \
  --model "$SMALL_MODEL" --split eval --limit "$EVAL_LIMIT" --workers "$WORKERS" \
  --skillbook "$RESULTS_DIR/skillbook" \
  --out "$RESULTS_DIR/skills_eval.jsonl"

echo
echo "[skills-only] done. Compare pass@1 printed above for baseline_eval vs skills_eval."
echo "[skills-only] results in $RESULTS_DIR"
