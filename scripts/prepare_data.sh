#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
INPUT="${INPUT:-data/raw/he_mbpp.jsonl}"
OUT_DIR="${OUT_DIR:-data/processed}"

"$PYTHON" -m verl_code_rl.prepare_data \
  --input "$INPUT" \
  --out-dir "$OUT_DIR" \
  "${@}"
