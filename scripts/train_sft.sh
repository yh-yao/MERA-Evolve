#!/usr/bin/env bash
set -euo pipefail

# SFT launchers vary across verl installations.  Keep the artifact contract
# stable and require the cluster-specific launcher to be explicit rather than
# silently invoking an incompatible trainer.
: "${SFT_DATA:?set SFT_DATA to cycle_N/sft_pairs.jsonl}"
: "${SFT_OUTPUT_DIR:?set SFT_OUTPUT_DIR}"

if [[ -z "${SFT_TRAIN_CMD:-}" ]]; then
  echo "[train_sft] SFT_DATA prepared at $SFT_DATA." >&2
  echo "[train_sft] Set SFT_TRAIN_CMD to your version-pinned SFT launcher; skipping." >&2
  exit 0
fi

export SFT_DATA SFT_OUTPUT_DIR MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
eval "$SFT_TRAIN_CMD"
