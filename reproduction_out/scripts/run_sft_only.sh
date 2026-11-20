#!/usr/bin/env bash
# SFT-only run on the student's own successful E1 trajectories (research_plan.md M step 4,
# "SFT ... rehearsal of the student's own successes", without the GRPO half).
#
# Every training detail is copied from the repository's own SFT stage in
# experiments/tau-2/scripts/run_cycle_pipeline.sh (lines ~465-490): same dataset class,
# same Qwen3.5 compat env, same tau2 defaults (lr 1e-7, 1 epoch, no thinking). No training
# logic is re-implemented here; only the scheduling gate is new.
#
# Order: E2 arm 1 training -> E2 evalckpt (uses 4 GPUs) -> this SFT (uses 2 GPUs).
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
SERIAL=$D/logs/serial_progress.log
R=$D/results/sft_only
LOG=$D/logs/sft_only.log
TRAIN_VENV=/home/powerop/work/zsc/newproj/.venv
MERA_PY="$TRAIN_VENV/bin/python3"
TRITON_OVERLAY=/home/powerop/work/yjh/MERA-Evolve/.deps/qwen35-triton33
mkdir -p "$R" "$D/logs"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[[ -s "$R/sft_pairs.parquet" ]] || { log "missing $R/sft_pairs.parquet"; exit 1; }
log "armed: waiting for E2 arm 1 training AND evalckpt to finish"

# 1) training done
for _ in $(seq 1 5760); do
  [[ -s "$D/results/e2_arm1_lr1e-5/train/group_O/verl/final_adapter_path.txt" ]] && break
  sleep 20
done
[[ -s "$D/results/e2_arm1_lr1e-5/train/group_O/verl/final_adapter_path.txt" ]] || { log "timeout waiting for training"; exit 1; }
log "E2 training finished; waiting for evalckpt to finish (it owns all 4 GPUs)"

# 2) evalckpt done (any rc) and no GPU-heavy processes left
for _ in $(seq 1 2880); do
  if grep -q 'E2_ARM1_EVAL EXIT' "$SERIAL" 2>/dev/null; then break; fi
  sleep 20
done
for _ in $(seq 1 120); do
  n_proc=$(pgrep -f 'one_step_off[_]policy|03_grpo|verl[ ]train' | wc -l)
  [[ "$n_proc" == 0 ]] && break
  sleep 10
done
sleep 20
log "GPUs before SFT:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while read -r q; do log "  $q"; done

# 3) SFT with the repository's own trainer and tau2 defaults
# 2026-11-19: 4/816 rows render to >24576 tokens (25.9k-33.0k, all AREAL airline) and the
# trainer raises ValueError instead of truncating. The plan keeps student thinking off and a
# 24k budget, so the filtered 812-row parquet is the input; provenance is in
# sft_pairs_24k.filter.json. Override with SFT_DATA_FILE for another file.
SFT_DATA_FILE="${SFT_DATA_FILE:-$R/sft_pairs_24k.parquet}"
[[ -s "$SFT_DATA_FILE" ]] || SFT_DATA_FILE="$R/sft_pairs.parquet"
log "starting SFT: $SFT_DATA_FILE, lr 1e-7, 1 epoch, max_length 24576, 1 GPU (CUDA_VISIBLE_DEVICES=1,2, nproc=1)"
start=$(date +%s)
( cd "$REPO" && source "$TRAIN_VENV/bin/activate" && \
  env CUDA_VISIBLE_DEVICES=1,2 \
  PYTHON="$MERA_PY" \
  SFT_DATA="$SFT_DATA_FILE" \
  SFT_OUTPUT_DIR="$R/sft_adapter" \
  MODEL_PATH="$MODELS_DIR/Qwen3.5-4B" \
  SFT_BATCH_SIZE=2 \
  SFT_MICRO_BATCH_SIZE_PER_GPU=1 \
  SFT_LR=1e-7 \
  SFT_TOTAL_EPOCHS=1 \
  SFT_MAX_LENGTH=24576 \
  QWEN35_ENABLE_VERL_PATCHES=1 \
  PYTHONPATH="$TRITON_OVERLAY:$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat:${PYTHONPATH:-}" \
  LORA_TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  FLA_TILELANG=0 \
  SFT_ENABLE_THINKING=False \
  SFT_ATTN_IMPLEMENTATION=sdpa \
  SFT_USE_TORCH_COMPILE=False \
  SFT_DATASET_PATH="$REPO/experiments/tau-2/tau2_evolve/sft_dataset.py" \
  SFT_DATASET_NAME=Qwen35MultiTurnSFTDataset \
  SFT_MAX_TOKEN_LEN_PER_GPU=24576 \
  SFT_SAVE_FREQ=3 \
  SFT_PROJECT_NAME=tau2_sft_only \
  SFT_EXPERIMENT_NAME="e1_successes_$(date -u +%Y%m%d)" \
  bash scripts/train_sft.sh data.num_workers=0 ) >> "$LOG" 2>&1
rc=$?
end=$(date +%s)
log "SFT_ONLY_EXIT rc=$rc elapsed=$((end-start))s adapter=$R/sft_adapter"
echo "SFT_ONLY EXIT rc=$rc elapsed=$((end-start))s" >> "$SERIAL"
exit "$rc"
