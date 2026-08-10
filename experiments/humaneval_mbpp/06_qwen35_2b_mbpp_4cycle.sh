#!/usr/bin/env bash
set -euo pipefail

# MBPP-only four-cycle run. GPU 2 serves the 2B endpoint between training
# phases; during GRPO, GPU 2 trains while GPU 3 runs the separated rollout
# engine. Both local endpoints are restarted for collection and evaluation.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VENV="${QWEN35_VENV:-$ROOT/.venv_qwen35}"
SERVE_VENV="${QWEN35_SERVE_VENV:-$ROOT/.venv_qwen35_sglang}"
CUDA_HOME="${QWEN35_CUDA_HOME:-$ROOT/.deps/cuda-12.8}"
SMALL_GPU="${SMALL_GPU:-2}"
LARGE_GPU="${LARGE_GPU:-3}"
TRAIN_GPU="${TRAIN_GPU:-$SMALL_GPU}"
TRAIN_GPUS="${TRAIN_GPUS:-$TRAIN_GPU,$LARGE_GPU}"
SMALL_PORT="${SMALL_PORT:-8280}"
LARGE_PORT="${LARGE_PORT:-8281}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen3.5-2B}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen3.5-4B}"
LARGE_BASE_URL="${LARGE_BASE_URL:-http://127.0.0.1:$LARGE_PORT/v1}"
N_CYCLES="${N_CYCLES:-4}"
LIMIT="${LIMIT:--1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_2b_mbpp_4cycle_$(date -u +%Y%m%d_%H%M%S)}"
LARGE_CACHE="${LARGE_CACHE:-$ROOT/results/cache/qwen35_4b_mbpp_t0_tok512.jsonl}"
SERVE_PATH="$CUDA_HOME/bin:$SERVE_VENV/bin:$PATH"
TRAIN_PYTHONPATH="$ROOT/experiments/tau-2/compat/qwen35_torch_fallback:$ROOT/experiments/tau-2:${PYTHONPATH:-}"
TRAIN_LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

export PYTHON="$VENV/bin/python"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"
# Qwen3.5 GDN can enter a zero-throughput state with larger dynamic batches.
# Two repeated 200-task MBPP pressure tests were stable at eight requests.
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-8}"
export SGLANG_LINEAR_ATTN_BACKEND="${SGLANG_LINEAR_ATTN_BACKEND:-triton}"
export SGLANG_MAMBA_SCHEDULER_STRATEGY="${SGLANG_MAMBA_SCHEDULER_STRATEGY:-no_buffer}"
export SGLANG_DISABLE_RADIX_CACHE=0
export SGLANG_DISABLE_CUDA_GRAPH=0

cleanup() {
  PORT="$SMALL_PORT" bash scripts/stop_sglang.sh >/dev/null 2>&1 || true
  PORT="$LARGE_PORT" bash scripts/stop_sglang.sh >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup
if ! LARGE_MODEL="$LARGE_MODEL" LARGE_MAX_TOKENS="${LARGE_MAX_TOKENS:-512}" \
  "$VENV/bin/python" - "$LARGE_CACHE" <<'PY'
import json, os, sys
from pathlib import Path
from verl_code_rl.collect_traces import large_cache_key, load_tasks

path = Path(sys.argv[1])
keys = set()
if path.exists():
    with path.open() as handle:
        keys = {json.loads(line)["cache_key"] for line in handle if line.strip()}
tasks = load_tasks(Path("data/raw/he_mbpp.jsonl"), "train", -1, "mbpp")
tasks += load_tasks(Path("data/raw/he_mbpp.jsonl"), "eval", -1, "mbpp")
expected = {
    large_cache_key(
        task, os.environ["LARGE_MODEL"], 0.0,
        int(os.environ["LARGE_MAX_TOKENS"]), 0,
    )
    for task in tasks
}
missing = expected - keys
print(f"[qwen35-mbpp] cache coverage={len(expected) - len(missing)}/{len(expected)}")
raise SystemExit(0 if not missing else 1)
PY
then
  if [[ "$LARGE_MODEL" == openai/* ]]; then
    echo "[qwen35-mbpp] hosted fallback cache is incomplete; refusing uncached API calls" >&2
    exit 1
  fi
  MODEL_PATH="$LARGE_MODEL" SERVED_NAME="$LARGE_MODEL" PORT="$LARGE_PORT" GPU="$LARGE_GPU" \
    PATH="$SERVE_PATH" CUDA_HOME="$CUDA_HOME" PYTHON="$SERVE_VENV/bin/python" \
    bash scripts/serve_sglang.sh
  "$VENV/bin/python" -m verl_code_rl.build_large_cache \
    --data "$ROOT/data/raw/he_mbpp.jsonl" --dataset mbpp \
    --model "$LARGE_MODEL" --base-url "http://127.0.0.1:$LARGE_PORT/v1" \
    --output "$LARGE_CACHE" --workers "${WORKERS:-8}" \
    --temperature 0.0 --max-tokens 512 --empty-retry-max-tokens 512
  PORT="$LARGE_PORT" bash scripts/stop_sglang.sh
else
  echo "[qwen35-mbpp] reusing complete fallback cache: $LARGE_CACHE"
fi

MODEL_PATH="$SMALL_MODEL" SERVED_NAME="$SMALL_MODEL" PORT="$SMALL_PORT" GPU="$SMALL_GPU" \
  PATH="$SERVE_PATH" CUDA_HOME="$CUDA_HOME" PYTHON="$SERVE_VENV/bin/python" \
  bash scripts/serve_sglang.sh

PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" \
CUDA_HOME="$CUDA_HOME" LD_LIBRARY_PATH="$TRAIN_LD_LIBRARY_PATH" PYTHONPATH="$TRAIN_PYTHONPATH" \
QWEN35_TRAIN_TRITON_OVERLAY="$ROOT/.deps/qwen35-triton33" FLA_TILELANG=0 \
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
DATA="$ROOT/data/raw/he_mbpp.jsonl" DATASET=mbpp SPLIT=train \
SMALL_MODEL="$SMALL_MODEL" LARGE_MODEL="$LARGE_MODEL" \
SMALL_SERVED_MODEL="$SMALL_MODEL" MERGE_ADAPTER_FOR_SERVING=1 \
RESUME_EXISTING="${RESUME_EXISTING:-1}" SFT_ATTN_IMPLEMENTATION=sdpa \
SFT_ENABLE_THINKING=false \
SFT_DATASET_PATH="$ROOT/experiments/tau-2/tau2_evolve/sft_dataset.py" \
SFT_DATASET_NAME=Qwen35MultiTurnSFTDataset \
SMALL_BASE_URL="http://127.0.0.1:$SMALL_PORT/v1" \
LARGE_BASE_URL="$LARGE_BASE_URL" \
LARGE_CACHE="$LARGE_CACHE" \
SMALL_STOP_CMD="PORT=$SMALL_PORT bash scripts/stop_sglang.sh; PORT=$LARGE_PORT bash scripts/stop_sglang.sh; sleep 5" \
SMALL_RELOAD_CMD="PORT=$SMALL_PORT bash scripts/stop_sglang.sh; PATH=$SERVE_PATH CUDA_HOME=$CUDA_HOME GPU=$SMALL_GPU PORT=$SMALL_PORT SERVED_NAME=$SMALL_MODEL PYTHON=$SERVE_VENV/bin/python SGLANG_MAX_RUNNING_REQUESTS=$SGLANG_MAX_RUNNING_REQUESTS SGLANG_LINEAR_ATTN_BACKEND=$SGLANG_LINEAR_ATTN_BACKEND SGLANG_MAMBA_SCHEDULER_STRATEGY=$SGLANG_MAMBA_SCHEDULER_STRATEGY bash scripts/serve_sglang.sh" \
ENABLE_SFT=1 SKIP_GRPO=0 RUN_POST_TRAIN_EVAL=1 \
N_CYCLES="$N_CYCLES" WORKERS="${WORKERS:-8}" LIMIT="$LIMIT" \
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE:-64}" COLLECT_CHUNK_TIMEOUT="${COLLECT_CHUNK_TIMEOUT:-120}" \
SMALL_MAX_TOKENS="${SMALL_MAX_TOKENS:-512}" \
LARGE_MAX_TOKENS="${LARGE_MAX_TOKENS:-512}" \
SFT_LR_SCHEDULE="${SFT_LR_SCHEDULE:-1e-7,1e-7,5e-8,5e-8}" \
SFT_EPOCHS_SCHEDULE="${SFT_EPOCHS_SCHEDULE:-1,1,1,1}" \
GRPO_LR_SCHEDULE="${GRPO_LR_SCHEDULE:-2e-6,1e-6,5e-7,5e-7}" \
MODEL_ATTN_IMPLEMENTATION=sdpa USE_REMOVE_PADDING=False \
VLLM_LANGUAGE_MODEL_ONLY=True VLLM_GDN_PREFILL_BACKEND=triton \
ENFORCE_EAGER=True \
N_GPUS=1 SEPARATE_ROLLOUT=1 ROLLOUT_N_GPUS=1 RAY_NUM_CPUS="${RAY_NUM_CPUS:-24}" \
CHECKPOINT_BACKEND=nixl_tau2 CHECKPOINT_CUSTOM_BACKEND_MODULE=tau2_evolve.nixl_checkpoint \
RAY_WORKER_SETUP_HOOK=tau2_evolve.qwen35_worker_setup.install_qwen35_worker_patches \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}" \
VAL_BEFORE_TRAIN=False TEST_FREQ=-1 \
DISTILLER_MODEL="${DISTILLER_MODEL:-openai/gpt-5.6-sol}" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  bash scripts/run_full_pipeline.sh --n-cycles "$N_CYCLES" --limit "$LIMIT" \
    --probe-only --experiment-name "$EXPERIMENT_NAME"

echo "[qwen35-mbpp] complete: results/$EXPERIMENT_NAME"
