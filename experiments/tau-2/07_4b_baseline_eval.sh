#!/usr/bin/env bash
set -euo pipefail

# Endpoint-only Qwen3.5-4B baseline on the fixed 35-task TAU-2 EVAL split.
# The same local model acts as the agent and user simulator, matching the
# Qwen3.5 experiments without SkillBook, adapters, routing, or fallback.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${TRAIN_VENV:-$ROOT/.venv_qwen35}"
CUDA_HOME="${QWEN35_CUDA_HOME:-$ROOT/.deps/cuda-12.8}"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
TAU2_PY="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
GPU="${GPU:-2}"
PORT="${PORT:-8280}"
USER_GPU="${USER_GPU:-$GPU}"
if [[ -z "${USER_PORT:-}" ]]; then
  [[ "$USER_GPU" == "$GPU" ]] && USER_PORT="$PORT" || USER_PORT="$((PORT + 1))"
fi
MODEL="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
SERVED_NAME="${SERVED_NAME:-evol-llm-4b}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/tau2_qwen35_4b_baseline_$(date -u +%Y%m%d_%H%M%S)}"
WORKERS="${EVAL_WORKERS:-1}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$WORKERS}"
GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-512}"
USER_MAX_TOKENS="${USER_MAX_TOKENS:-512}"
PREFIX_CACHE_ARGS=()
[[ "${VLLM_ENABLE_PREFIX_CACHING:-0}" == "1" ]] && PREFIX_CACHE_ARGS=(--enable-prefix-caching)

for executable in "$VENV/bin/python" "$VENV/bin/vllm" "$TAU2_PY" "$CUDA_HOME/bin/nvcc"; do
  if [[ ! -x "$executable" ]]; then
    echo "FATAL: required executable is missing: $executable" >&2
    exit 2
  fi
done

CACHED_MODEL="$HOME/.cache/huggingface/hub/models--${MODEL//\//--}/snapshots"
if [[ ! -d "$MODEL" && -d "$CACHED_MODEL" ]]; then
  MODEL="$(find "$CACHED_MODEL" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
fi
mkdir -p "$RESULTS_DIR"

printf '%s\n' \
  "model=$MODEL" \
  "agent_max_tokens=$AGENT_MAX_TOKENS" \
  "user_max_tokens=$USER_MAX_TOKENS" \
  "workers=$WORKERS" \
  "max_num_seqs=$MAX_NUM_SEQS" \
  "gdn_prefill_backend=$GDN_PREFILL_BACKEND" \
  "prefix_caching=${VLLM_ENABLE_PREFIX_CACHING:-0}" \
  > "$RESULTS_DIR/run_config.txt"

export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

SERVER_PIDS=()
cleanup() {
  ((${#SERVER_PIDS[@]})) && kill "${SERVER_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

start_server() {
  local gpu="$1" port="$2" log="$3"
  PORT="$port" "$ROOT/scripts/stop_vllm.sh" >/dev/null 2>&1 || true
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$ROOT/experiments/tau-2/compat/qwen35_torch_fallback" \
  nohup "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" --port "$port" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}" \
    --max-model-len 40960 --max-num-batched-tokens 40960 --max-num-seqs "$MAX_NUM_SEQS" \
    "${PREFIX_CACHE_ARGS[@]}" --dtype bfloat16 --trust-remote-code --enforce-eager \
    --no-async-scheduling --language-model-only --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    > "$log" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/v1/models" | grep -q "$SERVED_NAME"; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -100 "$log" >&2
      return 1
    fi
    sleep 5
  done
  return 1
}

start_server "$GPU" "$PORT" "$RESULTS_DIR/agent_server.log"
if [[ "$USER_GPU" == "$GPU" && "$USER_PORT" == "$PORT" ]]; then
  USER_PORT="$PORT"
else
  start_server "$USER_GPU" "$USER_PORT" "$RESULTS_DIR/user_server.log"
fi

VENV_SITE_PACKAGES="${TRAIN_SITE_PACKAGES:-$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')}"
COLLECT_ARGS=(
  --bucket EVAL --workers "$WORKERS" --max-steps 60 --max-errors 10
  --agent-model "openai/$SERVED_NAME" --agent-base-url "http://127.0.0.1:$PORT/v1"
  --user-model "openai/$SERVED_NAME" --user-base-url "http://127.0.0.1:$USER_PORT/v1"
  --agent-max-tokens "$AGENT_MAX_TOKENS" --user-max-tokens "$USER_MAX_TOKENS"
  --no-agent-thinking --no-user-thinking --tau2-log-level CRITICAL
  --out "$RESULTS_DIR/eval.jsonl"
)
[[ "${RESUME:-0}" == "1" ]] && COLLECT_ARGS+=(--resume)

PYTHONPATH="$ROOT/experiments/tau-2:$VENV_SITE_PACKAGES" \
  "$TAU2_PY" -m tau2_evolve.collect_traces "${COLLECT_ARGS[@]}" \
  > "$RESULTS_DIR/eval.log" 2>&1

"$VENV/bin/python" - "$RESULTS_DIR/eval.jsonl" <<'PY' | tee "$RESULTS_DIR/summary.txt"
import json
import sys
from collections import defaultdict

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
domains = defaultdict(lambda: [0, 0])
for row in rows:
    domains[row["domain"]][1] += 1
    domains[row["domain"]][0] += bool(row.get("passed"))
passed = sum(bool(row.get("passed")) for row in rows)
errors = sum(bool(row.get("error")) for row in rows)
print(f"overall: {passed}/{len(rows)} ({passed / len(rows):.1%}); errors={errors}")
for domain in ("airline", "retail", "telecom"):
    success, total = domains[domain]
    print(f"{domain}: {success}/{total} ({success / total:.1%})")
PY

test "$(wc -l < "$RESULTS_DIR/eval.jsonl")" -eq 35
echo "[tau2-4b-baseline] complete: $RESULTS_DIR"
