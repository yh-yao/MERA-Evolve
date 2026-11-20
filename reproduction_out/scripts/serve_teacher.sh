#!/usr/bin/env bash
# E6 step 0: serve the frozen Qwen3.5-9B candidate teacher and a frozen Qwen3.5-4B
# user simulator, so the teacher can be compared against the 4B student on VAL under
# the same simulator, tasks, trial seeds and (for the parser) the same protocol.
#
# Teacher thinking stays ON (research_plan.md E6); the user simulator stays OFF and is
# the same Qwen3.5-4B checkpoint used for every other policy.
set -euo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

AGENT_GPU="${AGENT_GPU:-2}"; AGENT_PORT="${AGENT_PORT:-8270}"
USER_GPU="${USER_GPU:-3}";   USER_PORT="${USER_PORT:-8271}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
VLLM_BIN="${VLLM_BIN:-/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm}"
RESULTS_DIR="${RESULTS_DIR:-$MERA_STORE/results/servers_teacher}"
mkdir -p "$RESULTS_DIR"

start() { # name model gpu port thinking
  local name="$1" model="$2" gpu="$3" port="$4" thinking="$5"
  local kwargs=("--default-chat-template-kwargs" '{"enable_thinking": false}')
  local extra=()
  if [[ "$thinking" == on ]]; then
    kwargs=("--default-chat-template-kwargs" '{"enable_thinking": true}')
    extra=(--reasoning-parser "$REASONING_PARSER")
  fi
  if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\""; then
    echo "$name already serving on :$port"; return 0
  fi
  local pid; pid="$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  echo "starting $name model=$model gpu=$gpu port=$port thinking=$thinking"
  CUDA_VISIBLE_DEVICES="$gpu" OPENAI_API_KEY=EMPTY VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid nohup "$VLLM_BIN" serve "$model" --served-model-name "$name" --port "$port" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN" \
      --enable-prefix-caching --dtype bfloat16 --trust-remote-code \
      --language-model-only --gdn-prefill-backend "${GDN_PREFILL_BACKEND:-triton}" \
      "${kwargs[@]}" --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER" \
      "${extra[@]}" > "$RESULTS_DIR/${name}_server.log" 2>&1 < /dev/null &
  echo $! > "$RESULTS_DIR/${name}_server.pid"
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\"" && { echo "$name ready"; return 0; }
    kill -0 "$(cat "$RESULTS_DIR/${name}_server.pid")" 2>/dev/null || { echo "FATAL $name died"; tail -30 "$RESULTS_DIR/${name}_server.log"; exit 1; }
    sleep 5
  done
  echo "FATAL $name not ready"; exit 1
}

start evol-llm-teacher "$MODELS_DIR/Qwen3.5-9B" "$AGENT_GPU" "$AGENT_PORT" on
start evol-llm-user2   "$MODELS_DIR/Qwen3.5-4B" "$USER_GPU"  "$USER_PORT"  off
echo "teacher + user ready"
