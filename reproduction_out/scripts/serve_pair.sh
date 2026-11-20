#!/usr/bin/env bash
# Start the two Qwen3.5 vLLM servers exactly as experiments/tau-2/08_phase0_eval_protocol.sh
# does (same flags, same served names), so collection/evaluation see the documented protocol.
# External adaptation only: VLLM_BIN points at this node's working vLLM instead of .venv-vllm.
#
# MAX_MODEL_LEN deviation from the protocol's 40960 -> 131072 (all four servers use the same
# value). Rationale: AReaL's first request already carries the full tau2 domain policy plus
# every tool definition, so a subset of tasks overflowed 40960 and litellm raised
# ContextWindowExceededError before a single message was written (verified: errored rows have
# 0 messages). research_plan.md requires provisioning the required context instead of silently
# truncating, and permits changing the serving profile for it; requests still over the limit
# are recorded as errors, never truncated. Raising the teacher 40960 -> 65536 earlier cut its
# error rate from 28% to 0.7%. This changes the serving profile, so it must stay frozen for
# every later comparison and belongs in the reproduction report.
set -euo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

AGENT_GPU="${AGENT_GPU:-0}"; AGENT_PORT="${AGENT_PORT:-8260}"
USER_GPU="${USER_GPU:-1}";   USER_PORT="${USER_PORT:-8261}"
MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-4B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
VLLM_BIN="${VLLM_BIN:-/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm}"
RESULTS_DIR="${RESULTS_DIR:-$MERA_STORE/results/servers}"
mkdir -p "$RESULTS_DIR"

stop_port() {
  local pid
  # `|| true`: under `set -o pipefail` an empty ss/grep result would abort the script.
  pid="$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}

start() { # name model gpu port
  local name="$1" model="$2" gpu="$3" port="$4"
  if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\""; then
    echo "$name already serving on :$port"; return 0
  fi
  stop_port "$port"
  echo "starting $name ($model) gpu=$gpu port=$port"
  CUDA_VISIBLE_DEVICES="$gpu" OPENAI_API_KEY=EMPTY VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid nohup "$VLLM_BIN" serve "$model" --served-model-name "$name" --port "$port" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN" \
      --enable-prefix-caching --dtype bfloat16 --trust-remote-code \
      --language-model-only --gdn-prefill-backend "${GDN_PREFILL_BACKEND:-triton}" \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER" \
      > "$RESULTS_DIR/${name}_server.log" 2>&1 < /dev/null &
  echo $! > "$RESULTS_DIR/${name}_server.pid"
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\"" && { echo "$name ready"; return 0; }
    kill -0 "$(cat "$RESULTS_DIR/${name}_server.pid")" 2>/dev/null || { echo "FATAL $name died"; tail -30 "$RESULTS_DIR/${name}_server.log"; exit 1; }
    sleep 5
  done
  echo "FATAL $name not ready"; exit 1
}

start evol-llm-agent "$MODEL" "$AGENT_GPU" "$AGENT_PORT"
start evol-llm-user  "$MODEL" "$USER_GPU"  "$USER_PORT"
echo "both servers ready"
