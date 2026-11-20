#!/usr/bin/env bash
# Restart ONLY the 9B teacher with the parser combination research_plan.md actually specifies
# for the teacher (--reasoning-parser qwen3 + --tool-call-parser qwen3_coder).
# The qwen3_xml + qwen3 reasoning-parser combination produced empty assistant messages
# (neither content nor tool_calls) on 28% of teacher turns.
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
PORT=8270
pid="$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
pid="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
[[ -n "$pid" ]] && { kill "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null; }
sleep 5
LOG=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro/results/servers_teacher/evol-llm-teacher_server.log
CUDA_VISIBLE_DEVICES=2 OPENAI_API_KEY=EMPTY VLLM_USE_FLASHINFER_SAMPLER=0 \
setsid nohup /home/powerop/.conda/envs/fave-vllm-py311/bin/vllm serve \
  "$MODELS_DIR/Qwen3.5-9B" --served-model-name evol-llm-teacher --port "$PORT" \
  --gpu-memory-utilization 0.85 --max-model-len 131072 --enable-prefix-caching \
  --dtype bfloat16 --trust-remote-code --language-model-only --gdn-prefill-backend triton \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q evol-llm-teacher && { echo "teacher ready ($i)"; exit 0; }
  sleep 5
done
echo "teacher NOT ready"; tail -20 "$LOG"; exit 1
