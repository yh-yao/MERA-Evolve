#!/usr/bin/env bash
# Re-run E6 step 0 cleanly.
# Two defects fixed relative to the earlier teacher runs:
#   1. earlier thinking-ON runs produced empty assistant messages;
#   2. even with thinking off, runs failed with ContextWindowExceededError because the
#      9B teacher's verbose turns push multi-turn tau2 conversations past max-model-len 40960.
# research_plan.md explicitly says to provision the required context rather than silently truncate,
# and this node has 95 GiB/card versus the plan's 40 GiB, so max-model-len is raised to 65536.
# Nothing about the model, prompt, algorithm, reward or decoding changes.
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
mkdir -p "$D/results/servers_teacher"

kill_port() {
  local pid
  pid="$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$pid" ]] && { kill "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null; }
  sleep 5
}
kill_port 8270; kill_port 8271

start() { # name model gpu port thinking
  local name=$1 model=$2 gpu=$3 port=$4 think=$5
  local kw='{"enable_thinking": false}'; [[ $think == on ]] && kw='{"enable_thinking": true}'
  CUDA_VISIBLE_DEVICES=$gpu OPENAI_API_KEY=EMPTY VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid nohup /home/powerop/.conda/envs/fave-vllm-py311/bin/vllm serve "$model" \
      --served-model-name "$name" --port "$port" --gpu-memory-utilization 0.85 \
      --max-model-len 65536 --enable-prefix-caching --dtype bfloat16 --trust-remote-code \
      --language-model-only --gdn-prefill-backend triton \
      --default-chat-template-kwargs "$kw" \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      > "$D/results/servers_teacher/${name}_server.log" 2>&1 < /dev/null &
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q "\"$name\"" && { echo "$name ready"; return 0; }
    sleep 5
  done
  echo "$name FAILED"; return 1
}

start evol-llm-teacher "$MODELS_DIR/Qwen3.5-9B" 2 8270 off || exit 1
start evol-llm-user2   "$MODELS_DIR/Qwen3.5-4B" 3 8271 off || exit 1

rm -f "$D/results/e6_teacher_val"/teacher_*.jsonl
for S in 300 1300; do
  PYTHONPATH=/home/powerop/work/yjh/MERA-Evolve/experiments/tau-2:/home/powerop/work/yjh/MERA-Evolve/experiments/tau-2/compat \
  TAU2_PARTITION_PATH=/home/powerop/work/yjh/MERA-Evolve/experiments/tau-2/data/partition_v2.json \
  OPENAI_API_KEY=EMPTY "$TAU2_WORKSPACE/.venv_tau2/bin/python" -m tau2_evolve.collect_traces \
    --bucket VAL --trials 4 --seed "$S" \
    --agent-model openai/evol-llm-teacher --agent-base-url http://127.0.0.1:8270/v1 \
    --user-model openai/evol-llm-user2 --user-base-url http://127.0.0.1:8271/v1 \
    --user-temperature 0.7 --agent-temperature 0.0 --no-agent-thinking --no-user-thinking \
    --workers 24 --max-steps 60 \
    --out "$D/results/e6_teacher_val/teacher_VAL_seed${S}.jsonl" \
    > "$D/logs/e6_teacher_val_seed${S}.log" 2>&1
  echo "seed $S done" >> "$D/logs/e6_teacher_rerun.log"
done
echo ALLDONE >> "$D/logs/e6_teacher_rerun.log"
