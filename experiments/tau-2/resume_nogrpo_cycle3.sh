#!/usr/bin/env bash
set -euo pipefail

ROOT="/shared_home/yuhang.yao/MERA-Evolve"
RESULTS_DIR="$ROOT/results/tau2_4cycle_nogrpo_retry"
OUT="$RESULTS_DIR/cycle_3"
LOG="$RESULTS_DIR/cycle_3_resume.log"
PID_FILE="$RESULTS_DIR/cycle_3_resume.pid"
AGENT_BASE_URL="http://127.0.0.1:8200/v1"
USER_BASE_URL="http://127.0.0.1:8201/v1"
TAU2_PY="/shared_home/yuhang.yao/router-skills-evolve/.venv_tau2/bin/python3"
MERA_PY="$ROOT/venv/bin/python3"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [[ "${1:-}" != "--worker" ]]; then
  for endpoint in "$AGENT_BASE_URL/models" "$USER_BASE_URL/models"; do
    curl -sf "$endpoint" >/dev/null || {
      echo "FATAL: required model server is unavailable: $endpoint" >&2
      exit 1
    }
  done

  used_mib="$(nvidia-smi --id=6 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( used_mib > 1000 )); then
    echo "FATAL: GPU 6 is not free (${used_mib} MiB used); refusing to risk another OOM" >&2
    exit 1
  fi

  if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "cycle 3 resume already running (pid=$(cat "$PID_FILE"))"
    exit 0
  fi

  nohup setsid "$0" --worker >> "$LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PID_FILE"
  sleep 2
  kill -0 "$pid" 2>/dev/null || {
    echo "FATAL: cycle 3 resume exited during startup; see $LOG" >&2
    exit 1
  }
  echo "started noGRPO cycle 3 resume (pid=$pid)"
  exit 0
fi

trap 'rm -f "$PID_FILE"' EXIT
cd "$ROOT/experiments/tau-2"
set -a
source "$ROOT/.env"
set +a

for required in train_traces.jsonl skillbook.json sft_pairs.parquet; do
  [[ -s "$OUT/$required" ]] || {
    log "FATAL: missing cycle 3 input: $OUT/$required"
    exit 1
  }
done

SFT_ADAPTER="$OUT/sft_adapter"
if [[ ! -s "$SFT_ADAPTER/adapter_model.safetensors" ]]; then
  log "resuming cycle 3 at SFT on GPU 6"
  (
    cd "$ROOT"
    source venv/bin/activate
    env CUDA_VISIBLE_DEVICES=6 \
      SFT_DATA="$OUT/sft_pairs.parquet" \
      SFT_OUTPUT_DIR="$SFT_ADAPTER" \
      MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct" \
      LORA_ADAPTER_PATH="$RESULTS_DIR/cycle_2/sft_adapter" \
      SFT_BATCH_SIZE=4 SFT_MICRO_BATCH_SIZE_PER_GPU=2 SFT_TOTAL_EPOCHS=5 \
      SFT_MAX_LENGTH=16384 \
      SFT_PROJECT_NAME=tau2_4cycle SFT_EXPERIMENT_NAME=cycle3_tau2_4cycle_nogrpo_retry \
      bash scripts/train_sft.sh data.max_token_len_per_gpu=16384 data.num_workers=0
  ) >> "$LOG" 2>&1
else
  log "cycle 3 SFT adapter already exists; skipping training"
fi

log "hot-swapping cycle 3 SFT adapter"
curl -sf -X POST "$AGENT_BASE_URL/unload_lora_adapter" \
  -H 'Content-Type: application/json' -d '{"lora_name":"evol-llm-agent"}' >/dev/null 2>&1 || true
curl -sf -X POST "$AGENT_BASE_URL/load_lora_adapter" \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"evol-llm-agent\",\"lora_path\":\"$SFT_ADAPTER\"}" >/dev/null

log "running held-out EVAL (35 tasks)"
"$TAU2_PY" collect_traces.py \
  --bucket EVAL --workers 6 --max-steps 60 \
  --agent-model openai/evol-llm-agent --agent-base-url "$AGENT_BASE_URL" \
  --user-model openai/evol-llm-user --user-base-url "$USER_BASE_URL" \
  --skillbook "$OUT/skillbook.json" --out "$OUT/eval.jsonl" >> "$LOG" 2>&1

"$MERA_PY" - "$OUT/eval.jsonl" <<'PY' >> "$LOG" 2>&1
import json, sys
from collections import defaultdict

rows = [json.loads(line) for line in open(sys.argv[1])]
by_domain = defaultdict(list)
for row in rows:
    by_domain[row["domain"]].append(row["passed"])
total = sum(row["passed"] for row in rows)
print(f"[cycle summary] total: {total}/{len(rows)} = {total / len(rows):.1%}")
for domain, values in sorted(by_domain.items()):
    print(f"[cycle summary]   {domain}: {sum(values)}/{len(values)} = {sum(values) / len(values):.1%}")
PY

log "cycle 3 resume complete"
