#!/usr/bin/env bash
# Approved retail-first, TRAIN/AReaL-only proxy SFT pilot. One arm at a time.
set -euo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

REPO=/home/powerop/work/yjh/MERA-Evolve
D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
PLAN=$D/results/e9_retail_utility_proxy_v3/plan/proxy_plan.json
RUN=$D/results/e9_retail_utility_proxy_v3/run_rerun_v3
TRAIN_VENV=/home/powerop/work/zsc/newproj/.venv
TRAIN_PY=$TRAIN_VENV/bin/python
VLLM_BIN=/home/powerop/.conda/envs/fave-vllm-py311/bin/vllm
TRITON_OVERLAY=$REPO/.deps/qwen35-triton33
LOG=$D/logs/retail_utility_proxy_rerun_v3.log
mkdir -p "$RUN" "$D/logs"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

stop_ours() {
  local port pid cmd
  for port in "$@"; do
    pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)
    [[ -n "$pid" ]] || continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    [[ "$cmd" == *"evol-llm-agent"* || "$cmd" == *"evol-llm-user"* ]] || { log "FATAL unexpected process holds :$port"; return 1; }
    kill "$pid" 2>/dev/null || true
  done
  sleep 8
}

gpu_free() {
  local gpu=$1 used
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" -lt 1024 ]] || { log "FATAL GPU$gpu not free: ${used}MiB"; return 1; }
}

preflight() {
  "$TRAIN_PY" - "$PLAN" <<'PY'
import hashlib,json,sys
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_text())
assert plan['kind']=='tau2_retail_utility_proxy_plan' and plan['scope']=='retail-first pilot'
assert len(plan['cohorts'])==3 and plan['prohibited']==['VAL','EVAL','airline proxy training','threshold tuning']
def sha(path):
 h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
for item in plan['cohorts']:
 c=json.loads(Path(item['path']).read_text())
 assert c['scope']=='retail-first utility-selector pilot' and c['cohort'].startswith('areal|retail|')
 assert c['evaluation']['seeds']==[300] and c['evaluation']['trials']==4
 assert c['frozen_train_dev']['sha256']==sha(c['frozen_train_dev']['partition'])
 assert all(x['bucket']=='UTILITY_DEV' and x['domain']=='retail' for x in c['frozen_train_dev']['tasks'])
 for arm in ('treatment','control'): assert c['runs'][arm]['sha256']==sha(c['runs'][arm]['train_file'])
print('PRECHECK PASS: final v3, retail-only, frozen UTILITY_DEV, zero VAL/EVAL')
PY
}

run_arm() { # cohort-folder arm train_gpu eval_agent_gpu eval_user_gpu agent_port user_port
  local folder=$1 arm=$2 train_gpu=$3 agent_gpu=$4 user_gpu=$5 port=$6 uport=$7
  local contract="$folder/proxy_contract.json" out="$RUN/$(basename "$folder")/$arm"
  local train_file partition cohort seed adapter eval_file
  readarray -t values < <("$TRAIN_PY" - "$contract" "$arm" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); arm=sys.argv[2]
for value in (c['runs'][arm]['train_file'], c['frozen_train_dev']['partition'], c['cohort'], c['evaluation']['seeds'][0]): print(value)
PY
)
  train_file=${values[0]}; partition=${values[1]}; cohort=${values[2]}; seed=${values[3]}
  adapter="$out/adapter"; eval_file="$out/eval/${arm}_UTILITY_DEV_seed${seed}.jsonl"
  [[ ! -e "$out" ]] || { log "FATAL refusing existing arm output: $out"; return 1; }
  gpu_free "$train_gpu"; mkdir -p "$out"
  log "SFT START cohort=$cohort arm=$arm gpu=$train_gpu rows=$("$TRAIN_PY" -c "import pandas as p; print(len(p.read_parquet('$train_file')))" )"
  ( cd "$REPO" && source "$TRAIN_VENV/bin/activate" && \
    env CUDA_VISIBLE_DEVICES="$train_gpu" PYTHON="$TRAIN_PY" SFT_DATA="$train_file" SFT_OUTPUT_DIR="$adapter" \
    MODEL_PATH="$MODELS_DIR/Qwen3.5-4B" SFT_BATCH_SIZE=2 SFT_MICRO_BATCH_SIZE_PER_GPU=1 SFT_LR=1e-7 \
    SFT_TOTAL_EPOCHS=1 SFT_MAX_LENGTH=24576 SFT_MAX_TOKEN_LEN_PER_GPU=24576 TRAINING_SEED=1 \
    QWEN35_ENABLE_VERL_PATCHES=1 PYTHONPATH="$TRITON_OVERLAY:$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat:${PYTHONPATH:-}" \
    LORA_TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" FLA_TILELANG=0 \
    SFT_ENABLE_THINKING=False SFT_ATTN_IMPLEMENTATION=sdpa SFT_USE_TORCH_COMPILE=False \
    SFT_DATASET_PATH="$REPO/experiments/tau-2/tau2_evolve/sft_dataset.py" SFT_DATASET_NAME=Qwen35MultiTurnSFTDataset \
    SFT_SAVE_FREQ=-1 SFT_PROJECT_NAME=tau2_utility_proxy SFT_EXPERIMENT_NAME="$(basename "$folder")-$arm" \
    bash scripts/train_sft.sh data.num_workers=0 ) >> "$out/train.log" 2>&1
  [[ -s "$adapter/adapter_model.safetensors" ]] || { log "FATAL adapter missing: $adapter"; return 1; }
  gpu_free "$train_gpu"; gpu_free "$agent_gpu"; gpu_free "$user_gpu"
  log "EVAL START cohort=$cohort arm=$arm bucket=UTILITY_DEV seed=$seed"
  ( cd "$REPO" && RESULTS_DIR="$out/eval" BUCKETS=UTILITY_DEV RUN_SEEDS="$seed" RUN_LABEL="$arm" TRIALS=4 \
    BASE_MODEL="$MODELS_DIR/Qwen3.5-4B" USER_MODEL_PATH="$MODELS_DIR/Qwen3.5-4B" AGENT_ADAPTER="$adapter" \
    AGENT_GPU="$agent_gpu" USER_GPU="$user_gpu" AGENT_PORT="$port" USER_PORT="$uport" VLLM_BIN="$VLLM_BIN" \
    TRAIN_PY="$TRAIN_PY" TAU2_PARTITION_PATH="$partition" bash experiments/tau-2/08_phase0_eval_protocol.sh ) >> "$out/eval.log" 2>&1
  [[ -s "$eval_file" ]] || { log "FATAL proxy eval missing: $eval_file"; return 1; }
  "$TRAIN_PY" - "$contract" "$eval_file" <<'PY'
import json,sys
from pathlib import Path
c=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines() if x]
allowed={(str(x['domain']),str(x['task_id'])) for x in c['frozen_train_dev']['tasks']}
assert {(str(x['domain']),str(x['task_id'])) for x in rows} == allowed
assert len(rows)==4*len(allowed) and all(isinstance(x.get('passed'),bool) and not x.get('error') and x.get('trial') in range(4) for x in rows)
print(f'proxy eval verified rows={len(rows)} tasks={len(allowed)}')
PY
  stop_ours "$port" "$uport"; gpu_free "$agent_gpu"; gpu_free "$user_gpu"
  log "ARM DONE cohort=$cohort arm=$arm"
}

preflight
for folder in "$D/results/e9_retail_utility_proxy_v3/plan/cohorts"/*; do
  [[ -d "$folder" ]] || continue
  run_arm "$folder" treatment 0 1 2 8310 8311
  run_arm "$folder" control 0 1 2 8310 8311
  out="$RUN/$(basename "$folder")" contract="$folder/proxy_contract.json"
  cohort=$("$TRAIN_PY" -c "import json; print(json.load(open('$contract'))['cohort'])")
  "$TRAIN_PY" "$REPO/experiments/tau-2/scripts/label_proxy_utility.py" --contract "$contract" \
    --control-eval "$out/control/eval/control_UTILITY_DEV_seed300.jsonl" \
    --treatment-eval "$out/treatment/eval/treatment_UTILITY_DEV_seed300.jsonl" \
    --cohort "$cohort" --output "$out/utility_label.json" | tee "$out/utility_label.log"
done
log "RETAIL_UTILITY_PROXY_RERUN_V3_DONE"
