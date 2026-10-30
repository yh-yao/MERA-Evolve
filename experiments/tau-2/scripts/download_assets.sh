#!/usr/bin/env bash
# Download the models and data of the tau2 4B RL line (research_plan.md, SETUP_4B_RL.md).
#
#   MODELS_DIR=/abs/models AREAL_DIR=/abs/data/areal_tau2 bash experiments/tau-2/scripts/download_assets.sh [models] [areal] [teacher]
#
#   models   Qwen/Qwen3.5-4B                -> $MODELS_DIR/Qwen3.5-4B   (student and user simulator, ~9 GB)
#   areal    inclusionAI/AReaL-tau2-data    -> $AREAL_DIR               (tau2_rl_train.jsonl + tau2_rl_database/, ~1 GB)
#   teacher  Qwen/Qwen3.8-27B (optional)    -> $MODELS_DIR/Qwen3.8-27B  (only for teacher / upper-bound runs, ~55 GB)
# Default: models areal.
#
# SOURCE=hf (default) downloads from the Hugging Face Hub; SOURCE=modelscope only for machines inside China.
# Both CLIs run through `uvx`, so no environment has to be prepared first.
# S3_CACHE=s3://bucket/prefix (AWS): pull each asset from S3 when it is there, otherwise download it from
# the hub and push it to S3, so later instances skip the hub. Needs the `aws` CLI and bucket access.
# Downloads resume; finished targets are verified and skipped.
set -euo pipefail
: "${MODELS_DIR:?set MODELS_DIR (absolute)}" "${AREAL_DIR:?set AREAL_DIR (absolute)}"
SOURCE="${SOURCE:-hf}"
log() { echo "[$(date -u +%FT%TZ)] [download] $*"; }
command -v uvx > /dev/null || { python3 -m pip install -q uv; export PATH="$HOME/.local/bin:$PATH"; }
[[ "$SOURCE" == modelscope && -z "${UV_INDEX_URL:-}" ]] && export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

S3_CACHE="${S3_CACHE:-}"; S3_CACHE="${S3_CACHE%/}"
s3_has() { [[ -n "$S3_CACHE" ]] && aws s3 ls "$S3_CACHE/$1/" > /dev/null 2>&1; }
s3_pull() { log "$1: syncing from $S3_CACHE/$1"; aws s3 sync --only-show-errors "$S3_CACHE/$1" "$2"; }
s3_push() { [[ -n "$S3_CACHE" ]] || return 0; log "$1: caching to $S3_CACHE/$1"; aws s3 sync --only-show-errors --exclude ".cache/*" "$2" "$S3_CACHE/$1"; }
[[ -z "$S3_CACHE" ]] || command -v aws > /dev/null || { log "FATAL S3_CACHE is set but the aws CLI is missing"; exit 1; }

fetch() { # model|dataset repo dest
  if [[ "$SOURCE" == modelscope ]]; then
    uvx --from modelscope modelscope download "--$1" "$2" --local_dir "$3"
  else
    local t=(); [[ "$1" == dataset ]] && t=(--repo-type dataset)
    uvx --from huggingface_hub hf download "$2" "${t[@]}" --local-dir "$3"
  fi
}

model_ok() { # dir
  [[ -s "$1/config.json" && -s "$1/tokenizer_config.json" ]] || return 1
  python3 - "$1" <<'EOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
files = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
missing = [f for f in files if not os.path.exists(os.path.join(d, f))]
sys.exit(1 if missing else 0)
EOF
}

get_model() { # repo name
  local dest="$MODELS_DIR/$2"
  if model_ok "$dest"; then log "$2: ok ($dest)"; return; fi
  if s3_has "models/$2"; then s3_pull "models/$2" "$dest"; model_ok "$dest" && { log "$2: ok (from S3)"; return; }; fi
  log "$2: downloading $1 from $SOURCE"
  fetch model "$1" "$dest"
  model_ok "$dest" || { log "FATAL $2: incomplete download in $dest"; exit 1; }
  s3_push "models/$2" "$dest"
  log "$2: ok ($(du -sh "$dest" | cut -f1))"
}

areal_ok() {
  [[ -s "$AREAL_DIR/tau2_rl_train.jsonl" ]] || return 1
  [[ $(ls "$AREAL_DIR"/tau2_rl_database/ 2> /dev/null | wc -l) -ge 9 ]] || return 1
  [[ $(wc -l < "$AREAL_DIR/tau2_rl_train.jsonl") -ge 1900 ]]
}

get_areal() {
  if areal_ok; then log "areal: ok ($AREAL_DIR)"; return; fi
  if s3_has data/areal_tau2; then s3_pull data/areal_tau2 "$AREAL_DIR"; areal_ok && { log "areal: ok (from S3)"; return; }; fi
  log "areal: downloading inclusionAI/AReaL-tau2-data from $SOURCE"
  fetch dataset inclusionAI/AReaL-tau2-data "$AREAL_DIR"
  areal_ok || { log "FATAL areal: expected tau2_rl_train.jsonl (>= 1900 tasks) and 9 files in tau2_rl_database/"; exit 1; }
  s3_push data/areal_tau2 "$AREAL_DIR"
  log "areal: ok, $(wc -l < "$AREAL_DIR/tau2_rl_train.jsonl") RL tasks; the pipeline's check stage builds valid_task_ids.json"
}

STEPS=("$@"); [[ ${#STEPS[@]} -gt 0 ]] || STEPS=(models areal)
mkdir -p "$MODELS_DIR" "$AREAL_DIR"
for s in "${STEPS[@]}"; do
  case "$s" in
    models) get_model Qwen/Qwen3.5-4B Qwen3.5-4B ;;
    teacher) get_model Qwen/Qwen3.8-27B Qwen3.8-27B ;;
    areal) get_areal ;;
    *) log "unknown step $s"; exit 1 ;;
  esac
done
log "done. In the pipeline config: STUDENT_MODEL=USER_MODEL=$MODELS_DIR/Qwen3.5-4B MERA_AREAL_DIR=$AREAL_DIR"
