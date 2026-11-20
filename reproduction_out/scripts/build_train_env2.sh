#!/usr/bin/env bash
# Build the GRPO training stack WITHOUT re-downloading torch or vLLM:
# reuse the node's already-working `fave-vllm-py311` (Python 3.11.18, torch 2.14.0+cu126,
# vLLM 0.33.0, already verified to serve Qwen3_5ForConditionalGeneration) through a
# --system-site-packages venv, then add VERL 0.8.0.
#   - venv lives on the shared FS (JuiceFS, 969 TB free) per the operator's instruction,
#     instead of /tmp which was 80% full.
#   - VERL is installed --no-deps first so its `vllm<=0.12.0` / `numpy<2` metadata pins
#     cannot silently replace the working torch/vLLM stack.
#   - long pip timeouts: the corporate proxy read-timed-out at 15 s on the previous attempt.
set -x
BASE=/home/powerop/.conda/envs/fave-vllm-py311
V=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro/envs/venv_train
PIP="--timeout 180 --retries 10 --index-url http://artifactory.release.ctripcorp.com/artifactory/api/pypi/trip-pypi-prod/simple --trusted-host artifactory.release.ctripcorp.com"

rm -rf "$V"
mkdir -p "$(dirname "$V")"
"$BASE/bin/python" -m venv --system-site-packages "$V" || exit 2
P="$V/bin/python"

"$P" -c "import torch,vllm;print('INHERITED torch',torch.__version__,'vllm',vllm.__version__)" || exit 3

"$P" -m pip install -q -U pip setuptools wheel $PIP 2>&1 | tail -2
"$P" -m pip install --no-deps $PIP "verl==0.8.0" 2>&1 | tail -3 || exit 4

# deps VERL actually needs that are not already satisfied by the base env
"$P" -m pip install $PIP "tensordict<=0.10.0,>=0.8.0,!=0.9.0" dill codetiming pylatexenc \
  "hydra-core" "omegaconf" "torchdata" "weave" 2>&1 | tail -3

"$P" -c "import verl,vllm,torch;print('VERL',verl.__version__,'VLLM',vllm.__version__,'TORCH',torch.__version__)" || exit 5
"$P" -c "
from vllm.model_executor.models.registry import ModelRegistry
print('Qwen3_5 supported:', any('Qwen3_5' in a for a in ModelRegistry.get_supported_archs()))"
"$P" -c "import verl.trainer.ppo.ray_trainer" 2>&1 | tail -3
echo TRAIN_ENV_DONE
