#!/usr/bin/env bash
# Bounded attempt to build the GRPO training stack (research_plan.md Section 3 / SETUP_4B_RL.md).
# The plan pins torch 2.10 cu128 + vLLM 0.18.1 + VERL 0.8.0. Previous attempt via uv hung because
# --torch-backend=cu130 resolves against download.pytorch.org (unreachable). This attempt uses pip
# through the node's configured corporate PyPI proxy instead, and installs VERL with --no-deps so
# its `vllm<=0.12.0` metadata pin cannot silently drag in a Qwen3.5-incapable vLLM.
set -x
PY=/home/powerop/.conda/envs/hf-cu130-py312/bin/python
V=/tmp/mera_envs/venv_train
rm -rf "$V"
"$PY" -m venv "$V"
P="$V/bin/python"
"$P" -m pip install -q -U pip setuptools wheel 2>&1 | tail -2
"$P" -m pip install --index-url http://artifactory.release.ctripcorp.com/artifactory/api/pypi/trip-pypi-prod/simple \
   --trusted-host artifactory.release.ctripcorp.com torch torchvision 2>&1 | tail -3
"$P" -c "import torch;print('TORCH',torch.__version__,torch.version.cuda)" || exit 2
"$P" -m pip install --index-url http://artifactory.release.ctripcorp.com/artifactory/api/pypi/trip-pypi-prod/simple \
   --trusted-host artifactory.release.ctripcorp.com "vllm==0.18.1" 2>&1 | tail -5
"$P" -c "import vllm;print('VLLM',vllm.__version__)" || exit 3
"$P" -m pip install --index-url http://artifactory.release.ctripcorp.com/artifactory/api/pypi/trip-pypi-prod/simple \
   --trusted-host artifactory.release.ctripcorp.com --no-deps "verl==0.8.0" 2>&1 | tail -3
"$P" -m pip install --index-url http://artifactory.release.ctripcorp.com/artifactory/api/pypi/trip-pypi-prod/simple \
   --trusted-host artifactory.release.ctripcorp.com "numpy<2" ray'>=2.41' "tensordict<=0.10.0,>=0.8.0,!=0.9.0" \
   peft accelerate hydra-core omegaconf torchdata dill codetiming pylatexenc 2>&1 | tail -3
"$P" -c "import verl, vllm, torch; print('VERL',verl.__version__,'VLLM',vllm.__version__,'TORCH',torch.__version__)"
"$P" -c "
from vllm.model_executor.models.registry import ModelRegistry
print('Qwen3_5 supported:', any('Qwen3_5' in a for a in ModelRegistry.get_supported_archs()))"
echo "TRAIN_ENV_ATTEMPT_DONE"
