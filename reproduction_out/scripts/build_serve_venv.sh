#!/usr/bin/env bash
# Phase 3 adaptation: build the evaluation-serving venv outside the repo tree.
# Deviates from experiments/tau-2/scripts/setup_envs.sh in exactly one way:
#   --torch-backend=cu130  (the plan's comment says cu128; this node runs a verified
#   torch 2.13.0+cu130 stack on driver 550.90.07 via /usr/local/cuda/compat, and there
#   is no cu128 toolchain here. See environment_report.md.)
set -euo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p /tmp/mera_envs
uv venv -q --allow-existing --python 3.12 "$SERVE_VENV"
uv pip install --python "$SERVE_VENV/bin/python" --torch-backend=cu130 vllm==0.29.0
"$SERVE_VENV/bin/python" -c "import vllm, torch; print('serve ok vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
