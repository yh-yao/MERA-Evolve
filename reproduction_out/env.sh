# Environment for the MERA-Evolve reproduction run (Phase 3 adaptation layer).
# Kept OUTSIDE the repo source tree; nothing in MERA-Evolve sources is modified.
export MERA_REPO=/home/powerop/work/yjh/MERA-Evolve
export MERA_OUT=$MERA_REPO/reproduction_out
# venvs stay on the local overlay for speed; everything else that persists goes to the share.
export TAU2_WORKSPACE=/tmp/mera_envs/router-skills-evolve
export SERVE_VENV=/tmp/mera_envs/venv_vllm
export TRAIN_VENV=/tmp/mera_envs/venv_qwen35
export UV_HTTP_TIMEOUT=300
# 08_phase0_eval_protocol.sh takes its python from TRAIN_PY (not TRAIN_VENV) for the
# Qwen3.5 adapter key-conversion and the adapter-active assertion; without it 08 falls back
# to $ROOT/.venv_qwen35, which does not exist on this machine.
export TRAIN_PY=${TRAIN_PY:-/home/powerop/work/zsc/newproj/.venv/bin/python}
# Heavy, persistent artifacts go to the JuiceFS share (969 TB free).
export MERA_STORE=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
export MODELS_DIR=/data/htlcrosssale-gps-share/powerop_transfer/models
export MERA_AREAL_DIR=$MERA_STORE/data/areal_tau2
export RESULTS_ROOT=$MERA_STORE/results
export HF_HUB_OFFLINE=0
# 2026-11-18: the local overlay is the scarce resource. Keep caches off it: the training run
# already used the shared HF cache; these defaults make every later stage agree.
export SHARED_CACHE=/data/htlcrosssale-gps-share/powerop_transfer/cache
export HF_HOME=${HF_HOME:-/data/htlcrosssale-gps-share/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$SHARED_CACHE/uv}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SHARED_CACHE/pip}
export TORCH_HOME=${TORCH_HOME:-$SHARED_CACHE/torch}
mkdir -p "$SHARED_CACHE"/{uv,pip,torch} 2>/dev/null || true
