#!/usr/bin/env bash
# Build the three Python environments the tau2 4B RL line uses (research_plan.md, SETUP_4B_RL.md).
#
#   bash experiments/tau-2/scripts/setup_envs.sh [tau2] [train] [serve]     # default: all three
#
#   tau2   $TAU2_WORKSPACE/.venv_tau2   tau2-bench simulator (+ router-skills-evolve stage2 package)
#          (the repo itself is not pip-installed anywhere; the launchers put it on PYTHONPATH)
#   train  <repo>/.venv_qwen35          VERL 0.8 + vLLM 0.18.1 + torch 2.10 cu128 (GRPO training, driver >= 570)
#                                       + .deps/qwen35-triton33 (actor-side Triton 3.3) + .deps/cuda-12.8 (nvcc)
#   serve  $SERVE_VENV                  vLLM 0.29 (evaluation servers, torch CUDA 13 -> driver >= 580)
#
# Environment:
#   MIRROR=cn|global   cn: Aliyun PyPI/PyTorch wheels, Tsinghua conda-forge, ghfast.top for GitHub (default global)
#   TAU2_WORKSPACE     default: <repo>/../router-skills-evolve
#   SERVE_VENV         default: <repo>/.venv_vllm
#   UV_CACHE_DIR       put it on a large disk; the three envs download ~15 GB of wheels
# Each step is skipped when its target already imports, so the script can be re-run after a failure.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
SERVE_VENV="${SERVE_VENV:-$ROOT/.venv_vllm}"
MIRROR="${MIRROR:-global}"
ROUTER_COMMIT=8a2cd9ed63aa4e065a1550a0718418a54a6fed64
TAU2_BENCH_COMMIT=17e07b1da2bbc0cadfddeea36412686e0604127b
CAUSAL_CONV1D_WHEEL=causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

if [[ "$MIRROR" == cn ]]; then
  export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
  TORCH_SRC=(--find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/)  # a flat file list, not a simple index
  GH=https://ghfast.top/https://github.com
  CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
else
  TORCH_SRC=(--extra-index-url https://download.pytorch.org/whl/cu128)
  GH=https://github.com
  CONDA_CHANNEL=conda-forge
fi
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
log() { echo "[$(date -u +%FT%TZ)] [setup] $*"; }
command -v uv > /dev/null || { log "installing uv"; python3 -m pip install -q uv; }
command -v uv > /dev/null || export PATH="$HOME/.local/bin:$PATH"

# GitHub tarball of one commit (works through ghfast.top, unlike git clone of a pinned commit).
fetch_tarball() { # owner/repo commit dest
  [[ -e "$3" ]] && return 0
  mkdir -p "$(dirname "$3")"
  local tmp; tmp="$(mktemp -d)"
  curl -fsSL --retry 3 "$GH/$1/archive/$2.tar.gz" | tar -xz -C "$tmp"
  mv "$tmp"/*-"$2" "$3"; rmdir "$tmp"
}

setup_tau2() {
  local py="$TAU2_WORKSPACE/.venv_tau2/bin/python"
  if [[ -x "$py" ]] && PYTHONPATH="$ROOT/experiments/tau-2/compat" "$py" -c "import tau2" 2> /dev/null; then
    log "tau2: ok"; return
  fi
  if [[ ! -d "$TAU2_WORKSPACE/tau2_stage2" ]]; then
    log "tau2: router-skills-evolve @ ${ROUTER_COMMIT:0:7}"
    fetch_tarball zeyuyuyu/router-skills-evolve "$ROUTER_COMMIT" "$TAU2_WORKSPACE"
  fi
  log "tau2: tau2-bench @ ${TAU2_BENCH_COMMIT:0:7}"
  fetch_tarball sierra-research/tau2-bench "$TAU2_BENCH_COMMIT" "$TAU2_WORKSPACE/tau2_stage2/code/vendor/tau2-bench"
  uv venv -q --allow-existing --python 3.12 "$TAU2_WORKSPACE/.venv_tau2"
  # The voice extra minus pyaudio (needs PortAudio headers); experiments/tau-2/compat/pyaudio.py stands in for it.
  uv pip install -q --python "$py" -e "$TAU2_WORKSPACE/tau2_stage2/code/vendor/tau2-bench[knowledge]" \
    "elevenlabs>=1.54.0" "deepgram-sdk>=3.7.3" "websockets>=13.0" "jiwer>=3.0.0" "scipy>=1.10.0" "pydub>=0.25.0" \
    "tqdm>=4.0" "aiohttp>=3.0" "google-genai>=1.0.0" "boto3>=1.35.0" "aws-sdk-bedrock-runtime>=0.3.0" \
    "google-cloud-aiplatform>=1.38.0" "google-auth>=2.23.0" mcp
  uv pip install -q --python "$py" -e "$TAU2_WORKSPACE/tau2_stage2/code"
  PYTHONPATH="$ROOT/experiments/tau-2/compat" "$py" -c "import tau2" > /dev/null && log "tau2: ok"
}

setup_train() {
  local v="$ROOT/.venv_qwen35" py="$ROOT/.venv_qwen35/bin/python"
  if [[ -x "$py" ]] && "$py" -c "import torch, verl, vllm, fla, causal_conv1d, tilelang, scipy" 2> /dev/null; then
    log "train: ok"
  else
    log "train: torch 2.10 cu128 + VERL 0.8 + vLLM 0.18.1 (see requirements-qwen35-cu129.txt for why the overrides)"
    uv venv -q --allow-existing --python 3.12 "$v"
    # Upstream metadata conflicts (VERL needs numpy<2, vLLM 0.18/OpenCV want numpy>=2, transformers 5 for Qwen3.5):
    # resolve once with overrides instead of the step-by-step pip sequence.
    local ovr; ovr="$(mktemp)"
    cat > "$ovr" <<'EOF'
torch==2.10.0+cu128
torchvision==0.25.0+cu128
torchaudio==2.10.0+cu128
numpy==1.26.4
opencv-python-headless==4.13.0.92
pandas==2.2.3
pyarrow==25.0.0
datasets==5.0.0
ray==2.56.0
tensordict==0.10.0
transformers==5.13.1
huggingface-hub==1.23.0
vllm==0.18.1
flashinfer-python==0.6.6
xgrammar==0.2.3
scipy==1.15.3
EOF
    uv pip install -q --python "$py" --override "$ovr" "${TORCH_SRC[@]}" --index-strategy unsafe-best-match \
      torch==2.10.0+cu128 vllm==0.18.1 verl==0.8.0 flashinfer-python==0.6.6 xgrammar==0.2.3 \
      cupy-cuda12x==13.6.0 nixl==1.2.0 tilelang==0.1.15 opencv-python-headless==4.13.0.92 \
      peft accelerate sentencepiece mcp setuptools scipy==1.15.3  # scipy: else the tau2 venv's numpy-2 build leaks in via site-packages
    rm -f "$ovr"
    uv pip install -q --python "$py" --no-build-isolation fla-core==0.5.1
    uv pip install -q --python "$py" "$GH/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/$CAUSAL_CONV1D_WHEEL" \
      || log "train: WARNING prebuilt causal-conv1d wheel failed; build it: CUDA_HOME=$ROOT/.deps/cuda-12.8 uv pip install --no-build-isolation causal-conv1d==1.6.2.post1"
  fi
  if [[ ! -d "$ROOT/.deps/qwen35-triton33/triton" ]]; then
    log "train: Triton 3.3.1 overlay for the FSDP actor (FLA backward rejects Triton 3.6)"
    uv pip install -q --python "$py" --no-deps --target "$ROOT/.deps/qwen35-triton33" triton==3.3.1
  fi
  if [[ ! -x "$ROOT/.deps/cuda-12.8/bin/nvcc" ]]; then
    log "train: CUDA 12.8 nvcc in .deps/cuda-12.8 (micromamba; pip nvcc wheels ship no nvcc binary)"
    local mm="$ROOT/.deps/micromamba"
    if [[ ! -x "$mm" ]]; then
      local mm_url=https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
      [[ "$MIRROR" == cn ]] && mm_url="https://ghfast.top/$mm_url"
      curl -fsSL --retry 3 -o "$mm" "$mm_url"; chmod +x "$mm"
    fi
    MAMBA_ROOT_PREFIX="$ROOT/.deps/mamba" "$mm" create -y -q -p "$ROOT/.deps/cuda-12.8" \
      -c "$CONDA_CHANNEL" --override-channels cuda-nvcc=12.8 cuda-cudart-dev=12.8
    [[ -e "$ROOT/.deps/cuda-12.8/lib64" ]] || ln -s targets/x86_64-linux/lib "$ROOT/.deps/cuda-12.8/lib64"
  fi
  "$py" -c "import torch, verl, vllm, fla, causal_conv1d, tilelang; print('train: ok torch', torch.__version__, 'verl', verl.__version__, 'vllm', vllm.__version__)"
}

setup_serve() {
  local py="$SERVE_VENV/bin/python"
  if [[ -x "$py" ]] && "$py" -c "import vllm" 2> /dev/null; then log "serve: ok"; return; fi
  log "serve: vLLM 0.29 (Qwen3.5 serving; torch CUDA 13 wheels need driver >= 580)"
  uv venv -q --allow-existing --python 3.12 "$SERVE_VENV"
  uv pip install -q --python "$py" vllm==0.29.0
  "$py" -c "import vllm, torch; print('serve: ok vllm', vllm.__version__, 'torch cuda', torch.version.cuda)"
}

STEPS=("$@"); [[ ${#STEPS[@]} -gt 0 ]] || STEPS=(tau2 train serve)
for s in "${STEPS[@]}"; do "setup_$s"; done
log "done. Next: experiments/tau-2/scripts/download_assets.sh, then the pipeline's check stage."
