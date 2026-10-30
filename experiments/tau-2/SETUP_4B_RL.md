# TAU-2 4B RL: Environment and Data Setup (AWS)

Companion to `research_plan.md` at the repository root. Follow this document to prepare an AWS GPU machine from scratch, then confirm
the setup with the pipeline's `check` stage and a smoke test. Run every command from the repository root.
Everything uses official sources by default (PyPI, download.pytorch.org, GitHub, Hugging Face); no mirrors are needed.

## 0. Machine

| Item | Recommendation |
|---|---|
| Instance | `p4de.24xlarge` (8 × A100 80 GB) or `p5.48xlarge` (8 × H100 80 GB) |
| AMI | AWS Deep Learning Base GPU AMI (Ubuntu 22.04/24.04) with **NVIDIA driver ≥ 580**: the vLLM 0.29 used for evaluation ships CUDA 13 torch; the training environment is cu128 and needs ≥ 570. Check with `nvidia-smi` |
| Disk | Put code, environments, the uv cache and results on the instance NVMe (DLAMI usually mounts it at `/opt/dlami/nvme`): the three environments take about 40 GB, the uv cache about 15 GB, each GRPO run 50–100 GB of results. NVMe is wiped when the instance stops, so sync results you need to keep to S3 |
| Other | `curl`, `tar`, `python3` (any 3.x, used to install `uv`); the S3 cache needs the `aws` CLI and bucket access (DLAMI ships the CLI; grant access through the instance role) |

The machine must be able to download `zeyuyuyu/router-skills-evolve` (the tau2 workspace) from GitHub.

Directory layout (script defaults; each can be changed with an environment variable):

```text
/opt/dlami/nvme/work/
  MERA-Evolve/                 this repository
    .venv_qwen35/              training env (VERL 0.8 + vLLM 0.18.1 + torch 2.10 cu128)
    .venv_vllm/                evaluation serving env (vLLM 0.29)        SERVE_VENV
    .deps/qwen35-triton33/     Triton 3.3.1 for the training actor
    .deps/cuda-12.8/           nvcc 12.8 (for building CUDA extensions)
  router-skills-evolve/        tau2 workspace                            TAU2_WORKSPACE
    .venv_tau2/                tau2-bench simulator env
  models/Qwen3.5-4B/           student and user simulator                MODELS_DIR
  data/areal_tau2/             AReaL-tau2-data                           AREAL_DIR
```

```bash
cd /opt/dlami/nvme && mkdir -p work && cd work
git clone -b tau2-4b-rl-handoff https://github.com/yh-yao/MERA-Evolve.git
cd MERA-Evolve
export UV_CACHE_DIR=/opt/dlami/nvme/uv-cache
```

## 1. Three Python environments (about 20 minutes)

Why three: tau2-bench's dependencies (numpy 2, etc.) conflict with VERL 0.8; VERL 0.8's bundled vLLM 0.18.1 does the training-time
rollouts, while evaluation serving uses vLLM 0.29, which supports Qwen3.5 better. The three environments never import each other.

```bash
bash experiments/tau-2/scripts/setup_envs.sh            # all: tau2 train serve
bash experiments/tau-2/scripts/setup_envs.sh train      # just one
```

Each step first checks whether its target already works, so after a failure just re-run the script. What it does:

- **tau2**: downloads `router-skills-evolve@8a2cd9e` and `tau2-bench@17e07b1` at their pinned commits (tau2-bench includes voice data;
  its tarball is over 1 GB), creates `.venv_tau2` (Python 3.12), and installs tau2-bench with its knowledge and voice dependencies
  **except `pyaudio`** (it needs PortAudio headers), which `experiments/tau-2/compat/pyaudio.py` stands in for; also installs `mcp` (needed by litellm).
- **train**: creates `.venv_qwen35` and resolves everything in one pass with uv `--override`: torch 2.10 cu128, vLLM 0.18.1, VERL 0.8.0,
  numpy 1.26.4, transformers 5.13.1, scipy 1.15.3, etc. (the upstream packages' metadata conflict; see the comments in
  `requirements-qwen35-cu129.txt`). Also installs:
  `tilelang==0.1.15` (imported by the VERL training script); `fla-core==0.5.1`;
  the prebuilt `causal-conv1d==1.6.2.post1` (required for Qwen3.5's GDN fast path; without it training silently falls back to a
  torch implementation 2.5× slower); `.deps/qwen35-triton33` (FLA's backward pass rejects Triton 3.6, so the actor process uses 3.3.1);
  `.deps/cuda-12.8` (nvcc 12.8 via micromamba; the pip nvcc packages contain no nvcc binary).
- **serve**: creates `.venv_vllm` and installs `vllm==0.29.0`.

Check afterwards:

```bash
PYTHONPATH=experiments/tau-2/compat ../router-skills-evolve/.venv_tau2/bin/python -c 'import tau2; print("tau2 ok")'
.venv_qwen35/bin/python -c 'import torch, verl, vllm, fla, causal_conv1d; print(torch.__version__, verl.__version__, vllm.__version__)'
.venv_vllm/bin/python -c 'import vllm, torch; print(vllm.__version__, torch.version.cuda)'
```

## 2. Models and data (about 5 minutes)

```bash
export MODELS_DIR=/opt/dlami/nvme/work/models AREAL_DIR=/opt/dlami/nvme/work/data/areal_tau2
bash experiments/tau-2/scripts/download_assets.sh            # Qwen3.5-4B + AReaL-tau2-data (Hugging Face)
bash experiments/tau-2/scripts/download_assets.sh teacher    # optional: Qwen3.8-27B (teacher / upper bound, about 55 GB)
```

For several instances, or instances you restart often, add the S3 cache: the first machine downloads from Hugging Face and uploads
to S3, later machines pull from S3.

```bash
S3_CACHE=s3://your-bucket/mera-tau2 bash experiments/tau-2/scripts/download_assets.sh
# layout on S3: models/Qwen3.5-4B/, data/areal_tau2/
```

- `Qwen/Qwen3.5-4B`: the student and the user simulator share these weights. Every baseline number in the docs uses the 4B user simulator.
- `inclusionAI/AReaL-tau2-data` (Apache-2.0): `tau2_rl_train.jsonl` (1,982 RL tasks) + `tau2_rl_database/` (9 database snapshots).
  The first run of the pipeline's `check` stage replays the gold actions and writes `valid_task_ids.json` (CPU, about 10 minutes; 1,844 valid tasks expected).
  With the S3 cache, run `aws s3 sync $AREAL_DIR s3://.../data/areal_tau2` after `check` to store that file as well.
- Downloads resume; the script finally checks that all weight shards are present and that the task count is right.
  The download tool (`hf`) runs through `uvx`, so no environment has to exist first.

## 3. Verification (about 1 hour)

Copy the pipeline config template `experiments/tau-2/config/pipeline_4b_rl.env.example` and fill in paths and GPUs:

```bash
STUDENT_MODEL=/opt/dlami/nvme/work/models/Qwen3.5-4B
USER_MODEL=/opt/dlami/nvme/work/models/Qwen3.5-4B
VLLM_BIN=/opt/dlami/nvme/work/MERA-Evolve/.venv_vllm/bin/vllm
TAU2_WORKSPACE=/opt/dlami/nvme/work/router-skills-evolve
MERA_AREAL_DIR=/opt/dlami/nvme/work/data/areal_tau2
```

Then:

```bash
bash experiments/tau-2/10_4b_rl_pipeline.sh my.env check            # environments, models, data, valid-task list
SMOKE=1 RUN_DIR=/opt/dlami/nvme/results/smoke bash experiments/tau-2/10_4b_rl_pipeline.sh my.env   # end-to-end smoke test
```

The smoke test passes when:
- `pipeline.log` shows the 7 `stage ... done` lines in order;
- the train stage prints the truncation rate;
- `train/group_O/train.log` contains `rollout_corr/rollout_is_mean` (explicit TIS is active);
- evalckpt prints the paired comparison of each checkpoint with base.

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `import tau2` fails with PortAudio / pyaudio | `PYTHONPATH=experiments/tau-2/compat` is missing (the pipeline sets it) |
| vLLM 0.29 fails at startup with a CUDA driver version error | Driver < 580: use a newer DLAMI, or install a torch/vLLM build in `.venv_vllm` that matches the driver |
| Training log shows "fast path is not available" | `causal-conv1d` is missing: `CUDA_HOME=.deps/cuda-12.8 uv pip install --python .venv_qwen35/bin/python --no-build-isolation causal-conv1d==1.6.2.post1` |
| Training fails with `peer access is not supported` | No P2P between GPUs (consumer cards only; not A100/H100): set `MERA_WEIGHT_TRANSFER_SHM=1` |
| Two vLLM servers on one GPU fail to start | Set `VLLM_MAX_NUM_SEQS=128` (the pipeline does) |
| Evaluation scores identical to base | The adapter is not active; `08_phase0_eval_protocol.sh` asserts that log-probs differ — do not bypass it |
| Environments gone after an instance restart | NVMe is ephemeral: reinstall the environments (about 20 minutes; the uv cache on NVMe is lost too), pull models and data from S3, and sync results to S3 regularly |
| Shared machines | Never run `ray stop --force` (it kills other users' Ray processes on the node); run long jobs detached with `setsid nohup ... < /dev/null` |

On machines inside China (no direct Hugging Face access), use `MIRROR=cn` (Aliyun PyPI/PyTorch, Tsinghua conda-forge, ghfast.top) and `SOURCE=modelscope`.
