# MERA-Evolve

This file is the operational guide for people and coding agents working in
this repository. Commands assume the repository root as the working directory.

## What this project does

MERA-Evolve trains and evaluates small language-model agents with a repeated
closed loop:

1. collect trajectories from the current student model;
2. obtain stronger, verified demonstrations for student failures;
3. distill reusable skills from successful trajectories;
4. update a LoRA adapter with SFT and optionally GRPO through VERL;
5. reload the adapter, evaluate on held-out tasks, and repeat;
6. optionally train a router that escalates difficult states to a stronger
   model.

The repository supports two benchmark adapters behind one launcher:

- `humaneval_mbpp`: single-turn Python code generation with executable tests;
- `tau2`: multi-turn customer-service agents that call airline, retail, and
  telecom tools in tau2-bench environments.

Use `scripts/run_experiment.sh` for both. Benchmark-specific implementations
live under `experiments/`; shared code-model training and serving utilities
live under `scripts/` and `verl_code_rl/`. Do not force the two benchmarks to
share trajectory schemas or reward functions: the common contract is the
experiment lifecycle and launcher, while each adapter owns its environment,
data conversion, and reward implementation.

## Repository layout

```text
scripts/run_experiment.sh       unified public experiment launcher
scripts/run_full_pipeline.sh    HumanEval/MBPP closed-loop implementation
scripts/{serve,stop}_vllm.sh    common model-server lifecycle
scripts/train_{sft,grpo}.sh     code-benchmark VERL launchers
verl_code_rl/                   code data, reward, skills, router, evaluation
experiments/humaneval_mbpp/     reproducible code benchmark recipes
experiments/tau-2/              tau2 adapter and reproducible recipes
  tau2_evolve/                  collection, OPD, skills, data, router plugins
  scripts/                      tau2 cycle and VERL launchers
  config/                       VERL multi-turn agent-loop configuration
  compat/                       Qwen3.5/VERL/vLLM compatibility shims
data/raw/                       HumanEval and MBPP source data
results/                        run outputs; not source code
tests/                          CPU unit and orchestration tests
```

List every supported benchmark/recipe pair:

```bash
scripts/run_experiment.sh --list
```

Legacy numbered scripts remain supported, but new runbooks and automation
should call the unified launcher.

## Common setup

Prerequisites are Linux, Bash 4+, Python 3.10+, NVIDIA drivers compatible with
the selected PyTorch wheel, and enough GPUs for serving plus training. Clone
the repository and create the general environment:

```bash
git clone git@github.com:yh-yao/MERA-Evolve.git
cd MERA-Evolve
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
# Install a VERL version compatible with this machine's torch/CUDA stack.
pip install verl
```

Store credentials in the untracked `.env` file. The unified launcher and both
pipelines load it automatically:

```bash
cat > .env <<'EOF'
COMMONSTACK_API_KEY=replace-me
EOF
chmod 600 .env
```

Never commit `.env`, API keys, generated adapters, model caches, or `results/`.
Run CPU tests before launching expensive jobs:

```bash
pytest -q
bash -n scripts/*.sh experiments/*/*.sh experiments/tau-2/scripts/*.sh
```

## HumanEval and MBPP

This adapter uses `data/raw/he_mbpp.jsonl`. Its small and fallback models are
OpenAI-compatible vLLM servers; SFT and GRPO use VERL LoRA training. Start the
servers on separate GPUs:

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct GPU=0 PORT=8000 \
  scripts/serve_vllm.sh
MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct GPU=1 PORT=8001 \
  scripts/serve_vllm.sh
```

Run recipes through the common launcher:

```bash
scripts/run_experiment.sh humaneval_mbpp skills
SFT_GPU=2 scripts/run_experiment.sh humaneval_mbpp sft
TRAIN_GPU=2 scripts/run_experiment.sh humaneval_mbpp grpo
SFT_GPU=2 SMALL_RELOAD_GPU=0 N_CYCLES=4 \
  scripts/run_experiment.sh humaneval_mbpp 4cycle-sft
GRPO_GPU=2 SMALL_RELOAD_GPU=0 N_CYCLES=4 \
  scripts/run_experiment.sh humaneval_mbpp 4cycle-sft-grpo
```

For a no-GPU orchestration smoke test:

```bash
bash scripts/run_full_pipeline.sh --mock --limit 8 \
  --experiment-name smoke_evolve
```

Useful overrides are `RESULTS_DIR`, `TRAIN_LIMIT`, `EVAL_LIMIT`, `WORKERS`,
`SMALL_MODEL`, `LARGE_MODEL`, `SMALL_BASE_URL`, and `LARGE_BASE_URL`. Detailed
recipe defaults are documented in `experiments/README.md` and in each numbered
script header.

Stop externally started servers when they are no longer needed:

```bash
PORT=8000 scripts/stop_vllm.sh
PORT=8001 scripts/stop_vllm.sh
```

## TAU-2 setup

TAU-2 needs two environments because the benchmark simulator and the modern
Qwen3.5 VERL/SGLang stack have conflicting dependency sets. By default,
MERA-Evolve discovers a sibling checkout named `router-skills-evolve`:

```text
work/
  MERA-Evolve/
  router-skills-evolve/
```

The sibling checkout must contain:

- `.venv_tau2/bin/python3` with tau2-stage2 dependencies;
- `tau2_stage2/code/vendor/tau2-bench`;
- `tau2_stage2/data_processed/stage2_v1/partition.json`.

Bootstrap that checkout when it is not already available:

```bash
git clone git@github.com:zeyuyuyu/router-skills-evolve.git \
  ../router-skills-evolve
git -C ../router-skills-evolve checkout 8a2cd9ed63aa4e065a1550a0718418a54a6fed64
mkdir -p ../router-skills-evolve/tau2_stage2/code/vendor
git clone https://github.com/sierra-research/tau2-bench \
  ../router-skills-evolve/tau2_stage2/code/vendor/tau2-bench
git -C ../router-skills-evolve/tau2_stage2/code/vendor/tau2-bench \
  checkout 17e07b1da2bbc0cadfddeea36412686e0604127b

python3.12 -m venv ../router-skills-evolve/.venv_tau2
TAU2_PIP=../router-skills-evolve/.venv_tau2/bin/pip
$TAU2_PIP install -U pip
$TAU2_PIP install -e '../router-skills-evolve/tau2_stage2/code/vendor/tau2-bench[voice,knowledge]'
$TAU2_PIP install -e ../router-skills-evolve/tau2_stage2/code
```

The tau2 package imports optional voice modules eagerly. Install the PortAudio
runtime/development package with the machine's OS or conda package manager if
`import tau2` reports a `pyaudio`/PortAudio error.

If the checkout lives elsewhere, set `TAU2_WORKSPACE`. Individual components
can be overridden with `TAU2_PYTHON`, `TAU2_STAGE2_ROOT`,
`TAU2_PARTITION_PATH`, and `TAU2_SITE_PACKAGES`.

Create the Qwen3.5 training/serving environment using the tested sequence in
`requirements-qwen35-cu129.txt`. That file is a version record, not a lockfile
for a fresh resolver. The important constraints are:

- use a CUDA-12-compatible PyTorch build on the CUDA 12.9 driver;
- install VERL and vLLM before the no-deps vLLM/Transformers upgrades;
- keep NumPy 1.26 for VERL;
- build FLA extensions with the CUDA 12.8 toolkit;
- put Triton 3.3 only in `.deps/qwen35-triton33`, not in vLLM's global path.

The expected training locations are `.venv_qwen35`, `.deps/cuda-12.8`, and
`.deps/qwen35-triton33`. Other layouts are supported through `TRAIN_VENV`,
`QWEN35_CUDA_HOME`, and `QWEN35_TRAIN_TRITON_OVERLAY`. For Qwen3.5 code-task
collection and evaluation, use a separate local-NVMe serving environment:

```bash
python3.12 -m venv /local_nvme/mera-sglang
/local_nvme/mera-sglang/bin/pip install -r requirements-qwen35-serving.txt
export QWEN35_SERVE_VENV=/local_nvme/mera-sglang
```

The split is intentional: VERL 0.8 uses vLLM 0.18.1 internally, while the
external endpoints use SGLang 0.5.10. The MBPP launcher bounds each external
server lifetime with restartable 64-task chunks and merges trained LoRA
adapters into standalone checkpoints before serving. This avoids the
long-lived Qwen3.5 GDN zero-throughput state and SGLang's incomplete direct
Qwen3.5 LoRA support on the current CUDA-12.9 driver.

Verify both environments before a full run:

```bash
TAU2_WORKSPACE=${TAU2_WORKSPACE:-../router-skills-evolve}
"$TAU2_WORKSPACE/.venv_tau2/bin/python3" -c 'import tau2; print("tau2 ok")'
.venv_qwen35/bin/python -c 'import torch, verl, vllm; print(torch.version.cuda)'
.venv_qwen35/bin/vllm --version
"$QWEN35_SERVE_VENV/bin/python" -c 'import torch, sglang; print(torch.version.cuda, sglang.__version__)'
```

## Running TAU-2

The full TAU-2 runner starts and health-checks its own student and user vLLM
servers. A four-cycle Qwen3.5 run needs four GPU assignments: student serving,
user simulation, training, and separate VERL rollout.

```bash
export TAU2_WORKSPACE=/path/to/router-skills-evolve
export TRAIN_VENV=$PWD/.venv_qwen35
export BASE_MODEL=Qwen/Qwen3.5-2B
export USER_MODEL_PATH=Qwen/Qwen3.5-4B
export AGENT_GPU=0 AGENT_PORT=8260
export USER_GPU=1 USER_PORT=8261
export TRAIN_GPU=2 ROLLOUT_GPU=3
export RESULTS_DIR=$PWD/results/tau2_qwen35_$(date -u +%Y%m%d_%H%M%S)

scripts/run_experiment.sh tau2 4cycle
```

Available TAU-2 recipes:

```bash
scripts/run_experiment.sh tau2 skills
scripts/run_experiment.sh tau2 sft
scripts/run_experiment.sh tau2 sft-grpo
scripts/run_experiment.sh tau2 4cycle
TRAIN_FILE=/absolute/path/to/results/run/verl_grpo.parquet \
TRAIN_GPU=2 ROLLOUT_GPU=3 scripts/run_experiment.sh tau2 grpo
FALLBACK_GPU=2 USER_PORT=8261 scripts/run_experiment.sh tau2 fallback-smoke
```

`4cycle` is the recommended reproducible full run. It fixes `N_CYCLES=4`,
Qwen3.5-2B/4B model defaults, separate training and rollout GPUs, local OPD,
and the validated worker/training settings. `sft` and `sft-grpo` remain
generic wrappers for ablations and custom cycle counts. The full cycle is:

1. run all 97 TRAIN tasks with the current student and no independent fallback;
2. replay failed prefixes with the OPD teacher and retain only verified fixes;
3. build the domain SkillBook and verified multi-turn SFT parquet;
4. train SFT, then optionally run VERL multi-turn GRPO;
5. hot-swap the resulting LoRA into the student vLLM server;
6. rerun TRAIN, train the escalation router, and evaluate 35 held-out tasks;
7. carry the adapter and SkillBook into the next cycle.

The recommended recipe uses the local Qwen3.5-4B server for user simulation,
OPD, SkillBook distillation, and routed fallback; no hosted teacher is needed.
Override `OPD_TEACHER_*` and `DISTILLER_*` to use another OpenAI-compatible
endpoint. Router-gated evaluation uses `OPD_TEACHER_*` for escalation.

Important tuning variables include `COLLECT_WORKERS`, `EVAL_WORKERS`,
`OPD_WORKERS`, `OPD_BRANCH_ATTEMPTS`, `SFT_LR`, `SFT_TOTAL_EPOCHS`,
`SFT_BALANCE_DOMAINS`,
`GRPO_TOTAL_STEPS`, `GRPO_TRAIN_BATCH_SIZE`, `GRPO_N_GENERATIONS`, and
`GRPO_ACTOR_LR`. Start with repository defaults; worker counts depend on CPU,
endpoint rate limits, and vLLM queue capacity.

For Qwen3.5, the default SFT learning rate is `1e-7` for one epoch. SFT rows
are deduplicated by benchmark task and domain oversampling is disabled by
default; use `SFT_BALANCE_DOMAINS=1` only for a deliberate ablation.

tau2 agent and user-simulator API requests set `temperature=0.0` explicitly
for reproducible collection and evaluation. This does not change verl's GRPO
rollout temperature, which is controlled by the GRPO recipe.

### Resume TAU-2

Use the same absolute `RESULTS_DIR`. Completed artifacts can be reused, and
VERL resumes SFT checkpoints automatically:

```bash
export RESULTS_DIR=/absolute/path/to/results/tau2_run
export START_CYCLE=1
export REUSE_EXISTING_ARTIFACTS=1
export INITIAL_ADAPTER="$(cat "$RESULTS_DIR/cycle_0/verl_grpo/final_adapter_path.txt")"
export INITIAL_SKILLBOOK=/absolute/path/to/cycle_0/skillbook.json
scripts/run_experiment.sh tau2 4cycle
```

For a failure inside the current cycle, set `START_CYCLE` to that cycle and
leave its completed files in place. Do not point `INITIAL_ADAPTER` at a VERL
checkpoint directory; it must contain `adapter_config.json` and
`adapter_model.safetensors`.

### Monitor TAU-2

```bash
tail -F "$RESULTS_DIR/pipeline.log"
rg '== cycle|cycle summary|training SFT|GRPO training|Traceback|FATAL' \
  "$RESULTS_DIR/pipeline.log"
nvidia-smi
```

Key outputs per cycle are `train_traces.jsonl`, `opd_traces.jsonl`,
`skillbook.json`, `sft_pairs.parquet`, `sft_adapter/`, `verl_grpo/`,
`router.json`, `eval.jsonl`, and `eval_routed.jsonl`. The run root also stores
the effective non-secret settings in `config_snapshot.json`.

## Architecture and invariants

- VERL is an installed dependency, not vendored source.
- Training uses LoRA by default. A new adapter must be reloaded before claims
  are made about post-training evaluation.
- HumanEval/MBPP reward executes generated Python in a timeout-limited
  subprocess. It is not a security sandbox for untrusted code.
- TAU-2's official pass signal remains the evaluation metric. Golden-action
  coverage is training shaping and demonstration filtering, not test reward.
- OPD prefixes are context only; SFT loss is applied to verified teacher
  suffixes. Parquet null `_trainable` markers mean the marker was omitted and
  therefore default to trainable.
- GRPO needs stochastic grouped rollouts. Keep temperature above zero and
  inspect reward variance before interpreting an update.
- TRAIN and EVAL splits must remain fixed. Never train on `eval.jsonl` or tune
  a router threshold against held-out EVAL outcomes.
- `RESULTS_DIR` is immutable experiment evidence. Resume in place or start a
  new directory; do not silently mix artifacts from different configs.

## Adding another benchmark

Create `experiments/<benchmark>/` with numbered reproducible recipes. Keep
environment interaction, trace conversion, and reward logic inside that
adapter. Register public recipes in `scripts/run_experiment.sh`, document the
required environment variables here, and add a `--dry-run` dispatch test plus
unit tests for data/reward boundaries. Reuse shared vLLM and VERL launchers
when their contracts fit; do not copy benchmark-specific TAU-2 assumptions
into generic code.
