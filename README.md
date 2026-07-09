# MERA-Evolve

MERA-Evolve is a compact evolve loop for HumanEval + MBPP code models. It keeps
the training core on `verl`, while adding the useful outer loop from
`router-skills-evolve`:

- `verl` for RL training
- vLLM for rollout inside verl and for standalone evaluation serving
- rule reward from Python test execution
- trace collection over small/large model outcomes
- a single global procedural SkillBook
- a learned prompt router
- offline ablation metrics across large / skills / router / full

## Layout

```text
data/raw/                  copied HumanEval/MBPP JSONL
data/processed/            verl parquet files, generated locally
verl_code_rl/code_eval.py   code extraction + subprocess test execution
verl_code_rl/reward.py      verl custom_reward_function.compute_score
verl_code_rl/prepare_data.py
verl_code_rl/eval_vllm.py
verl_code_rl/collect_traces.py
verl_code_rl/build_skillbook.py
verl_code_rl/train_router.py
verl_code_rl/run_ablation.py
scripts/prepare_data.sh
scripts/train_grpo.sh
scripts/serve_vllm.sh
scripts/eval_vllm.sh
scripts/run_full_pipeline.sh
```

## Setup

```bash
cd /Users/YuhangYao/Desktop/MERA-Evolve
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `verl` in the actual training environment that matches your CUDA stack.
For example, if you have a local clone:

```bash
pip install -e /path/to/verl
```

## Prepare Data

`verl` expects parquet rows with `data_source`, chat-style `prompt`, `ability`,
`reward_model.ground_truth`, and `extra_info`.

```bash
scripts/prepare_data.sh
```

Outputs:

```text
data/processed/train.parquet
data/processed/val.parquet
```

Smoke subset:

```bash
scripts/prepare_data.sh --max-train 16 --max-val 16
```

With a cycle skillbook:

```bash
python -m verl_code_rl.prepare_data \
  --traces results/EXP/cycle_0/traces.jsonl \
  --skillbook results/EXP/cycle_0/skillbook.json \
  --out-dir results/EXP/cycle_0/processed
```

## Evolve Pipeline

Fast orchestration smoke test, no model server and no verl training:

```bash
bash scripts/run_full_pipeline.sh --mock --limit 8 --experiment-name smoke_evolve
```

Outputs:

```text
results/smoke_evolve/cycle_0/
  traces.jsonl
  skillbook.json
  processed/train.parquet
  processed/val.parquet
  router/router.joblib
  e2e_ablation_summary.json
```

For a real trace collection run, start two OpenAI-compatible vLLM servers first:

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct PORT=8000 GPU=0 \
  scripts/serve_vllm.sh

MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct PORT=8001 GPU=1 \
  scripts/serve_vllm.sh
```

Then run the evolve loop:

```bash
bash scripts/run_full_pipeline.sh \
  --n-cycles 1 \
  --limit 64 \
  --small-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --large-model Qwen/Qwen2.5-Coder-3B-Instruct
```

Use `--skip-train` when you only want traces, skills, router, and ablation. By
default, non-mock runs call `scripts/train_grpo.sh` with cycle-local parquet
files and write verl checkpoints under `cycle_N/verl_checkpoints`.

For multi-cycle real training, restart the small-model vLLM server on port 8000
with the checkpoint printed at the end of the previous cycle before collecting
the next cycle's traces. The mock path and `--skip-train` path do not need this.

## Standalone vLLM Eval

Start a server:

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct \
PORT=8000 GPU=0 \
scripts/serve_vllm.sh
```

Evaluate held-out HumanEval + MBPP:

```bash
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct \
PORT=8000 \
scripts/eval_vllm.sh --split eval --dataset all --workers 32 \
  --out results/qwen25_1p5b_eval.jsonl
```

Evaluate with a distilled skill prefix:

```bash
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct PORT=8000 \
scripts/eval_vllm.sh --skillbook results/EXP/cycle_0/skillbook.json \
  --out results/qwen25_1p5b_skill_eval.jsonl
```

Stop the server:

```bash
PORT=8000 scripts/stop_vllm.sh
```

## GRPO Training With verl

Prepare data first, then:

```bash
source configs/grpo_qwen25_1p5b.env
scripts/train_grpo.sh
```

Useful smoke settings:

```bash
scripts/prepare_data.sh --max-train 16 --max-val 16
source configs/grpo_qwen25_1p5b.env
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=4 \
N_GENERATIONS=2 \
TEST_FREQ=1 \
SAVE_FREQ=5 \
OUTPUT_DIR=results/smoke_grpo/checkpoints \
scripts/train_grpo.sh trainer.total_training_steps=2
```

The train script passes:

```text
algorithm.adv_estimator=grpo
custom_reward_function.path=$PWD/verl_code_rl/reward.py
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.n=$N_GENERATIONS
```

so rollouts are generated through verl's vLLM backend and scored by local Python
test execution.

## Notes

- HumanEval and MBPP are normalized into one `he_mbpp.jsonl` file copied from the
  old project. Use `INPUT=data/raw/HumanEval.jsonl scripts/prepare_data.sh` to
  train on HumanEval only.
- Reward execution uses `python -I -` in a subprocess with a timeout. This is
  good enough for HumanEval/MBPP iteration, but it is not a hardened security
  sandbox for untrusted code.
- Keep training temperature above zero for GRPO (`TEMPERATURE=0.7` to `1.0` is a
  normal starting range), otherwise grouped rollouts can collapse to identical
  samples.
