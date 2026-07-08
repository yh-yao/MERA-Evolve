# verl-code-rl

Clean GRPO training scaffold for code models on HumanEval + MBPP, using:

- `verl` for RL training
- vLLM for rollout inside verl and for standalone evaluation serving
- rule reward from Python test execution
- the copied data from the old project under `data/raw/`

This project intentionally does not copy the old router/skills/cycle pipeline.
It keeps only the parts needed to train and evaluate a code model.

## Layout

```text
data/raw/                  copied HumanEval/MBPP JSONL
data/processed/            verl parquet files, generated locally
verl_code_rl/code_eval.py   code extraction + subprocess test execution
verl_code_rl/reward.py      verl custom_reward_function.compute_score
verl_code_rl/prepare_data.py
verl_code_rl/eval_vllm.py
scripts/prepare_data.sh
scripts/train_grpo.sh
scripts/serve_vllm.sh
scripts/eval_vllm.sh
```

## Setup

```bash
cd /Users/YuhangYao/Desktop/verl-code-rl
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
