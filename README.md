# MERA-Evolve

MERA-Evolve is a compact evolve loop for HumanEval + MBPP code models. It keeps
the training core on `verl`, while adding the useful outer loop from
`router-skills-evolve`:

- `verl` for RL training
- vLLM for rollout inside verl and for standalone evaluation serving
- rule reward from Python test execution
- oracle trace collection separated from the deployed routing policy
- a per-dataset SkillBook (one skill each for HumanEval/MBPP; router still owns
  per-prompt routing, the SkillBook only distills a procedure prefix)
- a cost-calibrated learned prompt router (frozen text-embedding features, not
  TF-IDF)
- held-out policy ablation across large / skills / router / full

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
verl_code_rl/traces_to_sft.py  teacher + self-repair SFT pair extraction (parquet)
verl_code_rl/extract_sft_lora_adapter.py  reconstructs a loadable LoRA adapter from a verl SFT checkpoint
verl_code_rl/embedding.py   frozen text-embedding featurizer shared by the router
verl_code_rl/train_router.py
verl_code_rl/run_ablation.py
verl_code_rl/trace_diagnostics.py
scripts/prepare_data.sh
scripts/train_grpo.sh
scripts/train_sft.sh           explicit version-pinned SFT launcher hook
scripts/reload_small_vllm.sh   local checkpoint reload helper
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
  --skillbook results/EXP/cycle_0/skillbook \
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
  trace_diagnostics.json
  skillbook/
    skill.md
    skill_statistics.json
  sft_pairs.jsonl
  processed/train.parquet
  processed/val.parquet
  router_train_traces.jsonl
  router_calibration_traces.jsonl
  router/router.joblib
  eval_traces.jsonl
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

Use `--skip-train` when you only want traces, skills, router, and held-out
ablation. By default, non-mock runs call `scripts/train_grpo.sh` with cycle-local
parquet files and write verl checkpoints under `cycle_N/verl_checkpoints`.

### What a cycle measures

Each cycle keeps two kinds of data separate:

```text
oracle train traces → SkillBook → optional SFT pairs → verl GRPO
                                         ↓
current checkpoint → router-train shard + router-calibration shard → calibrated router
                                         ↓
fixed eval split → oracle outcomes + deployed cascade simulation → ablation
```

`traces.jsonl` always records small and large oracle outcomes unless
`--probe-only` is set. `policy_*` records the deployment outcome: router-large
goes straight to the large model, while router-small uses small and falls back
to large only after a failed execution. The ablation includes this fallback cost
instead of treating a router-small decision as free.

### SkillBook: one skill per dataset, structured like a real runbook

`extract_signature()` buckets each task by dataset (`humaneval`/`mbpp`), not
into one global bucket -- HumanEval (docstring + doctest-driven) and MBPP
(plain-English + assert-driven) are different enough task shapes to deserve
different solving advice. Each skill is rendered as distinct, independently
scannable sections (`_render_sections` in `verl_code_rl/skills.py`) rather
than one flowing paragraph:

```text
# Skill: humaneval
## When to use
...
## Procedure
1. ...
2. ...
## Common pitfalls
- ...
## Recurring patterns
- ...
```

`when_to_use`/`procedure` are fixed, hand-written per dataset
(`_STATIC_SKILL_SECTIONS`) and never change -- they're the scaffolding, always
present even fully offline. `pitfalls`/`patterns` are the only sections an LLM
distiller touches (`--distiller-model`, `make_llm_distiller`): it's asked for
exactly those two labeled bullet lists, grounded in real successful exemplars,
so a bad distillation can corrupt at most the learned sections, never the
static scaffolding. Without a distiller configured, those two sections are
simply empty -- no frequency-counted or fabricated content fills the gap.

To enable it, point `DISTILLER_MODEL` at a CommonStack-hosted model (an
OpenAI-compatible aggregator, same one `router-skills-evolve` uses --
`https://api.commonstack.ai/v1`, provider-prefixed model ids like
`openai/gpt-5.5`):

```bash
DISTILLER_MODEL=openai/gpt-5.5 bash scripts/run_full_pipeline.sh --limit 64
```

`run_full_pipeline.sh` auto-loads `.env` (never committed) for
`COMMONSTACK_API_KEY`, and defaults `DISTILLER_BASE_URL` to CommonStack's URL
-- override either if you'd rather point the distiller at a local model
instead. Local small/large model traffic never touches CommonStack; only the
distiller call does.

`--output`/`--previous`/`--skillbook` all point at a **directory**, written by
`SkillBook.save()` as two files:

```text
skillbook/
  skill.md               the real skill -- exactly what gets injected into prompts
  skill_statistics.json  stats/history/exemplars/pitfalls/patterns skill.md is derived from
```

`skill.md` is the human-facing artifact -- open it directly to see (and diff,
across cycles) exactly what's being prepended to prompts. `skill_statistics
.json` is the source of truth `skill.md` is deterministically re-rendered
from on load (`Skill.from_dict` calls `_render_sections` again rather than
trusting a cached string), so the two files can never drift out of sync.
To measure whether the SkillBook is actually helping, compare two independent
`eval_vllm.sh` runs on the same held-out split, with and without
`--skillbook` -- there's no need
for a dedicated ablation arm for this since `eval_vllm.py`'s `--skillbook` flag
is already optional and independent of routing.

### Router: frozen embedding features, not TF-IDF

`train_router.py` featurizes prompts with a frozen text-embedding model
(`verl_code_rl/embedding.py`, default `Qwen/Qwen3-Embedding-0.6B`, last-token
pooling) instead of a `TfidfVectorizer`, then fits a `LogisticRegression` head
on those embeddings. The classifier alone is joblib-dumped to `router.joblib`;
`router_meta.json` records which embedding model produced its features
(`embedding_model`), and every caller (`collect_traces.py`, `run_ablation.py`)
reads that field back so training and inference always agree on the
featurizer. Override the embedding model with `--embed-model` /
`ROUTER_EMBED_MODEL`. This is a genuine semantic signal rather than a
bag-of-words one, at the cost of a forward pass per routing decision instead
of near-zero-cost string vectorization.

The Router learns a failure probability from `SMALL_SAMPLES` independent small
rollouts. Its threshold is selected on a distinct deterministic task-id shard,
not on its training traces. Set actual model rates to make cost units USD:

```bash
SMALL_INPUT_COST_PER_MILLION=0.30 SMALL_OUTPUT_COST_PER_MILLION=1.20 \
LARGE_INPUT_COST_PER_MILLION=3.00 LARGE_OUTPUT_COST_PER_MILLION=12.00 \
SMALL_SAMPLES=4 ROUTER_TARGET_PASS_RATE=0.95 \
bash scripts/run_full_pipeline.sh --limit 64
```

### Checkpoint reload is required for real closed loops

vLLM does not load a new checkpoint merely because the OpenAI request supplies
a new model name. After GRPO, the pipeline therefore requires
`SMALL_RELOAD_CMD` before it will perform post-training calibration/evaluation
or begin the next cycle. For a local server managed by this repository:

```bash
SMALL_RELOAD_CMD='scripts/reload_small_vllm.sh' \
GPU=0 bash scripts/run_full_pipeline.sh --n-cycles 2 --limit 64
```

The helper understands a Hugging Face model or a standard LoRA adapter. If a
given `verl` checkpoint is not directly vLLM-loadable, export/merge it in your
cluster-specific reload command; the pipeline deliberately refuses to silently
score the old server.

### Optional SFT warm start

`traces_to_sft.py` extracts teacher pairs (`small fail ∧ large pass`) and
self-repair pairs, writing **parquet** with a `messages` column whose final
turn is the assistant's actual completion -- this matches verl's native
`MultiTurnSFTDataset` contract exactly (it reads `messages` directly and
computes the loss mask from `role == "assistant"`; it does not read a separate
`completion` field).

```bash
ENABLE_SFT=1 bash scripts/run_full_pipeline.sh --limit 64
```

By default `scripts/train_sft.sh` runs verl's own native SFT trainer
(`torchrun -m verl.trainer.sft_trainer`), LoRA by default (same algorithm as
`train_grpo.sh`: `LORA_RANK=16`/`LORA_ALPHA=32`/`LORA_TARGET_MODULES=
[q_proj,k_proj,v_proj,o_proj]`). verl's generic FSDP checkpoint saver doesn't
write a `lora_adapter/adapter_config.json` the way the GRPO actor does (that
logic is specific to `verl/workers/fsdp_workers.py`'s PPO path, not shared by
the SFT trainer) -- `scripts/train_sft.sh` runs `verl_code_rl/
extract_sft_lora_adapter.py` right after training to reconstruct the
LoRA-wrapped model and re-save just the adapter via PEFT's own
`save_pretrained`, written directly to `SFT_OUTPUT_DIR` so it's immediately
loadable by `serve_vllm.sh` and picked up by the next GRPO cycle's
`LORA_ADAPTER_PATH` continuation. This extractor currently only supports
single-GPU (`N_GPUS=1`, FSDP `NO_SHARD`) checkpoints.

If your verl installation's SFT trainer differs, set `SFT_TRAIN_CMD` to
override with your own version-pinned launcher instead:

```bash
ENABLE_SFT=1 \
SFT_TRAIN_CMD='your-version-pinned-sft-launcher --data "$SFT_DATA" --output "$SFT_OUTPUT_DIR" --model "$MODEL_PATH"' \
bash scripts/run_full_pipeline.sh --limit 64
```

Without `ENABLE_SFT=1`, the SFT artifact is still produced for inspection and
GRPO starts from the current small model.

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
scripts/eval_vllm.sh --skillbook results/EXP/cycle_0/skillbook \
  --out results/qwen25_1p5b_skill_eval.jsonl
```

Stop the server:

```bash
PORT=8000 scripts/stop_vllm.sh
```

## GRPO Training With verl

Prepare data first, then:

```bash
source configs/humaneval_mbpp/grpo_qwen25_1p5b.env
scripts/train_grpo.sh
```

Useful smoke settings:

```bash
scripts/prepare_data.sh --max-train 16 --max-val 16
source configs/humaneval_mbpp/grpo_qwen25_1p5b.env
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

### LoRA is the default training mode

`scripts/train_grpo.sh` trains a LoRA adapter (`LORA_RANK=16`, `LORA_ALPHA=32`,
`LORA_TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj]`) on a frozen base model by
default, matching the algorithm `router-skills-evolve` uses for its HumanEval
GRPO trainer: train against the raw base model, produce an adapter, then hand
that adapter to serving. Set `LORA_RANK=0` to fall back to full-parameter
fine-tuning.

verl syncs the adapter into its own internal vLLM rollout engine every step via
vLLM's dynamic `add_lora`/`remove_lora` API — no `--enable-lora` flag needs to
be passed to that internal engine manually. The trained adapter is written to
`global_step_N/actor/lora_adapter/` (`adapter_config.json` +
`adapter_model.safetensors`); `scripts/run_full_pipeline.sh` picks this up as
the next cycle's `CURRENT_MODEL` in preference to the full checkpoint.
`scripts/serve_vllm.sh` already auto-detects a LoRA adapter directory (checks
for `adapter_config.json`, resolves the base model from it) and serves it with
`vllm serve <base> --enable-lora --lora-modules <name>=<adapter_dir>
--enforce-eager` — this is the same kill-and-relaunch handoff
`router-skills-evolve` uses (no live/dynamic reload API; `SMALL_RELOAD_CMD`
just restarts the standalone serving process pointed at the new adapter).

Across multiple cycles, `actor_rollout_ref.model.path` must stay pinned to the
original base checkpoint (verl loads it fresh with `AutoModelForCausalLM`,
which can't load a bare adapter directory). `run_full_pipeline.sh` therefore
tracks `BASE_SMALL_MODEL` separately from `CURRENT_MODEL`: once `CURRENT_MODEL`
becomes an adapter dir, it's passed to `scripts/train_grpo.sh` as
`LORA_ADAPTER_PATH`, which forwards it as `actor_rollout_ref.model.
lora_adapter_path` so verl continues training the *same* adapter next cycle
(loaded via `PeftModel.from_pretrained(..., is_trainable=True)`), rather than
starting a fresh one each cycle. This differs slightly from
`router-skills-evolve`'s own algorithm, which merges the previous adapter into
the base and attaches a brand-new LoRA each cycle (to keep the GRPO KL
reference clean) — verl has no built-in merge-then-fresh option, so this repo
uses verl's native continuation mechanism instead.

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
- `REWARD_COMPILE_BONUS` is disabled by default, preserving binary execution
  reward. If enabled, keep it at or below `0.1`; `trace_diagnostics.json` should
  guide whether all-pass/all-fail GRPO groups need more sampling or a harder
  task mix.
