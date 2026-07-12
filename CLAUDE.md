# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

MERA-Evolve is a compact "evolve loop" for HumanEval/MBPP code models. It keeps
`verl` as the RL training core (installed separately, not vendored here) and
adds the outer loop from the sibling project `router-skills-evolve`
(`/shared_home/yuhang.yao/router-skills-evolve`): oracle trace collection, a
SkillBook, a cost-calibrated learned prompt router, and held-out policy
ablation. When in doubt about the intent of a piece of outer-loop logic (e.g.
SFT pair extraction, the training/reload handoff), the sibling project is the
reference implementation this repo trimmed down and rewired onto `verl` — with
two deliberate exceptions: the SkillBook's signature scheme and the router's
featurizer diverge from that project on purpose (see the SkillBook/Router
notes in the architecture section below) rather than porting its current
single-global-skill + TF-IDF design.

`verl` itself is not vendored in this repo — it's installed into a venv
(`pip install -e /path/to/verl`) matching the local CUDA stack. Reward scoring
in `verl_code_rl/` has zero dependency on verl internals (stdlib-only), so it
can be read/tested without a GPU or a working verl install.

## Setup and commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # openai, pandas, pyarrow, scikit-learn, joblib, pytest
pip install -e /path/to/verl           # separate, CUDA-stack-specific
```

Run all tests (pytest config sets `pythonpath=["."]`, `testpaths=["tests"]`):

```bash
pytest
pytest tests/test_evolve_loop.py::test_skillbook_distills_procedure_from_successes  # single test
```

Data prep (raw JSONL -> verl parquet):

```bash
scripts/prepare_data.sh                                   # data/raw/he_mbpp.jsonl -> data/processed/{train,val}.parquet
scripts/prepare_data.sh --max-train 16 --max-val 16        # smoke subset
INPUT=data/raw/HumanEval.jsonl scripts/prepare_data.sh     # HumanEval only
```

Fastest way to sanity-check the whole orchestration (no GPU, no model servers,
no verl needed — routes through deterministic hashed mock results):

```bash
bash scripts/run_full_pipeline.sh --mock --limit 8 --experiment-name smoke_evolve
```

Real cycle (needs two OpenAI-compatible vLLM servers first):

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct PORT=8000 GPU=0 scripts/serve_vllm.sh
MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct  PORT=8001 GPU=1 scripts/serve_vllm.sh

bash scripts/run_full_pipeline.sh --n-cycles 1 --limit 64 \
  --small-model Qwen/Qwen2.5-Coder-1.5B-Instruct --large-model Qwen/Qwen2.5-Coder-3B-Instruct
PORT=8000 scripts/stop_vllm.sh
```

Standalone GRPO training (after `prepare_data.sh`):

```bash
source configs/humaneval_mbpp/grpo_qwen25_1p5b.env
scripts/train_grpo.sh                        # wraps `python -m verl.trainer.main_ppo`
scripts/train_grpo.sh trainer.total_training_steps=2   # extra verl overrides pass through as argv
```

Standalone eval against a running vLLM server:

```bash
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct PORT=8000 \
scripts/eval_vllm.sh --split eval --dataset all --workers 32 --out results/eval.jsonl
```

## Architecture

### Reward path (the only piece verl actually calls into)

`verl_code_rl/reward.py:compute_score` is registered in verl via
`custom_reward_function.path=$PWD/verl_code_rl/reward.py` /
`custom_reward_function.name=compute_score` (see `scripts/train_grpo.sh`). It
delegates to `code_eval.py:score_solution`, which: extracts a code block from
the model response (`extract_code`), decodes the task JSON stored in
`reward_model.ground_truth` (`load_ground_truth`), and executes the task's
test string in a `python -I -` subprocess with a timeout (`run_code_tests`).
Reward is strictly binary (1.0/0.0) unless `REWARD_COMPILE_BONUS` is set,
which adds a small shaping bonus (capped at 0.1) for code that merely
compiles — kept off by default so pass/fail stays the dominant signal. This
subprocess sandbox is not hardened against untrusted/malicious code, only
adequate for HumanEval/MBPP-style iteration.

### One evolve cycle (`scripts/run_full_pipeline.sh`)

Each cycle strictly separates *oracle* data (used to improve the model/skills)
from *deployed policy* data (used to measure what a router-gated cascade would
actually cost/pass), and does these steps in order, threading `CURRENT_MODEL`,
the skillbook, and the router forward from the previous cycle:

1. `collect_traces.py` — for each task, run the small model (optionally
   multiple independent samples for `small_pass_rate`) and, unless the router
   + probe-only logic decides otherwise, the large model too. Every row
   records both oracle outcomes (`small_success`, `large_success`) *and* a
   simulated deployed decision (`policy_*` fields / `final_*`): a router
   decision of "small" is a cascade that falls back to large only if small's
   execution fails, while "large" bypasses small entirely. This fallback cost
   is what makes the ablation numbers meaningful instead of pretending
   router-small decisions are free.
2. `build_skillbook.py` — folds trace outcomes into `skills.py`'s `SkillBook`
   and re-distills each skill from successful exemplars. `extract_signature
   (prompt, dataset)` buckets into exactly two skills, `"humaneval"` and
   `"mbpp"` (falls back to `"coding"` for anything else) — deliberately *not*
   `router-skills-evolve`'s current single global bucket. Each skill renders
   as distinct sections (`_render_sections`) — `# Skill: <name>` /
   `## When to use` / `## Procedure` / `## Common pitfalls` / `## Recurring
   patterns` — not one flowing paragraph. `when_to_use`/`procedure` are
   fixed, hand-written per dataset (`_STATIC_SKILL_SECTIONS`) and never
   change; `pitfalls`/`patterns` are the only sections `--distiller-model`
   touches (`make_llm_distiller` asks for exactly those two labeled bullet
   lists, grounded in real successful exemplars) — deliberately *not* a tally
   of recurring function/call names from exemplars, and structured so a bad
   distillation can corrupt only the learned sections, never the static
   scaffolding. Without a distiller, those two sections are simply empty.
   `--output`/`--previous`/`--skillbook` all point at a *directory*:
   `SkillBook.save()` writes `skill.md` (the real skill, exactly what gets
   injected into prompts) and `skill_statistics.json` (stats/history/
   exemplars/pitfalls/patterns — the source data skill.md is re-rendered
   from on every load, so the two can't drift out of sync).
3. `trace_diagnostics.py` — reports reward variance / all-pass / all-fail
   rates from the small-sample rollouts, meant to catch degenerate GRPO
   groups (see Notes below on temperature).
4. `traces_to_sft.py` — pulls two kinds of SFT pairs out of oracle traces:
   teacher pairs (small failed, large passed) and self-repair pairs (a later
   turn in `small_turns` succeeded after an earlier turn failed). Writes
   **parquet** with a `messages` column whose final turn is the actual
   completion — verl's native `MultiTurnSFTDataset` reads `messages` directly
   and masks loss by `role == "assistant"`; it does not read a separate
   `completion` field. Always produced for inspection; only fed into an
   actual SFT run if `ENABLE_SFT=1`. `scripts/train_sft.sh` defaults to
   verl's own native SFT trainer (`torchrun -m verl.trainer.sft_trainer`,
   LoRA by default) unless `SFT_TRAIN_CMD` overrides it with a different
   (version/FSDP-specific) launcher. Since verl's generic FSDP checkpoint
   saver doesn't produce a `lora_adapter/adapter_config.json` the way the
   GRPO actor does, `train_sft.sh` runs `extract_sft_lora_adapter.py`
   afterward (single-GPU/`NO_SHARD` only) to reconstruct the PEFT-wrapped
   model and re-save just the adapter, written directly to `SFT_OUTPUT_DIR`
   so it's immediately `serve_vllm.sh`-loadable and picked up by the next
   GRPO cycle's `LORA_ADAPTER_PATH` continuation.
5. `prepare_data.py` — builds the cycle-local `train.parquet`/`val.parquet`
   verl expects (`data_source`, chat `prompt`, `ability`, `reward_model.
   ground_truth` as a JSON blob, `extra_info`), prepending the skillbook's
   distilled procedure to each prompt when available.
6. GRPO training via `scripts/train_grpo.sh` (skipped with `--skip-train` or
   `SKIP_TRAIN=1`), producing a new checkpoint that becomes `CURRENT_MODEL`
   for the rest of the cycle and for cycle N+1. **LoRA is the default training
   mode** (`LORA_RANK=16`/`LORA_ALPHA=32`/`LORA_TARGET_MODULES=[q_proj,k_proj,
   v_proj,o_proj]`, set `LORA_RANK=0` for full-parameter fine-tuning) — this
   mirrors `router-skills-evolve`'s HumanEval GRPO algorithm (frozen base +
   trained adapter). verl syncs the adapter into its own internal vLLM
   rollout engine every step via vLLM's dynamic `add_lora`/`remove_lora` API
   (triggered automatically by `lora_rank>0` plus `rollout.load_format=
   safetensors`/`rollout.layered_summon=True`) — no manual `--enable-lora`
   flag is needed for that internal engine. The adapter is written to
   `global_step_N/actor/lora_adapter/` (`adapter_config.json` +
   `adapter_model.safetensors`); `run_full_pipeline.sh` prefers this path over
   the full checkpoint when picking the next cycle's `CURRENT_MODEL`. Because
   verl's `model.path` must always be a loadable full checkpoint,
   `run_full_pipeline.sh` keeps `BASE_SMALL_MODEL` pinned to the original
   model separately from `CURRENT_MODEL`, and forwards a prior cycle's
   adapter dir as `LORA_ADAPTER_PATH` (-> `actor_rollout_ref.model.
   lora_adapter_path`) so verl continues training the same adapter rather
   than reloading it as `model.path` (which would fail — an adapter dir has
   no base weights/config). This continues the same growing adapter across
   cycles, unlike `router-skills-evolve`'s merge-then-fresh-LoRA pattern
   (verl has no built-in merge option).
7. **Checkpoint reload is mandatory for real closed loops**: vLLM does not
   hot-swap a model just because a new model name/path is requested. After
   GRPO, the pipeline calls `SMALL_RELOAD_CMD` (e.g.
   `scripts/reload_small_vllm.sh`, which understands a plain HF checkpoint or
   a LoRA adapter dir) and refuses to silently keep scoring against the old
   server if that variable is unset.
8. Router train/calibration traces are collected *separately* from the GRPO
   train split, using a deterministic task-id hash shard
   (`shard_tasks`/`--task-modulo`/`--task-remainder`, default modulo 5,
   remainders 0 and 1) so the router never trains or calibrates on data the
   policy model was just trained on.
9. `train_router.py` fits a logistic regression classifier predicting
   P(small fails) per prompt over frozen embedding features
   (`verl_code_rl/embedding.py`, default `Qwen/Qwen3-Embedding-0.6B`,
   last-token pooling — deliberately *not* TF-IDF) (`load_binomial_examples`
   expands multi-sample rollouts into a binomial label set, grouped by
   `task_id` for leakage-free CV), then `select_threshold` sweeps thresholds
   on the disjoint calibration shard to pick the cost-minimal threshold
   subject to `--target-pass-rate` (env: `ROUTER_TARGET_PASS_RATE`). Only the
   `LogisticRegression` head is joblib-dumped to `router.joblib`;
   `router_meta.json`'s `embedding_model` field tells every caller
   (`collect_traces.py`, `run_ablation.py`) which embedding model to reload
   for scoring new prompts, so training and inference features always match.
10. A final held-out `eval` split is scored with the current model + the
    freshly calibrated router, and `run_ablation.py` reports routing
    accuracy, F1, and — via `policy_metrics` — the real fallback-inclusive
    cost/pass numbers for the `large`-only, `skills`-only(no-route), `router`,
    and `full` policy variants.

### Cost-aware routing

Router threshold selection and ablation cost numbers are only meaningful in
real currency if you set actual per-model pricing:
`SMALL_INPUT_COST_PER_MILLION`, `SMALL_OUTPUT_COST_PER_MILLION`,
`LARGE_INPUT_COST_PER_MILLION`, `LARGE_OUTPUT_COST_PER_MILLION`,
`SMALL_SAMPLES`, `ROUTER_TARGET_PASS_RATE` — otherwise trace costs default to
0 and `run_ablation`/`train_router` fall back to synthetic per-call costs.

### Mock mode

`SCALING_MOCK=1` / `--mock` (which also implies `--skip-train`) makes
`collect_traces.py` fabricate deterministic pass/fail outcomes from a hash of
`(task_id, model_role, cycle, sample)` instead of calling a model server —
this is the only way to exercise the full pipeline shape without GPUs.

## Notes and gotchas

- Keep GRPO sampling temperature above zero (`0.7`–`1.0` is normal); at
  temperature 0 grouped rollouts collapse to identical samples and GRPO gets
  no advantage signal. `trace_diagnostics.json`'s all-pass/all-fail rates are
  the diagnostic for this.
- `data/raw/he_mbpp.jsonl` is HumanEval + MBPP normalized into one file
  (carried over from the old project); `data/raw/HumanEval.jsonl` and
  `data/raw/mbpp.jsonl` are the untouched per-dataset sources.
- `verl_code_rl/code_eval.py` is deliberately stdlib-only because it's
  imported directly by the verl reward function subprocess/worker path — don't
  add heavy dependencies there.
- `scripts/serve_vllm.sh` defaults `--enforce-eager` on and
  `VLLM_USE_FLASHINFER_SAMPLER=0` for this node: with CUDA graphs enabled, a
  `torch.AcceleratorError: CUDA error: an illegal memory access` reproduced
  twice under concurrent eval load (consistently after ~480-510 requests);
  with the flashinfer sampler enabled, its JIT-compiled kernel targets this
  node's local CUDA toolchain rather than torch's bundled runtime and crashes
  the engine on the first sampling call. Both are opt-out via
  `ENFORCE_EAGER=0` / `VLLM_USE_FLASHINFER_SAMPLER=1` if a fixed vLLM version
  or matching toolchain is available later.
