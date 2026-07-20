# Reproducible experiment recipes

All public recipes are dispatched through one command from the repository
root:

```bash
scripts/run_experiment.sh --list
scripts/run_experiment.sh humaneval_mbpp skills
scripts/run_experiment.sh tau2 4cycle
```

The numbered files below are stable recipe implementations. They remain
directly executable for compatibility, but runbooks and schedulers should use
`scripts/run_experiment.sh` so benchmark and recipe names are explicit.

One subdirectory per task domain, so domain-specific tuning (learning rate,
batch sizes, eager-mode requirements, etc.) never gets silently mixed up
across domains. Public entry points are numbered; domain-specific support
code and VERL configuration stay inside that domain's directory.

- `humaneval_mbpp/` — HumanEval + MBPP code generation
  (`data/raw/he_mbpp.jsonl`).
- `tau-2/` — tau2-bench SkillBook, SFT, and verl GRPO experiments. Its
  numbered public entry points are documented in `tau-2/README.md`.

## humaneval_mbpp/

Each script isolates one piece of the evolve loop so results are
reproducible independently of each other. All of them assume small + large
OpenAI-compatible vLLM servers are already running (`scripts/serve_vllm.sh`)
and that `.env` has `COMMONSTACK_API_KEY` set if `DISTILLER_MODEL` is used.
Write-ups of past runs of these scripts live in `docs/experiments/`.

| Script | Isolates | Training | Skill text |
|---|---|---|---|
| `01_skills_only.sh` | SkillBook alone | none | yes |
| `02_sft_only.sh` | SFT alone | SFT (LoRA) | no |
| `03_grpo_only.sh` | GRPO alone | GRPO (LoRA) | no |
| `04_4cycle_sft.sh` | Skills+SFT compounding over cycles | SFT (LoRA), 4 cycles | yes |
| `05_4cycle_sft_grpo.sh` | Skills+SFT+GRPO compounding over cycles | SFT+GRPO (LoRA), 4 cycles | yes |

`grpo_qwen25_1p5b.env` / `sft_qwen25_1p5b.env` in that same directory pin the
standalone-training entry points' (`scripts/train_grpo.sh` /
`scripts/train_sft.sh`) settings for this domain, for use outside these
experiment scripts (see `CLAUDE.md`'s "Standalone GRPO training" section).
The experiment scripts below don't source them -- the same values are baked
in directly as their own env-var defaults.

### Quick start

```bash
# Start the small/large model servers first (adjust GPU/port to taste)
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct PORT=8000 GPU=0 scripts/serve_vllm.sh
MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct  PORT=8001 GPU=1 scripts/serve_vllm.sh

# Single-mechanism isolations (01-03) each need one more free GPU for training
# (02/03 only; 01 does no training at all)
scripts/run_experiment.sh humaneval_mbpp skills
SFT_GPU=2 scripts/run_experiment.sh humaneval_mbpp sft
TRAIN_GPU=2 scripts/run_experiment.sh humaneval_mbpp grpo

# Full closed loops (04-05) need one more free GPU for training, plus a
# second small-model GPU for SMALL_RELOAD_CMD to reload onto after each
# cycle's training step (SMALL_RELOAD_GPU) -- see each script's header.
SFT_GPU=2 SMALL_RELOAD_GPU=0 scripts/run_experiment.sh humaneval_mbpp 4cycle-sft
GRPO_GPU=2 SMALL_RELOAD_GPU=0 scripts/run_experiment.sh humaneval_mbpp 4cycle-sft-grpo
```

Every env var each script reads has a documented default in its header;
override any of them inline, e.g. `TRAIN_LIMIT=64 EVAL_LIMIT=100
bash experiments/humaneval_mbpp/03_grpo_only.sh` for a fast smoke run.

## Why these specific defaults

- **`LR=5e-5` for GRPO**: an LR sweep (1e-5 / 2e-5+compile-bonus / 5e-5)
  found the project's old default of `1e-6` left GRPO's policy almost
  stationary after 64 steps (KL-divergence from the reference policy stayed
  under 0.2 the whole run, vs 0.28-0.36 for the higher LRs) -- `5e-5` was the
  strongest and still stable of the three tested.
- **`ENFORCE_EAGER=True` everywhere**: this node's vLLM build reproduces a
  CUDA-graph-only crash (`illegal instruction` / `illegal memory access` /
  `unspecified launch failure`, all the same underlying instability) and,
  short of crashing outright, gives numerically *unreliable* results under
  CUDA graphs -- confirmed by reproducing a large internal-vs-external eval
  gap that closed to ~1-2pts once both sides ran in eager mode. Eager mode is
  the only setting confirmed reproducible on this hardware; trust eager-mode
  numbers over anything computed with CUDA graphs enabled.
- **Bigger GRPO micro-batch (`PPO_MICRO_BATCH_SIZE_PER_GPU=32`,
  `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=64`, `PPO_MINI_BATCH_SIZE=32`)**: the
  smaller defaults (8/16/16) left GPU memory at ~20% utilization and MFU
  (Model FLOPs Utilization) at ~8%. The bigger micro-batch raised MFU to
  ~29% and cut `timing_s/update_actor` from ~36s to ~10s per step, with
  memory still comfortably under budget (~34GB/80GB observed).
- **`--probe-only` for oracle collection**: the small model is tried first
  on every task; the large model is only invoked as a fallback when the
  small model's execution fails. This is what makes the reported cost
  numbers meaningful (a router-gated cascade, not "run both and see who
  wins") -- see `CLAUDE.md`'s "One evolve cycle" section for the full
  rationale.

## tau-2/

Use `06_4cycle.sh` through the common launcher for the validated full run:

```bash
AGENT_GPU=0 AGENT_PORT=8260 \
USER_GPU=1 USER_PORT=8261 \
TRAIN_GPU=2 ROLLOUT_GPU=3 \
  scripts/run_experiment.sh tau2 4cycle
```

This recipe starts its own Qwen3.5-2B student and Qwen3.5-4B user/teacher
servers. Each cycle collects current-student trajectories, replays failed
prefixes with the local teacher, trains SFT only on verified continuations,
runs VERL GRPO, trains the escalation router, and evaluates the fixed held-out
split. `02_sft_only.sh`, `03_grpo_only.sh`, and `04_sft_grpo.sh` remain the
lower-level ablation/custom-run entry points. See `tau-2/README.md` and
`CLAUDE.md` for environment setup and resume commands.
