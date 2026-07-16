# tau2 experiments

## Entry points

Use the repository-wide launcher from the repository root:

```bash
scripts/run_experiment.sh --list
scripts/run_experiment.sh tau2 skills
scripts/run_experiment.sh tau2 sft
scripts/run_experiment.sh tau2 grpo
scripts/run_experiment.sh tau2 sft-grpo
```

The numbered scripts are the public experiment entry points:

1. `01_skills_only.sh`: SkillBook ablation with no model training.
2. `02_sft_only.sh`: multi-cycle SkillBook + verl SFT, without GRPO.
3. `03_grpo_only.sh`: strict Qwen3.5 verl GRPO, without SkillBook or SFT.
4. `04_sft_grpo.sh`: multi-cycle SkillBook + verl SFT + verl GRPO.

Everything else is an implementation detail:

- `tau2_evolve/`: benchmark adapter, data preparation, SkillBook, and VERL plugins.
- `config/`: VERL multi-turn interaction and agent-loop configuration.
- `scripts/`: the shared cycle runner and low-level VERL launcher.
- `compat/`: optional dependency shims required by the tau2 environment.

See [`RESULTS.md`](RESULTS.md) for the fixed-split Qwen3.5 results and exact
training configuration used by the latest GRPO-only run.

For example:

```bash
AGENT_GPU=2 AGENT_PORT=8210 \
USER_GPU=3 USER_PORT=8211 \
TRAIN_GPU=5 N_CYCLES=4 \
scripts/run_experiment.sh tau2 sft

AGENT_GPU=2 AGENT_PORT=8210 \
USER_GPU=3 USER_PORT=8211 \
TRAIN_GPU=5 N_CYCLES=4 \
scripts/run_experiment.sh tau2 sft-grpo
```

For Qwen3.5 GRPO-only, first create a strict, domain-interleaved parquet. Do
not pass `--skillbook`:

```bash
PYTHONPATH=experiments/tau-2 .venv_qwen35/bin/python \
  -m tau2_evolve.prepare_grpo_data \
  --traces results/<run>/train_traces.jsonl \
  --output results/<run>/verl_grpo.parquet \
  --user-base-url http://127.0.0.1:8242/v1 \
  --no-user-thinking

TRAIN_FILE=$PWD/results/<run>/verl_grpo.parquet \
TRAIN_GPU=2 ROLLOUT_GPU=0 \
bash experiments/tau-2/03_grpo_only.sh
```

Set `RESULTS_DIR` to override the timestamped output directory. Resume a run
with `START_CYCLE`, `INITIAL_ADAPTER`, and `INITIAL_SKILLBOOK` rather than a
run-specific resume script.

## Pipeline

The legacy four-cycle tau2 training pipeline defaults to
`Qwen/Qwen2.5-1.5B-Instruct`. Model and environment variables can override it:

1. Collect unbiased TRAIN trajectories with the current LoRA.
2. Replay failed student decision prefixes and ask the configured OPD teacher
   for a continuation. Only officially passing, action-complete suffixes are
   retained, and SFT loss is masked off for the preceding student turns.
3. Distill a domain skillbook and run verl SFT only on verified trajectories.
4. Run real verl multi-turn GRPO against the tau2 environment. Its training
   reward is an equal blend of the official reward and golden-action recall.
5. Collect fresh post-training trajectories, train a lightweight escalation
   router on student successes and verified teacher rescues, then evaluate
   both the policy and routed system on the fixed 35-task EVAL split.

`tau2_evolve.prepare_grpo_data` balances and interleaves airline, retail, and
telecom sampling by default. The held-out evaluator still uses the official
tau2 reward; action recall only shapes training and filters demonstrations.

Skillbook distillation and OPD correction use GPT-5.5 through CommonStack by
default. The student and user simulator remain local. Override the
`DISTILLER_*` and `OPD_TEACHER_*` variables to use other endpoints.

The external tau2-stage2 checkout defaults to a sibling directory named
`router-skills-evolve` next to this repository. Override `TAU2_WORKSPACE`,
`TAU2_STAGE2_ROOT`, `TAU2_PARTITION_PATH`, `TAU2_SITE_PACKAGES`, or
`TAU2_PYTHON` when using a different installation. See the TAU-2 setup and
resume sections in the root `CLAUDE.md` for a complete cross-machine runbook.

Qwen3.5 uses an isolated environment described by
`requirements-qwen35-cu129.txt`. Prefix separated VERL runs with
`compat/qwen35_torch_fallback` as `03_grpo_only.sh` does; it gives training
workers the Triton 3.3 + FLA runtime while rollout remains on Triton 3.6, and
hides the stale NIXL-EP module without disabling the NIXL checkpoint engine.
Adapters remain in vLLM's text-only key layout on disk. Before continued SFT
or GRPO, the training scripts create a local `training_init_adapter` using
VERL's `language_model` key layout so PEFT actually restores every LoRA key.

Compare an evaluation against the SFT-only baseline with:

```bash
PYTHONPATH=experiments/tau-2 venv/bin/python -m tau2_evolve.compare_eval \
  --baseline results/tau2_sft_only/eval_sft_v2.jsonl \
  --candidate results/<run>/cycle_<n>/eval.jsonl
```

The acceptance gate is a higher pass rate and one-sided exact paired McNemar
`p < 0.05` on the same 35 tasks.
