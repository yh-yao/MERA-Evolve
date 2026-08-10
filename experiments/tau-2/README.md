# tau2 experiments

## Entry points

Use the repository-wide launcher from the repository root:

```bash
scripts/run_experiment.sh --list
scripts/run_experiment.sh tau2 skills
scripts/run_experiment.sh tau2 sft
scripts/run_experiment.sh tau2 grpo
scripts/run_experiment.sh tau2 sft-grpo
scripts/run_experiment.sh tau2 4cycle
scripts/run_experiment.sh tau2 4b-baseline
```

The numbered scripts are the public experiment entry points:

1. `01_skills_only.sh`: SkillBook ablation with no model training.
2. `02_sft_only.sh`: multi-cycle SkillBook + verl SFT, without GRPO.
3. `03_grpo_only.sh`: strict Qwen3.5 verl GRPO, without SkillBook or SFT.
4. `04_sft_grpo.sh`: generic multi-cycle SkillBook + verl SFT + verl GRPO.
5. `05_fallback_4b_smoke.sh`: one-task Qwen3.5-4B runtime smoke test.
6. `06_4cycle.sh`: recommended Qwen3.5 OPD + SkillBook + SFT + GRPO + router
   four-cycle reproduction.
7. `07_4b_baseline_eval.sh`: endpoint-only Qwen3.5-4B evaluation on the fixed
   35-task split, without skills, adapters, routing, or fallback.

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

AGENT_GPU=0 AGENT_PORT=8260 \
USER_GPU=1 USER_PORT=8261 \
TRAIN_GPU=2 ROLLOUT_GPU=3 \
scripts/run_experiment.sh tau2 4cycle

GPU=2 USER_GPU=3 PORT=8280 USER_PORT=8281 \
scripts/run_experiment.sh tau2 4b-baseline
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

The recommended four-cycle pipeline uses `Qwen/Qwen3.5-2B` as the student and
`Qwen/Qwen3.5-4B` as the local user simulator, OPD teacher, distiller, and
routed fallback. Model and endpoint variables remain overridable:

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

SFT keeps the first current-cycle trajectory for each `(domain, task_id)` and
does not oversample domains by default. This avoids repeatedly weighting the
same small set of OPD repairs across cycles. The recommended Qwen3.5 recipe
uses `SFT_LR=1e-7` for one epoch; set `SFT_BALANCE_DOMAINS=1` only for an
explicit oversampling ablation.

Agent and user-simulator requests explicitly use `temperature=0.0`, matching
the tau2 evaluation default instead of relying on backend-specific API
defaults. verl GRPO rollout sampling remains stochastic and is configured
separately by the GRPO training recipe.

SkillBook distillation and OPD correction use the local 4B endpoint by
default, so the full recipe does not require a hosted-model key. Override
`DISTILLER_*` and `OPD_TEACHER_*` to use other endpoints.

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
