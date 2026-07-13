# tau2 experiments

## Entry points

The numbered scripts are the public experiment entry points:

1. `01_skills_only.sh`: SkillBook ablation with no model training.
2. `02_sft_only.sh`: multi-cycle SkillBook + verl SFT, without GRPO.
3. `03_sft_grpo.sh`: multi-cycle SkillBook + verl SFT + verl GRPO.

Everything else is an implementation detail:

- `tau2_evolve/`: benchmark adapter, data preparation, SkillBook, and VERL plugins.
- `config/`: VERL multi-turn interaction and agent-loop configuration.
- `scripts/`: the shared cycle runner and low-level VERL launcher.
- `compat/`: optional dependency shims required by the tau2 environment.

For example:

```bash
AGENT_GPU=2 AGENT_PORT=8210 \
USER_GPU=3 USER_PORT=8211 \
TRAIN_GPU=5 N_CYCLES=4 \
bash experiments/tau-2/02_sft_only.sh

AGENT_GPU=2 AGENT_PORT=8210 \
USER_GPU=3 USER_PORT=8211 \
TRAIN_GPU=5 N_CYCLES=4 \
bash experiments/tau-2/03_sft_grpo.sh
```

Set `RESULTS_DIR` to override the timestamped output directory. Resume a run
with `START_CYCLE`, `INITIAL_ADAPTER`, and `INITIAL_SKILLBOOK` rather than a
run-specific resume script.

## Pipeline

The default four-cycle tau2 training pipeline for
`Qwen/Qwen2.5-1.5B-Instruct`:

1. Collect TRAIN trajectories with the current LoRA.
2. Retry failed or golden-action-incomplete trajectories with a stronger
   teacher (`openai/openai/gpt-5.5` through CommonStack by default).
3. Distill a domain skillbook and run verl SFT only on action-complete passes.
4. Run real verl multi-turn GRPO against the tau2 environment. Its training
   reward is an equal blend of the official reward and golden-action recall.
5. Hot-load the exported LoRA into the external vLLM server and evaluate the
   fixed 35-task EVAL split.

`tau2_evolve.prepare_grpo_data` balances airline, retail, and telecom sampling by
default. The held-out evaluator still uses the official tau2 reward; action
recall only shapes training and filters demonstrations.

The pipeline requires `COMMONSTACK_API_KEY` in the environment or `.env`.
CommonStack model IDs retain their provider prefix after LiteLLM routing, so
the teacher model is intentionally written as `openai/openai/gpt-5.5`.

The external tau2-stage2 checkout defaults to
`/shared_home/yuhang.yao/router-skills-evolve`. Override `TAU2_WORKSPACE`,
`TAU2_STAGE2_ROOT`, `TAU2_PARTITION_PATH`, or `TAU2_PYTHON` when using a
different installation.

Compare an evaluation against the SFT-only baseline with:

```bash
PYTHONPATH=experiments/tau-2 venv/bin/python -m tau2_evolve.compare_eval \
  --baseline results/tau2_sft_only/eval_sft_v2.jsonl \
  --candidate results/<run>/cycle_<n>/eval.jsonl
```

The acceptance gate is a higher pass rate and one-sided exact paired McNemar
`p < 0.05` on the same 35 tasks.
