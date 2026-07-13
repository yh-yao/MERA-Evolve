# tau2 experiments

Four-cycle tau2 training pipeline for `Qwen/Qwen2.5-1.5B-Instruct`:

1. Collect TRAIN trajectories with the current LoRA.
2. Retry failed or golden-action-incomplete trajectories with a stronger
   teacher (`openai/openai/gpt-5.5` through CommonStack by default).
3. Distill a domain skillbook and run verl SFT only on action-complete passes.
4. Run real verl multi-turn GRPO against the tau2 environment. Its training
   reward is an equal blend of the official reward and golden-action recall.
5. Hot-load the exported LoRA into the external vLLM server and evaluate the
   fixed 35-task EVAL split.

`prepare_verl_grpo_data.py` balances airline, retail, and telecom sampling by
default. The held-out evaluator still uses the official tau2 reward; action
recall only shapes training and filters demonstrations.

The pipeline requires `COMMONSTACK_API_KEY` in the environment or `.env`.
CommonStack model IDs retain their provider prefix after LiteLLM routing, so
the teacher model is intentionally written as `openai/openai/gpt-5.5`.

Compare an evaluation against the SFT-only baseline with:

```bash
venv/bin/python experiments/tau-2/compare_eval.py \
  --baseline results/tau2_sft_only/eval_sft_v2.jsonl \
  --candidate results/<run>/cycle_<n>/eval.jsonl
```

The acceptance gate is a higher pass rate and one-sided exact paired McNemar
`p < 0.05` on the same 35 tasks.
