# TAU-2: GRPO experiment plan for the 4B student + AReaL RL tasks

Starting 2026-10-28. Follows from the S2 conclusion in `PLAN_4B_A800.md`:

- After LoRA SFT of 4B on the AReaL SFT dialogues (1,983 segments, Claude-generated), VAL dropped 34.0% → 25.4% / 26.5%
  (a significant drop at both learning rates), and frontier 58.2% → 47% / 48%. There was no collapse (repeated calls ~0); instead it "does less":
  tool calls per trajectory 7.6 → 5.3, messages 38 → 31, "talk + call a tool in the same turn" 7.2 → 1.1.
  Assessment: it learned the style of the synthetic data, while the synthetic tasks/databases do not match the official distribution.
- 2B also dropped with the same data (17.2% / 14.6%; repeated-call collapse at 3e-5).

So this plan **drops SFT** and moves to AReaL's second stage: GRPO on the student's own rollouts, with reward from the official tau2 judgment only.
The RL update direction comes from "which rollout succeeded", so it does not pull the student toward the teacher's speaking style.

## Fixed conventions

- The split `partition_v2.json` is unchanged; VAL/EVAL are evaluation only, never trained on; EVAL is reported only at the end.
- 4 telecom tasks in the AReaL RL set have the same fault combinations as VAL/EVAL (exact match on `initial_state`); they are removed.
- Evaluation uses merged full checkpoints (the LoRA covers the GDN projections, and vLLM's LoRA path is incomplete).
- Training-time user simulator: local Qwen3.5-4B (same as all baselines measured so far).

## Reference

| Model | VAL | frontier (58 tasks × 8) |
|---|---|---|
| 4B base | 34.0% | 58.2% |
| 27B answering throughout | 31.7% | 59.3% |

## Steps

### R0 Environment: run AReaL synthetic tasks (about 3–4 hours)

What to do:
- `benchmark.load_task` supports task ids of the form `areal:<id>`, reading from `tau2_rl_train.jsonl` and converting to a tau2 Task.
- Each task's `db_path` database must take effect **at both run and evaluation time**: replace the tau2 registry's default constructor
  with an environment constructor selected by call context (contextvar), so concurrent rollouts do not leak into each other's databases.
- Unit tests: running one query tool on the same task with the default database and with the specified database should give different results; replaying
  the reference actions in `evaluation_criteria` for an AReaL task should give reward 1.

If not met:
- Reference-action replay reward is not 1: first check that the database is loaded correctly and that telecom's `initial_state` is executed;
  tasks that still fail (AReaL incompatible with our tau2 version) are removed, and the fraction is recorded. If the fraction exceeds 20%, use only airline/retail for now.

### R1 Task filtering with 4B base (about 2–3 hours, 4 GPU groups in parallel)

What to do: run 4B base on the AReaL tasks 4 times per task (temperature 1.0, same as the frontier diagnostic) to get each task's success rate p and opening user message;
also reuse the existing 4B results for the 139 tasks in our TRAIN.

Expected: enough tasks with 0 < p < 1 (target ≥ 300 tasks).

If not met:
- Too few tasks that are sometimes right, sometimes wrong: raise sampling to 8 per task; or relax to keep all tasks with p ≤ 0.75.
- The vast majority are always solved: these synthetic tasks are too easy for 4B; keep only airline (the harder part of AReaL) and telecom.

### R2 4B GRPO smoke test (about 1–2 hours)

What to do: run 2–3 steps on the 5090s to confirm memory. Starting point: LoRA on all linear layers, batch 4 × 8 generations, response limit 8k,
total length 16k, gradient checkpointing, fused kernels (settings validated on 2B).

If not met (OOM): try in order batch 2, response limit 6k, per-GPU micro batch 1, parameter offload to CPU;
if it still fails, train LoRA on attention and MLP only (excluding the GDN projections).

### R3 Full GRPO (about 15–25 hours)

Two groups in parallel (2 × 4 GPUs on 002), about 150 steps each, lr 1e-5, checkpoint every 25 steps:
- G1: AReaL sometimes-right/sometimes-wrong tasks + our TRAIN frontier tasks (mixed).
- G2: our TRAIN frontier tasks only (control: whether AReaL tasks add extra benefit).

Monitor: training reward, within-group reward variance, repeated calls, tool calls per trajectory.

If not met:
- Reward does not rise and the model barely moves (KL tiny): raise lr to 3e-5, or double the tasks per step.
- Reward rises but frontier/VAL drops: overfitting to the synthetic tasks; raise the TRAIN task fraction, or fall back to an earlier checkpoint.
- Repeated-call collapse: roll back to the previous checkpoint and halve the lr.

### R4 Evaluation (interleaved during training, about 2 hours at the end)

Each checkpoint is first tested on frontier (sensitive). Bar: ≥ 5 points above 4B base with 95% CI lower bound > 0;
those that pass the bar are then tested on VAL (seed 300; the best one also gets 1300).

### Parallel: CUDA 12 forward-compatibility test on the A800

Download CUDA 12 builds of torch/vLLM and the compatibility libraries on the data platform and test them on the A800. If it works, subsequent RL and evaluation can all move to the A800 (8 × 80 GB).

## Timeline (estimated)

| Time | Content |
|---|---|
| 10/28 morning | R0 environment and tests |
| 10/28 afternoon | R1 filtering; R2 smoke test |
| 10/28 evening – 10/29 | R3 training, interleaved with R4 |
| 10/29 evening – 10/30 | Conclusions and next steps |
