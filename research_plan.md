# MERA Research Plan: TAU-2 and BFCL-V4 with a 4B Student

This document is for the people taking over these experiments with more compute. After reading it you should be able to
reproduce the existing results on your own and run the key experiments of the next phase.
General repository instructions (environments, launchers, invariants) are in `CLAUDE.md`.
Sections 0–11 cover the TAU-2 handoff; Section 12 adds BFCL-V4 as the second tool-use benchmark.
The BFCL adapter and experiments are planned work, not existing code or measured results.

Branch: `tau2-4b-rl-handoff` (based on `tau2-phase0-1-opd-27b`; contains the existing TAU-2 code and experiment scripts).

**Target configuration:** one node with **8 × A100 40 GB**; **Qwen3.5-4B student + frozen Qwen3.5-9B teacher**
for both benchmarks. TAU-2 also uses a separate frozen Qwen3.5-4B user simulator. The 9B teacher is a candidate
whose advantage must be measured, not an established stronger model. Deployment and training settings below are
starting points pending smoke tests on this node; historical results remain unchanged.

---

## 0. Status in one paragraph

On tau2-bench (airline / retail / telecom customer-service tasks), **switching the student from Qwen3.5-2B to Qwen3.5-4B is the only
significant improvement so far (VAL +10.4 pt)**. Later attempts to push further with a teacher, extra data for SFT, or routing had no
significant effect. The latest 4B GRPO ran 250 steps on 32 GB GPUs; on the frontier training tasks it mostly moved within noise.
G2, trained only on TRAIN tasks, had two checkpoints pass the bar (+5.4 / +5.6 pt), but the passing step 200 scored −2.2 pt on VAL
(not significant): the gain did not transfer (see 1.4).
**The main suspect is the length cap forced by GPU memory: about 30% of training conversations were truncated and scored 0.**
One component of the next phase is to remove truncation and enlarge the effective batch using multi-GPU training
and infra optimization on the 8 × A100 40 GB node, then re-test GRPO.
Training infrastructure is also an explicit optimization target: profile rollout, user simulation, actor updates and weight
sync, then improve memory use and end-to-end throughput with the same model checkpoints and evaluation protocol (Section I).
The whole workflow is a one-command script, `experiments/tau-2/10_4b_rl_pipeline.sh` (Section 5.0).
A second open problem is **teacher headroom**: our local 27B teacher is no better than the 4B student on VAL. The next
phase therefore **qualifies Qwen3.5-9B as the local teacher, starting with single-GPU BF16 serving** (E6, no API
budget), next to GRPO without a teacher (E2). Parameter count alone does not establish headroom. Teacher-free fallbacks are the
action-recall process reward (E4) and gold-action-guided self-rollouts (E5).

**The main TAU-2 experiment is the full MERA loop with a cost–quality curve (Section 5, M)**, not a distillation result:
the 4B student serves every turn behind a learned router that escalates to the strong teacher, and each cycle improves
the student (teacher corrections + GRPO), the SkillBook and the router. Cost is the **number of tokens the teacher
generates**. The claim to test is the paper's claim: improving the student moves the curve — the same pass rate for
fewer teacher tokens, more so with every cycle — instead of only routing around a fixed student.
E0–E6 are prerequisites and components of that loop.

---

## 1. Background and results so far

### 1.1 Evaluation protocol (Phase 0, fixed — do not change)

- Split: `experiments/tau-2/data/partition_v2.json`, TRAIN 139 / VAL 67 / EVAL 67 tasks. Model selection and tuning use only
  TRAIN-derived sets and VAL; EVAL is reported once, at the end.
- 4 trials per task (user-simulator temperature 0.7, a different seed per trial); report pass^1 and pass^4 with paired
  bootstrap 95% CIs and McNemar tests (`tau2_evolve/eval_stats.py`).
- Minimum detectable difference: about **10 pt** for a single VAL run (seed 300), about **7 pt** for seeds 300 + 1300 combined (8 trials).
- **Frontier training tasks**: 58 TRAIN tasks that the 2B base sometimes solves and sometimes fails across 8 samples;
  8 trials per task, agent temperature 1.0, user temperature 0, at most 40 steps.
  Much more sensitive than VAL (about 5 pt is visible); used to tell whether training learned anything at all. The 58 tasks are fixed in
  `experiments/tau-2/data/partition_frontier_only.json` (retail 30 / airline 22 / telecom 6, in the TRAIN bucket);
  point `TAU2_PARTITION_PATH` at it for evaluation. They come from `tau2_evolve/frontier.py` bucketing 8 samples of the 2B base.
- User simulator for training, TRAIN diagnostics, VAL and final EVAL: the same local **Qwen3.5-4B base checkpoint**, frozen
  throughout all cycles and ablations. Never attach the student's trained adapter or SkillBook to the user simulator.
  Pin its model/tokenizer revision, chat template, thinking setting and serving configuration. Keep the stage-specific
  decoding protocol above (VAL/EVAL user temperature 0.7; frontier/R1 user temperature 0); compare policies under the
  same protocol and seeds. Final EVAL does not switch to a hosted model.

### 1.2 Reference numbers

| Model | VAL pass^1 | Frontier (58 tasks × 8) | TRAIN greedy (139 tasks × 1) |
|---|---|---|---|
| Qwen3.5-2B | 23.5% | 45.0% | 31.2% |
| **Qwen3.5-4B** | **34.0%** (vs 2B +10.4 [+4.1, +16.8]) | **58.2%** | 44.9% (airline 81.8 / retail 66.7 / telecom 1.9) |
| Qwen3.8-27B (answers every turn) | 31.7% | 59.3% | — |

Takeaways: 4B already matches the 27B teacher, so "teach 4B with 27B" has no measurable headroom; telecom is near 0 (2–20%) for every local model.

### 1.3 Directions tried without success (do not repeat)

| Direction | Result | Conclusion |
|---|---|---|
| 2B: GRPO, full-token distillation (strong/weak), error-point correction SFT, etc. — 13 configurations | None significantly above base; strong distillation caused repeated-call collapse | 2B struggles to absorb the teacher's ability |
| Error-point correction SFT (teach the teacher's action at the student's first mistake) | Fixes the targeted decisions (4/45 → 34/45), but VAL goes down, not up | A failed trajectory usually has more than one mistake; fixing one is not enough |
| Per-turn router (hand turns to 27B at different points) | −3 to +5 pt at 10–30% teacher turns, none significant; the "teacher on disagreement" oracle only +1.5 | Per-action substitution does not recover the teacher's advantage |
| 4B LoRA SFT on the 1,983 SFT dialogues of AReaL-tau2-data | VAL 34.0 → 25.4 / 26.5 (two learning rates, both significantly worse) | It learned the style of the synthetic data (Claude-generated, synthetic DBs) and "does less" |
| 4B GRPO (see 1.4) | Frontier moves between 58–62%, no trend | See below |

### 1.4 Most recent: 4B GRPO (32 GB RTX 5090, LoRA)

- Data: the 1,844 AReaL RL tasks whose gold actions replay correctly (airline 1148 / retail 563 / telecom 133); 4 samples per task
  from the 4B base, keeping the 546 tasks with 0 < p < 1; plus the 36 TRAIN tasks with 0 < p < 1.
- Two groups: G1 = 546 + 36, G2 = only the 36 TRAIN tasks (control). Reward is the official tau2 verdict only (group `O`).
- Settings: LoRA rank 16 (attention + MLP), 4 prompts × 8 generations per step, lr 1e-5 for the first 100 steps, then another
  150 steps at lr 3e-5 from the step-100 adapter; prompt cap 8192, **response cap (the whole multi-turn conversation) 5632–6144 tokens**, at most 30 turns.
- Frontier results (4B base 58.2%):

| Checkpoint | G1 | G2 |
|---|---|---|
| 50 (lr 1e-5) | 60.8% (+2.6 [−2.6, +7.8]) | 61.4% (+3.2 [−2.4, +9.1]) |
| 100 (lr 1e-5) | 58.0% | **63.6% (+5.4 [+0.4, +10.6], passes)** |
| 150 (lr 3e-5) | 61.6% (+3.4 [−2.4, +9.3]) | 59.9% |
| 200 (lr 3e-5) | 58.2% | **63.8% (+5.6 [+0.4, +11.2], passes)** |
| 250 (lr 3e-5) | 59.9% (+1.7) | 62.7% (+4.5 [−1.9, +11.2]) |

  (CIs are per-task paired bootstrap against the 4B base, `scripts/compare_tasks.py`.)
  - G2 stays slightly higher on the frontier (+4.5 to +5.6); G1 barely changes.
  - **But G2 step 200 on VAL: 34.0% → 31.7% (−2.2 [−7.8, +3.4]), not significant and in the wrong direction.**
    27 of G2's 36 training tasks (TRAIN tasks with 0 < p < 1) are among the 58 frontier evaluation tasks, so the frontier gain most likely
    means "learned the training tasks", not generalization.
    Therefore the frontier is only a signal of whether training learned anything; beyond being the gate for VAL, **final conclusions must
    come from VAL**. E2/E3 should consider evaluating the frontier on TRAIN tasks not used for training, or on an AReaL held-out set.

- Training reward stayed at 0.2–0.35 for all 250 steps; PPO clip fraction about 0.1–0.2%; behavior healthy (0 repeated calls,
  same number of tool calls as base).
- **Diagnosis**: `response_length/clip_ratio` shows **24–40% of training conversations truncated by the length cap**, scored 0.
  In R1 sampling (no length cap) 4B solves these tasks about half the time, yet training reward is only 0.25–0.3; the gap is mostly
  "false failures" from truncation and the turn cap. In addition, only 4 prompts per step gives high gradient variance.
  Both are consequences of 32 GB of GPU memory.

---

## 2. Questions for the next phase

0. **Q0 (main question)**: on TAU-2, does the MERA loop move the cost–quality curve? At the same pass^1, does the routed
   system (4B student + learned router + strong teacher) need fewer teacher-generated tokens after each cycle than with
   the fixed base student (cycle 0 = routing around a fixed student, the prior-work setting)? Q1–Q3 and Q5 below are
   about the components of the loop; Q4 is the final report.
1. **Q1 (student update without a teacher)**: with truncation removed and a larger batch, does GRPO significantly improve 4B on the frontier
   (≥ +5 pt with CI lower bound > 0), and does it transfer to VAL?
2. **Q2**: do the AReaL synthetic tasks (G1) add anything over the official TRAIN tasks alone (G2)?
3. **Q3**: if Q1 is no, is the problem credit assignment (long conversations, terminal reward only)? Test with PPO + critic or per-turn advantage estimates.
4. **Q4**: report the 4B base and the best model on EVAL with the same frozen local Qwen3.5-4B user simulator as VAL.
5. **Q5 (teacher headroom)**: if the local Qwen3.5-9B teacher proves stronger, does it teach the 4B student what GRPO
   alone cannot — especially the tasks it never solves (p = 0, so GRPO gets no signal) — when its corrections and
   demonstrations are generated in the real tau2 environment (E6)? Teacher-free fallback: the tasks' own gold actions (E5). Earlier attempts failed for two separate reasons: the local 27B teacher was no
   stronger than 4B, and AReaL's Claude-written SFT dialogues were strong but off-distribution (synthetic DBs and style).

---

## 3. Hardware and environment

- **To build the environments and download models and data from scratch, follow `experiments/tau-2/SETUP_4B_RL.md`**
  (two scripts, `scripts/setup_envs.sh` and `scripts/download_assets.sh`; official sources by default; `MIRROR=cn` and `SOURCE=modelscope` for machines in China; optional S3 cache on AWS).
- Target machine: **one node with 8 × A100 40 GB, NVIDIA driver ≥ 580**. Driver ≥ 580 is required: the vLLM 0.29 used for evaluation ships CUDA 13
  torch; the training environment is cu128 and needs ≥ 570. Below 535, no vLLM that supports Qwen3.5 starts, so GRPO cannot run (SFT can).
- The training environment (VERL 0.8 + bundled vLLM 0.18, `.venv_qwen35`) and the serving environment (vLLM 0.29, for evaluation)
  are built by `setup_envs.sh` (background in the comments of `requirements-qwen35-cu129.txt`); tau2 runs in its own venv (`TAU2_WORKSPACE`).
- Qwen3.5's GDN layers **must use the fast path**: this needs `flash-linear-attention` and `causal-conv1d`. Without `causal-conv1d`,
  transformers silently falls back to a pure-torch implementation (training 2.5× slower or worse); search the logs for "fast path is not available".
- Models: **`Qwen/Qwen3.5-4B`** (student starting checkpoint and separate frozen user simulator),
  **`Qwen/Qwen3.5-9B`** (frozen candidate teacher, subject to E6 step 0);
  optionally Qwen3.8-27B as the earlier teacher reference. Infra comparisons keep these checkpoints fixed; changing
  model size or swapping the user simulator is a separate experiment.
- The current setup runbook and downloader contain earlier hardware/teacher references. Keep their environment
  instructions, but use the model identities and GPU layouts in this plan. Add a pinned 9B download target when
  implementing E6: `download_assets.sh teacher` currently downloads 27B and `sea` downloads 235B; neither selects 9B.
  The pipeline/example config now default to bf16; record `ACTOR_MODEL_DTYPE=bf16` explicitly in the run config.

---

## 4. Data preparation (about 1 hour, mostly CPU; the replay check alone takes ~40 minutes)

1. tau2-bench is pinned at commit `17e07b1da2bbc0cadfddeea36412686e0604127b` (see `CLAUDE.md`).
2. AReaL data: `inclusionAI/AReaL-tau2-data` (Apache-2.0, on both Hugging Face and ModelScope); download it with
   `scripts/download_assets.sh areal`. Point `MERA_AREAL_DIR` at the download directory.
3. Drop tasks that cannot be replayed (about 45% of telecom tasks do not reach their asserted state under our pinned tau2 version):
   ```bash
   python experiments/tau-2/scripts/check_areal_tasks.py --write-valid $MERA_AREAL_DIR/valid_task_ids.json
   ```
   `tau2_evolve/areal_tasks.py` reads this file automatically and also excludes 4 telecom tasks whose fault combinations match VAL/EVAL tasks.
   Expected: airline 1148/1148, retail 563/563, telecom about 133/267 pass. (The pipeline's `check` stage runs this for you.)
4. Each AReaL task loads its own database snapshot (tau2's environment constructor is replaced inside a contextvar scope, so it is
   concurrency-safe), for both rollouts and evaluation; `collect_traces --bucket AREAL` samples them.

---

## 5. Experiments (in priority order)

### M. Main experiment: the full MERA loop on TAU-2 with a cost–quality curve (answers Q0)

**System.** The 4B student (Qwen3.5-4B + LoRA adapter + SkillBook in the prompt) answers every turn. A learned router
(`tau2_evolve/router.py`, logistic regression on conversation-state features) decides when to escalate to the teacher,
Qwen3.5-9B served locally and qualified in E6. After an escalation the teacher finishes the conversation (`RoutedAgent`).
At replay the official tau2 evaluator is the verifier.

**Cost = teacher-generated tokens.** For each conversation, the sum of `completion_tokens` over all teacher calls,
**including reasoning tokens** (the teacher runs in thinking mode and vLLM counts them as completion tokens).
Reported as the mean per conversation and normalized as
`cost% = teacher tokens of the policy / teacher tokens of always-teacher` on the same tasks, trials and seeds.
Student tokens and user-simulator tokens are logged but not counted (the student is the fixed local base cost; the user
simulator is part of the environment).

**One cycle** (c = 0 … K, K = 3 → four cycles as in the code-generation study):
1. *Collect*: current student (adapter_c, SkillBook_c) on TRAIN + valid AReaL tasks, no teacher.
2. *Teacher corrections*: verified error-point corrections on the failed rollouts (E6 step 1); only fixes that the
   official evaluator passes are kept.
3. *SkillBook update* (`tau2_evolve/build_skillbook.py`) from successful traces; it is prepended to the student prompt.
4. *Student update*: SFT on corrections (+ rehearsal of the student's own successes), then GRPO with the E2 settings
   → adapter_{c+1} (E6 step 2, group G).
5. *Router update*: student-only rollouts of the new student on TRAIN give escalate/continue labels
   (`tau2_evolve/train_router.py`); the router is fit last so its labels reflect the current student.
6. *Admission* (joint replay on a TRAIN shard, never on VAL/EVAL): the new adapter + SkillBook + router are admitted
   only if routed pass^1 is not lower than the previous cycle's at no more teacher tokens. Otherwise the previous state
   carries over.
7. *Measure* the curve on VAL (below).

**The curve** (every cycle; VAL, seeds 300 + 1300, 67 tasks × 4 trials each):
- Endpoints: always-student (0 teacher tokens) and always-teacher (cost% = 100).
- Learned router: thresholds {0.3, 0.4, 0.5, 0.6, 0.7, 0.8} trace the curve. The headline operating point is chosen
  on a TRAIN calibration shard, not on VAL.
- Baselines at matched cost: per-turn `random:P` (cost-blind, P ∈ {0.1, 0.3, 0.5}) and `tool_error` from
  `tau2_evolve/turn_router.py`.
- Headline numbers, with bootstrap CIs over tasks:
  (a) **cost at matched quality** — the teacher-token % needed to reach 95% of always-teacher pass^1 (interpolated);
  (b) **pass^1 at a fixed budget** — e.g. 30% of always-teacher tokens.
- Cycle 0 (base student, routing only) is the "route around a fixed student" baseline. The main result compares the
  curves of cycles 1 … K with cycle 0.

**Ablations** (each a loop run; two cycles are enough if compute is short):
- routing only (no student update) — equals cycle 0 repeated with a re-fit router;
- no SkillBook;
- student update by GRPO only (E2, no teacher data) vs teacher corrections + GRPO (E6);
- one cycle vs K cycles;
- learned router vs `random:P` at equal cost.

**Pass rules.**
- Primary: at cycle K, cost at matched quality is lower than at cycle 0 and the bootstrap CI of the difference
  excludes 0; at the headline operating point, routed pass^1 is within the CI of always-teacher.
- Secondary: always-student pass^1 rises across cycles (the E2/E6 rules) — it explains *why* the curve moves.
- If cycle 1 does not move the curve, stop the loop and fall back to the component experiments (E4/E5) before spending
  more cycles.

**Prerequisites.** E0 (baseline), E6 step 0 (the teacher must be measurably stronger than the student — with the 27B teacher
always-teacher ≈ always-student, so there was no curve to move; this is why our earlier per-turn router results were
inconclusive), E1 (R1 rollouts, reused as cycle-0 collection).

**New code needed (not written yet).**
1. Token accounting: record `completion_tokens` (with reasoning) and the source (student / teacher) of every assistant
   message in `RoutedAgent` and `TurnRoutedAgent`, and add `teacher_completion_tokens` / `student_completion_tokens`
   to each trace row. Check first whether tau2's message objects already carry `usage` from LiteLLM.
2. `scripts/cost_quality_curve.py`: routed eval files → curve points with bootstrap CIs, cost at matched quality,
   pass at fixed budget, plot.
3. Loop driver: `scripts/run_cycle_pipeline.sh` (the `4cycle` recipe) still uses the old 97/35 split, the 2B student,
   and runs its routed eval on the 35 EVAL tasks. Port it to partition_v2, the 4B student and the local teacher; run the
   curve on VAL; add the TRAIN-shard admission step and explicit GPU profiles for 9B teacher serving versus student
   training. Teacher shutdown can release a GPU, but is no longer forced by a node-wide teacher allocation.

### 5.0 One-command workflow: `10_4b_rl_pipeline.sh`

E0–E2 are chained in one script. All settings live in one config file; stages run in order and each writes `.done_<stage>` to
`RUN_DIR` when it finishes, so after an interruption re-running with the same config resumes where it stopped.

```bash
cp experiments/tau-2/config/pipeline_4b_rl.env.example my_e2_lr1e-5.env   # edit paths, GPUs, learning rate
# 1) smoke test (about 1 hour): a few tasks per stage, 2 GRPO steps, to check the whole chain works
SMOKE=1 RUN_DIR=/abs/results/smoke bash experiments/tau-2/10_4b_rl_pipeline.sh my_e2_lr1e-5.env
# 2) full run: all stages
bash experiments/tau-2/10_4b_rl_pipeline.sh my_e2_lr1e-5.env
# evaluate checkpoints while training runs: a second process
WATCH=1 bash experiments/tau-2/10_4b_rl_pipeline.sh my_e2_lr1e-5.env evalckpt
```

| Stage | What it does | Corresponds to |
|---|---|---|
| `check` | Checks driver, models, the three environments, `causal_conv1d`; on first run builds the AReaL `valid_task_ids.json` | Sections 3, 4 |
| `base` | 4B base on VAL (seed 300) and on the frontier (58 tasks × 8) | E0 |
| `r1` | 4B base samples each valid AReaL task and each TRAIN task `R1_TRIALS` times, sharded over `EVAL_GPUS` | E1 |
| `select` | Picks tasks with 0 < p < 1 as GRPO prompts (`DATA_MIX=areal+train` or `train`, i.e. G1 / G2) | E1, E3 |
| `train` | Builds the GRPO parquet and runs group `O` of 09, multi-GPU `TRAIN_GPUS` / `ROLLOUT_GPUS`, optional TIS; prints the mean truncation rate at the end and warns above 5% | E2 |
| `evalckpt` | Each checkpoint → adapter → frontier → per-task paired comparison with base (`compare_tasks.py`), written to `eval/step_N/compare.json` | Section 6 |
| `val` | VAL (seeds 300 + 1300 by default) only for checkpoints that pass the bar (≥ +5 pt and CI lower bound > 0), compared with base | Decision |

- All results and summaries are appended to `$RUN_DIR/pipeline.log`; re-run a single stage with `FORCE=1`.
- Use one config file / `RUN_DIR` per E2 learning rate and per E3 group (G1/G2).
- Note: the default frontier evaluation set overlaps the TRAIN-derived training prompts (see 1.4); it is only a "did it learn anything" gate,
  and conclusions come from VAL. For a held-out gate, point `FRONTIER_PARTITION` at a partition without the training tasks.
- Smoke-tested on RTX 5090s: `check → base → r1 → select → train → evalckpt → val` (about 1 hour), including explicit TIS
  (the log shows `rollout_corr/rollout_is_mean ≈ 0.997`). Multi-GPU `TRAIN_GPUS` has not been tested yet; run the smoke test first.
- `TRAIN_BATCH_SIZE × N_GENERATIONS` must be a multiple of the number of agent-loop workers (16 by default); the script checks this up front.

The subsections below give the details and decision rules for each step. Training underneath is always group `O` of
`experiments/tau-2/09_phase1_opd_ablation.sh` (GRPO, official reward only) with `SKIP_EVAL=1`.
`experiments/tau-2/scripts/r1_job.sh`, `r3_group.sh` and `r3_watchdog.sh` are our wrappers on the original machines
(**paths are machine-specific**); use them only as a reference for argument combinations.

### E0 Reproduce the baseline (runtime from the A100 40 GB serving profile)

- 4B base on VAL (seed 300) and on the frontier should give about 34% and 58% (within ±3 pt counts as consistent).
- Use `08_phase0_eval_protocol.sh` (VAL) and `scripts/chunked_collect.sh --bucket TRAIN --trials 8 --seed 2000 --agent-temperature 1.0 --user-temperature 0.0 --max-steps 40`
  with `TAU2_PARTITION_PATH=experiments/tau-2/data/partition_frontier_only.json` (frontier).
- If the numbers differ, check first: is the user simulator Qwen3.5-4B, is `enable_thinking` off, is the tau2 commit the same.

### E1 R1 selection (shard after serving/memory profiling)

- 4B base samples each valid AReaL task (1,844) and each TRAIN task (139) 4 times (temperature 1.0, 40 steps):
  `collect_traces --bucket AREAL|TRAIN --trials 4 --seed 4000 --agent-temperature 1.0 --user-temperature 0.0 --max-steps 40` (parallelize with `--shard i/n`).
- Pick tasks with 0 < p < 1 with `scripts/select_frontier.py`, then build the GRPO parquet with `tau2_evolve.prepare_grpo_data --max-steps 40 --no-balance-domains`.
- Expected: about 546 AReaL tasks and about 36 TRAIN tasks (our result; ±10% with different seeds).

### I. Training infrastructure optimization (before full E2 runs and M student updates)

**Goal.** Make the existing 4B training stack produce more complete, usable trajectories per GPU-hour and finish updates
faster on 40 GB GPUs. Start with bf16 and micro-batch 1; use sharding and gradient accumulation to accommodate long
trajectories and the target effective batch. This stage is planned work; multi-GPU training and the gains below are
not yet validated on A100 40 GB.

1. **Establish a profile.** After the 2-step plumbing smoke test, use a fixed TRAIN-only workload for a short warmed-up
   benchmark (e.g. 10–20 steps). Record rollout and user-simulator latency (median / p95), actor forward/backward and
   log-prob time, weight-sync time, checkpoint time, GPU utilization and peak memory per role. Report wall time and
   total allocated GPU-hours, including serving GPUs. Count complete valid trajectories regardless of task success;
   separately log official reward, token/turn truncations, tool errors and infrastructure failures.
2. **Actor memory and compute.** Start with bf16, gradient checkpointing, remove-padding and the existing fused-kernel
   path; verify that FLA / causal-conv1d are active. Start `PPO_MICRO_BATCH_SIZE=1`, then tune it with the global prompt batch and
   generations fixed, using gradient accumulation to preserve the update batch. Keep CPU offload off initially
   (see the warning in `09_phase1_opd_ablation.sh`); benchmark it only if memory is the measured constraint. Evaluate
   dynamic token batching or length bucketing only after checking support in the pinned hybrid-model stack and
   preserving per-group GRPO normalization and loss weighting.
3. **Rollout and user serving.** Tune `AGENT_LOOP_WORKERS`, `REWARD_WORKERS`, `RAY_NUM_CPUS`,
   `ROLLOUT_GPU_MEMORY_UTILIZATION` and serving concurrency against the measured bottleneck. Track long-tail
   conversations and user-server queues; higher concurrency only helps if complete-trajectory throughput rises.
   Preserve seeds, sampling parameters, tool protocol and token/turn budgets. Check vLLM graph mode versus the
   current eager path only in an isolated compatibility benchmark.
4. **GPU allocation and communication.** With the teacher stopped, benchmark 4 actor + 3 rollout + 1 user-serving
   GPU against 2 actor + 5 rollout + 1 user-serving GPU on the same workload. If the teacher remains resident,
   start with 4 actor + 2 rollout + 1 teacher + 1 user-serving GPU. Verify actual rank placement,
   device isolation, batch divisibility for the actual actor ranks and weight synchronization. Keep
   `CHECKPOINT_BUCKET_MB` above the largest transferred tensor;
   measure sync stalls before tuning transfer settings. Reserve separate evaluation GPUs only when the reduction in
   training capacity is justified; otherwise evaluate between runs. Keep offline teacher-data collection and student
   updates as separate logical phases; a resident teacher does not enable online distillation by itself.
5. **Freeze and compare.** Select infra settings using TRAIN throughput, memory and stability, then freeze them for
   the scientific comparisons. Keep model revisions, data order, reward, optimizer, global batch, generations, LoRA,
   TIS, decoding and length limits fixed within each infra comparison. A bf16 change needs finite-loss/gradient and
   reward-distribution checks; asynchronous scheduling changes also need rollout-policy lag / IS diagnostics. Changes
   to response cap, global batch or TIS are explicit E2 training-setting changes, not throughput-only improvements.

**Implementation work.** Done (2026-11-10, CPU-tested): every variable in the pipeline config is now exported, so
infra knobs such as `PPO_MICRO_BATCH_SIZE`, `AGENT_LOOP_WORKERS` or `ROLLOUT_GPU_MEMORY_UTILIZATION` reach
09 / `03_grpo_only.sh` / `train_verl_grpo.sh`; `EXTRA_TRAIN_ARGS` from the config is appended to the pipeline's own
overrides and wins on a shared key; the effective settings are written to `$RUN_DIR/train_settings.env` and per-stage
wall times to `$RUN_DIR/stage_times.tsv`; the pipeline and example config default to bf16. Adapter extraction
(`verl_code_rl/extract_sft_lora_adapter.py`, used by `evalckpt` and by `train_verl_grpo.sh` for the final adapter)
previously accepted only world_size = 1 checkpoints and now merges multi-rank FSDP shards in their stored dtype;
on synthetic 2-rank checkpoints (DTensor and plain shards) built from a real 4B actor checkpoint the result is
bit-identical to the single-rank extraction. (`verl.model_merger` is not used: it casts to bf16, and for plain
shards its mesh inference counts one shard and would merge rank 0 only.)
Still open: a real multi-GPU actor run to confirm the shard format and checkpoint save/resume, and a profiling
report. VERL logs only `num_turns/{min,mean,max}`, so the turn-budget truncation rate needs a per-rollout count
(e.g. from the agent loop) before the acceptance rule below can be checked.

**Acceptance.** Token-budget and turn-budget truncation must each stay below 5%, with no OOM or silently dropped
rollouts in the benchmark. Report complete valid trajectories / second, generated tokens / second, actor update time,
end-to-end step time and GPU-hours; keep only changes with measured throughput or memory benefit and stable training
diagnostics. Re-estimate the full-run schedule from this profile. Infrastructure GPU-hours are reported separately
from the main curve's online teacher-token cost.

### E2 GRPO without truncation (student update without a teacher; answers Q1; an M ablation)

| Item | Old setting on 32 GB | New setting |
|---|---|---|
| Response cap (whole multi-turn conversation) | 5632–6144 | **16384** (target truncation rate < 5%) |
| Prompt cap | 8192 | 8192 |
| Max turns `MAX_TURNS` | 30 | 30 (raise to 40 if truncation mostly comes from the turn cap) |
| Prompts × generations per step | 4 × 8 | **16 × 8** (or 8 × 16) |
| Learning rate | 1e-5 → 3e-5 | one run each at 1e-5 and 3e-5 |
| LoRA | rank 16, attention + MLP | same; rank 64 is an optional capacity ablation after memory profiling; full fine-tuning is outside the initial 40 GB plan |
| Actor precision / micro-batch | `fsdp_config.model_dtype=bf16` | bf16 (pipeline default since 2026-11-10), `PPO_MICRO_BATCH_SIZE=1`; validate and freeze in Section I |
| Importance-sampling correction | VERL one-step off-policy defaults to `bypass_mode=True`: old_log_prob is the rollout log-prob, and the PPO ratio π_θ/π_rollout is clipped (implicit correction, no explicit IS weights) | **Explicit TIS**: `bypass_mode=False` (decoupled, recomputes old_log_prob) + `rollout_is=token`, `rollout_is_threshold=2.0`, `calculate_log_probs=True`; the pipeline's `ENABLE_TIS=1` sets all four |
| Steps / checkpoints | 250 / every 50 | 200 / every 25 |

- Starting GPU layout with teacher stopped: user simulator on GPU 0, actor on GPUs 1–4 and rollout on GPUs 5–7.
  With a resident teacher: teacher on GPU 0, user simulator on GPU 1, actor on GPUs 2–5 and rollout on GPUs 6–7.
  These are proposed profiles, not tested allocations; run a 2-step smoke test first. The current `09` driver also
  starts a student evaluation server beside the user simulator: verify their combined memory on 40 GB or change
  server lifecycle management so that evaluation runs between training jobs.
- The 16k response cap plus 8k prompt cap is a 24k configured model budget, not a single-turn output budget.
  If memory is tight, tune micro-batching, concurrency and sharding before reducing trajectory length. Treat any
  unavoidable global-batch or length change as a new config and recheck truncation; 16 × 8 is a target, not a fit guarantee.
- Key environment variables (beyond those in `09`): `MAX_RESPONSE_LENGTH`, `TRAIN_BATCH_SIZE`, `N_GENERATIONS`, `ACTOR_LR`, `TOTAL_STEPS`, `SAVE_EVERY`,
  `LORA_TARGET_MODULES`, `GRPO_PARQUET`, `EXTRA_TRAIN_ARGS` (extra VERL overrides).
- **Metrics to watch every step** (`group_O/train.log`):
  - `response_length/clip_ratio` (truncation rate, must be < 5%);
  - 25-step moving average of `critic/score/mean` (should rise);
  - `actor/pg_clipfrac`, `actor/ppo_kl` (too small means almost no update; too large may mean collapse);
  - repeated tool calls per rollout (base about 0.01; > 0.2 is an early sign of collapse).
- Decision rules:
  - **Pass**: a checkpoint's frontier is ≥ 5 pt above the 4B base with CI lower bound > 0 → VAL (seeds 300 + 1300), then the final evaluation in Section 7.
  - **Truncation already < 5% but reward flat and frontier not rising** → go to E4.
  - **Collapse** (repeated calls rising, tool-call count falling) → roll back to the previous checkpoint and halve the learning rate.

### E3 Data comparison (answers Q2, same settings as E2)

- G1 (AReaL frontier + TRAIN frontier) vs G2 (TRAIN frontier only). G2 has few tasks (about 36), so watch for overfitting and
  draw conclusions from VAL, not the frontier.

### E4 Credit assignment (answers Q3; only if E2 fails)

- **Per-turn advantage estimates**: branch and resample at intermediate turns and estimate each turn's value by Monte Carlo
  (the branching / per-turn machinery of `tau2_evolve/collect_error_points.py` and `turn_router.py` can be reused).
- **PPO + critic**: VERL supports GAE; a 4B critic needs a separate memory/sharding smoke test on this 40 GB node.
  Rebalance actor, critic and rollout GPUs explicitly; this arm is not assumed to fit the GRPO allocation unchanged.
- **Process reward** (early in training, annealed to 0 later): `TAU2_ACTION_RECALL_WEIGHT` and `TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS` are implemented (action recall).
  This is the cheapest E4 arm: it uses the tasks' gold actions as partial credit, needs no teacher and no new code
  (group `O` sets the weight to 0; the training script's default is 0.5). Try it first.

### What the gold "answers" can and cannot do

Every official tau2 task and every AReaL RL task ships evaluation criteria: the gold tool calls (name + arguments) and the
expected final database state. They are **not** full trajectories. They do not show how to obtain the IDs from the user,
when to verify identity, when to ask for explicit confirmation before a write, when policy requires refusing (the correct
action is then "change nothing"), or what to say. The user simulator's messages also differ from run to run, so a bare
action list cannot be used as SFT data. What the gold answers already do:

| Use | Where | Needs a teacher? |
|---|---|---|
| Outcome reward (final DB state = official tau2 verdict) | GRPO, every experiment | No |
| Partial credit (action recall) | E4 process reward | No |
| Hints for generating correct trajectories in the real environment | **E5 (new)** | No |

Their limit: a reward cannot teach a task the student never solves (p = 0 gives no GRPO signal), and hints cannot show the
conversational behaviour the student gets wrong. That is what a strong teacher adds (E6).

### E5 Gold-action-guided self-rollouts (answers Q5 without a teacher; needs new code)

- **Idea**: the 4B student talks to the user simulator in the real tau2 environment as usual, but before each agent turn it
  receives a hidden hint with the next gold action (e.g. `next: update_reservation_flights(reservation_id=..., ...)`).
  The student writes all language itself. Keep only rollouts that the official evaluator passes, **remove the hints**, and
  train on what remains (SFT with loss on the student's turns, or add them as positive samples next to GRPO groups).
- **Why it might work where AReaL SFT failed**: the demonstrations are in the real tau2 distribution (real DBs, real user
  simulator) and in the student's own style, so there is no style shift.
- **Targets**: TRAIN and valid AReaL tasks with p = 0 or low p in R1 (E1 already measures p per task).
- **Implementation (not written yet)**: an agent wrapper that inserts the hint into the system or last user message;
  reuse the per-turn injection machinery of `tau2_evolve/turn_router.py` / `_FirstMessageAgent` in `benchmark.py`.
  Read actions (lookups) can be hinted as well; tasks whose gold actions are only writes need the IDs from earlier turns.
- **Risk**: the trained model never sees the hint, so it may learn the surface of correct trajectories without the cue that
  triggered them. Measure on VAL; also compare against a "hints at train time, hint dropout 50%" variant.
- **Pass**: VAL improves over the E2 best (or over base if E2 failed), and p rises on the targeted p = 0 tasks in a
  held-out subset of them.

### E6 Local Qwen3.5-9B teacher (candidate for the M loop; answers Q5; no API)

Keep the **Qwen3.5-4B student** and qualify **Qwen3.5-9B as the frozen teacher** on the 8 × A100 40 GB node.
The existing 27B result shows why a larger parameter count is insufficient. There are no 9B results in this plan yet.

- **Model**: `Qwen/Qwen3.5-9B`, using the released post-trained model, not the `-Base` checkpoint. Pin its revision
  and tokenizer. This is the same Qwen3.5 series as the student; the selected teacher is 9B, not Qwen3-8B.
  [Official model and serving guide](https://huggingface.co/Qwen/Qwen3.5-9B)
- Keep the teacher unchanged across all cycles and both benchmark lines. The 4B user simulator is a separate
  frozen role and remains unchanged. Student adapters and SkillBooks never modify either frozen model.
- The earlier 235B deployment is outside this hardware plan. There is no default teacher quantization or FP8 arm.

#### Serving on A100 40 GB (driver ≥ 580 for the selected serving environment)

- Start with one dedicated GPU, **BF16, TP = 1**, a 32k context budget and low concurrency. Rough weight-only memory
  is about 18–20 GB; runtime buffers, GDN state and KV cache are additional. This is a sizing estimate, not a verified
  single-card concurrency or long-context fit claim. Profile peak memory and complete-trajectory throughput.
- Use the pinned serving environment with `--reasoning-parser qwen3`, `--enable-auto-tool-choice` and
  `--tool-call-parser qwen3_coder`; use text-only serving (`--language-model-only`) when supported by that version.
  Smoke-test reasoning separation, native tool calls, actual `completion_tokens` usage and multi-turn history order.
  These parser options follow the [official 9B serving guide](https://huggingface.co/Qwen/Qwen3.5-9B).
- Teacher thinking stays **on**, while student and user-simulator thinking stay **off**, as in the current protocol.
  Log generated reasoning tokens as teacher cost but remove them from student training targets. Freeze decoding,
  context and output limits after qualification. Do not silently truncate BFCL long-context inputs to fit 32k;
  provision the required context and evaluate that serving profile separately if needed.
- If the required context or concurrency does not fit, first lower concurrent sequences, then benchmark **TP = 2**
  with a revised GPU allocation. Recheck quality after any quantization change; it is a separate experiment.
- Distill **offline** through verified corrections/demonstrations, then SFT + GRPO. A 9B teacher can remain resident
  during training if the allocation fits, but stopping it releases one GPU for actor or rollout work. Online reverse-KL
  groups C/D/E are outside the initial comparison; same-series models still need explicit tokenizer-vocabulary and
  tool-format checks before token-level distillation is attempted.

#### Step 0 — qualify the teacher (right after E0; runtime determined by the serving profile)

1. Evaluate 9B on the frontier (58 × 8) as a diagnostic, then compare 9B and the 4B base on **VAL seeds 300 + 1300**
   with the same frozen 4B user simulator, tasks and paired trial seeds. Use matched generation budgets and the
   declared per-model thinking settings. Report overall/domain pass^1 and the per-task paired CI of the difference.
2. Require a **positive VAL gap whose 95% paired CI excludes 0**, plus verified teacher recoveries of student failures
   on a fixed TRAIN shard. A gain of about 10 pt is the planning target, not an assumed 9B score. Report telecom
   separately; its improvement is an empirical question. Select infra settings on TRAIN rather than sweeping them on VAL.
3. Freeze the qualified serving/decoding configuration and use it for corrections and always-teacher endpoints.
   BFCL needs its own B1 qualification on independent DEV tasks; a TAU-2 pass does not qualify it automatically.

If the teacher does not pass, stop the teacher-driven M/E6 run and continue E2, E4 or E5. Do not spend additional
cycles on an unqualified teacher or assume that switching to 2B would solve the headroom problem.

#### Step 1 — teacher data (dedicated 9B server plus 4B student/user servers)

All in the real tau2 environment; TRAIN tasks and valid AReaL tasks only (never VAL/EVAL).
- **Error-point corrections** on the student's failed rollouts (the R1 rollouts from E1 can be reused):
  `tau2_evolve/collect_error_points.py --teacher-model ... --teacher-base-url ...`; keep only teacher fixes that the
  official evaluator verifies, then `tau2_evolve/error_points_to_sft.py` with `--rehearsal-traces` (student's own
  successful rollouts, to avoid forgetting) and `--correction-repeat` (corrections are ~3% of tokens otherwise).
- **Full demonstrations** on tasks with no observed student successes in E1 (empirical p = 0): `collect_traces` with the teacher as agent
  (`--agent-base-url`, thinking on, the reasoning removed by the parser); keep passing rollouts only.
- The student serves its own rollouts with thinking off, so strip any reasoning from the training targets and keep
  the native tool-call format (`check_native_alignment.py`).

#### Step 2 — train the student (offline teacher data; GPU profiles below)

- **Group F** (`09_phase1_opd_ablation.sh`, `CORRECTIONS_PARQUET=<step 1 parquet>`): SFT on teacher corrections and
  demonstrations.
- **Group G**: F, then GRPO from F's adapter with the E2 settings (16k response cap, 16 × 8, explicit TIS).
- Use BF16 and micro-batch 1 initially; qualify the long-trajectory configuration on 40 GB before a full run. Start
  with the teacher stopped to reclaim its GPU, or profile the resident-teacher allocation without changing the loss.
- Compare against **group O** (E2, GRPO without a teacher) on the same VAL seeds. The question is whether the teacher
  adds anything beyond the reward alone.
- Watch the SFT failure mode we saw with AReaL's SFT dialogues (fewer tool calls, "does less"): compare tool calls per
  trajectory against base on the frontier before trusting a VAL number.

#### Pass

Same rule as E2: frontier gate (≥ +5 pt, CI lower bound > 0), then VAL seeds 300 + 1300; G should beat both base and O.

### Schedule on one 8 × A100 40 GB node

| Phase | GPUs | What runs |
|---|---|---|
| E0 baseline + E6 step 0 | 1 GPU each for 4B student, 4B user and 9B teacher; run paired policies under the same simulator protocol | base VAL/frontier; teacher qualification |
| E1 R1 selection | teacher stopped; shard student/user serving over available GPUs after a memory smoke test | 4B rollouts on TRAIN + AReaL |
| E6 step 1 / M teacher phase | 9B teacher on GPU 0 (TP = 1 initially), 4B user on GPU 1, 4B student on GPU 2; other GPUs available for measured serving replicas | corrections, demonstrations, router data and routed curves |
| E2 / E6 step 2 / M training phase, teacher stopped | GPU 0: user-serving; GPUs 1–4: actor; GPUs 5–7: rollout | offline SFT + GRPO; profile 2 actor + 5 rollout as an alternative |
| Optional resident-teacher training | GPU 0: teacher; GPU 1: user-serving; GPUs 2–5: actor; GPUs 6–7: rollout | same offline objective, with the teacher endpoint kept available |
| Checkpoint evaluation | separate serving jobs between training runs; start with isolated student/user GPUs | frontier / VAL, adapter activation checks |

These allocations are untested on this node. Current training drivers may also start an evaluation server on the
user-serving GPU; budget that explicitly or stop it during updates. TP = 2 teacher serving needs a new allocation,
not an extra ninth GPU. A 3 actor + 3 rollout + 1 teacher + 1 user layout is another candidate only if its effective
mini-batch divides correctly over three actor ranks; do not assume the target 16 × 8 batch works unchanged.
For BFCL, release the user-simulator GPU and re-profile the allocation.

### Decision flow

```text
E0 baseline ─► E6 step 0: 9B teacher has a verified advantage over the 4B student?
                 │ no  ─► stop the teacher loop; run component experiments (E2 GRPO, E4, E5)
                 │ yes
                 ▼
E1 R1 rollouts ─► M cycle 0: router-only curve on VAL (baseline)
                 ─► M cycle 1: corrections + SkillBook + student update + router ─► curve
                      │ curve moved (cost at matched quality ↓) ─► cycles 2 … K + ablations ─► final report (Section 7)
                      │ curve did not move ─► stop; diagnose with E2 (GRPO only), E4 (process reward), E5 (gold-guided)
```

## 6. Evaluation workflow

- A trained LoRA that covers only attention + MLP can be loaded by vLLM directly: `AGENT_ADAPTER` in `08_phase0_eval_protocol.sh`
  converts VERL's exported key names to the `language_model` form vLLM needs, and **asserts that the adapter is actually active**
  (log-probs differ from base); it fails loudly otherwise.
- If the LoRA covers GDN projections (`in_proj_*`, `out_proj`), vLLM's LoRA path is incomplete: merge it into a full checkpoint with
  `scripts/merge_lora_qwen35.py` before evaluating.
- VERL checkpoint → adapter: `python -m verl_code_rl.extract_sft_lora_adapter --checkpoint-dir <run>/group_O/verl/global_step_N/actor ...`.
- Long-running Qwen3.5 vLLM servers drift into a GDN "zero-throughput" state: always evaluate through `scripts/chunked_collect.sh`
  (restarts the servers every 128 tasks).
- Two vLLM servers on one GPU need `VLLM_MAX_NUM_SEQS=128` (otherwise there are not enough Mamba state blocks and the server does not start).

---

## 7. Final report (Q4)

- On EVAL, with the same frozen local Qwen3.5-4B user simulator, model/tokenizer revision and serving/decoding settings
  as VAL, report pass^1, pass^4 and 95% CIs for the 4B base and the best checkpoint, seeds 300 + 1300.
- The headline is the **cost–quality curve on EVAL** for cycle 0 and the last admitted cycle: always-student,
  always-teacher, and the learned router at the operating points fixed beforehand on TRAIN (no new threshold choices
  on EVAL), with teacher-generated tokens as cost.
- EVAL must not be used for any selection before that.
- **Compute budget**: all TAU-2 stages, including final EVAL, run on local models; no hosted-model API budget is required.
  Final EVAL is 67 tasks × 4 trials × 2 seeds = 536 conversations per reported policy, run once. Record the frozen
  user-simulator configuration alongside the agent, router and SkillBook artifacts.

---

## 8. Pitfalls we hit (all fixed; watch for them in a new environment)

| Problem | Symptom | Fix |
|---|---|---|
| vLLM silently ignores Qwen3.5 LoRAs exported by VERL | Output identical to base | Convert key names before evaluation and assert that log-probs differ |
| Training used JSON tool format, evaluation uses Qwen3.5's native XML | What training learned is not used | Train with the native protocol (`TOOL_FORMAT=qwen3_coder`), checked by `check_native_alignment.py` |
| LoRA only on q/k/v/o | Under Qwen3.5's hybrid architecture only 6/24 layers are trained | MLP added by default |
| Injected teacher messages kept old timestamps | tau2 sorts by time and the order breaks; evaluation fails with "Tool message expected" | Reset the timestamp on injection |
| Telecom users' own tool calls mixed into the agent history | tau2 rejects the history; SFT data gets empty messages | Filter with tau2's `is_valid_agent_history_message` |
| Missing `causal-conv1d` | GDN falls back to torch; SFT 2.5× slower | Install the prebuilt wheel or build from source (works with CUDA 11.8 / 12.x) |
| Actor defaults to fp32 | 4B OOMs during the update on 32 GB | `fsdp_config.model_dtype=bf16` |
| Weight-sync bucket smaller than the largest tensor | Error during sync | `CHECKPOINT_BUCKET_MB` must exceed the vocabulary embedding (about 1.27 GB for 4B bf16) |
| Length cap too low | 24–40% of conversations truncated and scored 0 | The core issue of this handoff; see E2 |
| Assuming `rollout_is` alone turns on TIS | One-step off-policy defaults to `bypass_mode=True`, where `rollout_is` weights are ignored (the config is accepted without any error) | Also set `algorithm.rollout_correction.bypass_mode=False`; confirm `rollout_corr/*` metrics appear in the log |

---

## 9. Code map

| Path | Purpose |
|---|---|
| `experiments/tau-2/SETUP_4B_RL.md`, `scripts/setup_envs.sh`, `scripts/download_assets.sh` | Existing environment/data setup; hardware examples and teacher download targets need alignment with the A100 40 GB / 9B plan |
| `experiments/tau-2/10_4b_rl_pipeline.sh`, `config/pipeline_4b_rl.env.example` | One-command workflow (Section 5.0) |
| `experiments/tau-2/scripts/compare_tasks.py` | Per-task paired comparison of two multi-trial runs and the pass-the-bar decision |
| `experiments/tau-2/08_phase0_eval_protocol.sh` | Evaluation protocol (servers, adapter assertion, chunked collection, statistics) |
| `experiments/tau-2/09_phase1_opd_ablation.sh` | Training entry point; group `O` = GRPO with the official reward; `EXTRA_TRAIN_ARGS`, `SKIP_EVAL` |
| `experiments/tau-2/03_grpo_only.sh`, `scripts/train_verl_grpo.sh` | VERL GRPO launch (accepts multi-GPU lists) |
| `experiments/tau-2/scripts/chunked_collect.sh` | Chunked collection (avoids the GDN stall) |
| `experiments/tau-2/tau2_evolve/collect_traces.py` | Sampling; `--bucket AREAL`, `--shard`, `--trials`, `--turn-router` |
| `experiments/tau-2/tau2_evolve/areal_tasks.py` | AReaL task loading and per-task databases |
| `experiments/tau-2/tau2_evolve/eval_stats.py` | pass^k, paired bootstrap, McNemar |
| `experiments/tau-2/tau2_evolve/frontier.py`, `scripts/select_frontier.py` | Frontier split and GRPO prompt selection |
| `experiments/tau-2/tau2_evolve/prepare_grpo_data.py` | GRPO parquet (native protocol) |
| `experiments/tau-2/tau2_evolve/collect_error_points.py`, `turn_router.py` | Error points, per-turn routing (reusable for E4) |
| `experiments/tau-2/scripts/check_areal_tasks.py` | AReaL task replay check |
| `experiments/tau-2/scripts/merge_lora_qwen35.py` | Merging LoRAs that include GDN projections |
| `experiments/tau-2/tau2_evolve/router.py`, `train_router.py` | Learned escalation router (state features, threshold) and its training (M) |
| `experiments/tau-2/tau2_evolve/build_skillbook.py`, `skills.py` | SkillBook construction (M) |
| `experiments/tau-2/scripts/run_cycle_pipeline.sh` | Multi-cycle loop driver; needs porting to partition_v2 / 4B / local teacher (M) |
| `experiments/tau-2/tau2_evolve/agent_loop.py` | GRPO reward, including the action-recall process reward (`TAU2_ACTION_RECALL_*`, E4) |
| `experiments/tau-2/tau2_evolve/error_points_to_sft.py` | Teacher corrections → SFT parquet (E6) |
| `experiments/tau-2/PLAN_4B_A800.md`, `PLAN_4B_RL.md` | Plans of the previous phase and their step-by-step adjustment log |

---

## 10. Time and compute budget (8 × A100 40 GB; profile before estimating full runs)

No end-to-end timings have been measured for this 4B/9B configuration on the target node. Previous 80 GB / 235B
estimates do not apply. Budget the infra smoke/profile and one admitted update first; continue later cycles only
after observing a curve improvement. Record elapsed time and allocated GPU-hours per phase, including idle reserved
GPUs and serving jobs. The workload counts below are planning inputs, not throughput guarantees.

| Step | Estimate |
|---|---|
| Environments + data (before E0) | about 1 hour (`SETUP_4B_RL.md`) |
| E0 baseline reproduction | VAL seed 300: 268 conversations; frontier: 464; time from measured 4B serving throughput |
| E1 R1 selection | about 7,932 conversations at 4 trials over 1,844 valid AReaL + 139 TRAIN tasks; profile sharding on 40 GB |
| I: infra profiling and optimization | budget half a day initially; measured throughput determines the revised training estimates |
| E2 LR runs / E3 data arms | each planned run: 200 steps plus 464 frontier conversations per checkpoint; estimate from warmed-up step times and evaluation throughput; run one training job at a time |
| E4 (if needed) | process reward adds one profiled GRPO run; PPO + critic needs a separate memory/allocation test |
| E5 (if needed) | budget ~1 day for the hint wrapper initially; rollout and SFT/GRPO time from profiling |
| E6 step 0 (teacher qualification) | pinned 9B download, 464 teacher frontier conversations and paired 536-conversation VAL runs per model; reuse matching frozen base results |
| E6 step 1 (teacher data) | time from failed-prefix count, branch attempts, passing demonstration yield and measured 9B thinking-token throughput |
| E6 step 2 (groups F and G) | F: profile SFT token throughput; G: one additional profiled GRPO run |
| M, per cycle | collection + teacher corrections + SFT/GRPO + router/admission; the 8-point VAL curve (2 endpoints + 6 router thresholds) is 8 × 536 = 4,288 conversations, before extra baselines |
| M, cycle 0 + one update first | replace the full-loop calendar estimate after this pilot; later cycles and ablations are conditional on the pass rules |
| Final VAL / EVAL | 536 conversations per reported policy per split, plus curve baselines; same local 4B user simulator, no hosted API |

## 11. Conventions

- One configuration per `RESULTS_DIR`; never mix settings in an existing directory; resume inside the original directory.
- Never train on VAL/EVAL, and never use EVAL results for any selection.
- For TAU-2, keep the user simulator frozen at the same Qwen3.5-4B base checkpoint across training, VAL and EVAL; student
  updates never modify it. Snapshot model/tokenizer revisions and effective serving, decoding and infra settings.
- Keep the Qwen3.5-9B teacher frozen after qualification across cycles and benchmarks. Record both the model
  revision and thinking/serving settings; benchmark-specific qualification is still required.
- **No outside trajectories on official tau2 tasks.** Public tau2 leaderboard submissions and similar logs cover the
  official tasks, i.e. our VAL and EVAL. Training on them leaks the test set. Only TRAIN-task trajectories and AReaL
  synthetic tasks may be used; generate teacher data yourself (E5/E6).
- Draw conclusions only after reloading (or merging) the adapter; every evaluation must pass the adapter-active assertion.
- `.env`, API keys, model weights and `results/` stay out of the repository.

---

## 12. Second benchmark: BFCL-V4 (planned)

### 12.1 Question, models and scope

**Question.** Does the MERA loop move the teacher-token cost–quality curve on tool-use tasks beyond TAU-2?
Use the same **Qwen3.5-4B student base checkpoint**, LoRA setup and frozen candidate **Qwen3.5-9B teacher**
on the **8 × A100 40 GB** node.
Start the BFCL-specific loop from a fresh adapter on that base; the TAU-2-trained adapter is an optional transfer
ablation, selected on TAU-2 VAL before BFCL testing. Keep each model's thinking mode fixed across cycles. Re-qualify
the teacher on BFCL development tasks: TAU-2 headroom does not establish BFCL headroom.

BFCL-V4 adds agentic web search and memory to the earlier function-calling and multi-turn evaluations. Prioritize
multi-turn tool use and memory for loop development, then report the full scoring suite for the final comparison.
The official V4 composition weights Agentic 40%, Multi-Turn 30%, Live 10%, Non-Live 10% and Hallucination 10%; use the
pinned official scorer rather than a pooled task average. Format sensitivity is a separate diagnostic, not part of
the headline score. [Official V4 methodology](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)

| Phase | Categories | Purpose |
|---|---|---|
| First integration | `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context` | Stateful calls, missing information and long context |
| Agentic extension | `memory_kv`, `memory_vector`, `memory_rec_sum` | Memory writing and retrieval across sessions |
| Final coverage | `all_scoring`, including web search, single-turn/live and irrelevance | Full V4 score and category breakdown |
| Optional diagnostic | `format_sensitivity` in its supported prompting mode | Report separately from native-FC results |

These are the official category names; confirm availability in the pinned version before writing launchers.
[Official category reference](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/TEST_CATEGORIES.md)

**Environment.** Preserve BFCL's supplied user turns, tool backends and termination rules; do not add TAU-2's 4B user
simulator. Preserve the memory prefill → snapshot → isolated-question protocol. A routed policy must carry the
current conversation and tool state into the teacher; for memory, route prefill and retrieval calls with explicit
model ownership and retain all associated teacher-token costs.
[Official inference loop](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py),
[memory protocol](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html)

### 12.2 Freeze the evaluation and build separate training data

- Pin the BFCL code commit, data checksums, scorer, model/tokenizer revisions, FC handler, chat template, decoding,
  context limits and tool-step limits before running comparisons. Verify native Qwen3.5 tool-call round trips rather
  than substituting a Qwen3 model name and assuming compatibility.
- Keep the official BFCL evaluation entries and their reference answers out of training, teacher-correction
  collection, SkillBook construction, router fitting and calibration. Do not use public model response logs as
  demonstrations. Inspecting full official scores must not become a checkpoint or threshold selection loop.
- Create a separate executable TRAIN/DEV corpus with new scenarios and initial states. Audit provenance and overlap
  against official cases; renaming entities or paraphrasing a benchmark question is insufficient. Keep related task
  variants together, and split memory data by source conversation/session group across all backends. Freeze the
  manifest before training. This data work is a prerequisite for a BFCL-trained MERA claim.
- Use DEV for headroom checks, adapter selection, loop admission and threshold calibration. Reserve the full official
  suite for the final, predeclared policies. An early smoke test uses independent development cases; any official
  cases used for debugging must be disclosed and excluded from an untouched-test claim.
- Web search needs a separately configured search service in the current official harness (SerpAPI). Record provider,
  run time and tool responses, and interleave paired policy runs to reduce temporal drift. Cached replay is a
  separate controlled diagnostic. Local model serving removes hosted-model charges, not search-service costs.
  [Official setup and evaluation guide](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md)

### 12.3 Experiments and decision gates

| Experiment | Run | Gate / deliverable |
|---|---|---|
| B0: harness and baseline | Independent DEV smoke cases; 4B base, candidate teacher; optional frozen TAU-2 adapter | Native tool parsing, state isolation, token usage and scorer verified; DEV baseline by category |
| B1: teacher qualification | Paired student/teacher DEV runs, including multi-turn and memory | Positive paired quality gap with CI excluding 0 and verified recoveries of student failures; otherwise stop the teacher loop |
| B2: cycle 0 | Fixed base student + learned router, always-student/teacher endpoints, random and tool-error routing | DEV cost–quality curve; operating points calibrated without official-test feedback |
| B3: one MERA update | TRAIN rollouts → verified teacher suffixes + rehearsal → SkillBook → SFT + GRPO → refit router | Admit jointly on DEV; continue to cycles 2–3 only if the curve improves |
| B4: final report | Frozen cycle 0 and last admitted cycle, endpoints and predeclared ablations on official `all_scoring` | Full V4 score, category scores, teacher-token curves, paired CIs and infra GPU-hours |

During B3, rebuild tool and memory state from the saved prefix before counterfactual teacher continuation. Verify
each corrected rollout with BFCL-compatible checks for that training category; the TAU-2 reward cannot be reused.
Record invalid tool calls, repeated calls, context truncation and step-limit failures separately. Reuse Section I's
profiling workflow, then tune the actor/rollout/serving allocation for BFCL without reserving a user-simulator GPU.

**Cost and statistics.** Retain teacher-generated tokens, including reasoning, as the primary online cost and log
student tokens, latency and total GPU-hours separately. For memory, include prefill plus retrieval costs in the total
workload and report the phases separately; do not treat teacher-created memory as free. Freeze the evaluation
workload and prefill reuse policy across policies. Apply official category weights to quality; for a pooled token
curve state the task mix explicitly and also report category curves. Bootstrap paired task groups, keeping memory
questions sharing a source session together, and recompute weighted scores and interpolated curve statistics in
each replicate. Choose budgets and a numerical quality non-inferiority margin on DEV before final evaluation.

**Success.** Relative to cycle 0, the final admitted policy needs fewer teacher tokens at the predeclared matched
quality, with the paired CI of the cost difference excluding 0; its quality difference must pass the predeclared
non-inferiority test. Report student-only improvement and no-SkillBook / GRPO-only / routing-only ablations to
explain the curve movement. Multi-turn-only or memory-only runs are labeled **BFCL-V4 subsets**; without every
scoring category, do not report the result as full BFCL-V4 Overall Accuracy.

### 12.4 Implementation and execution order

All paths and recipes below are proposed, not runnable yet:

- `experiments/bfcl-v4/`: pinned setup, independent data manifests, model/FC handlers, trace collection and official
  evaluation wrappers. Keep BFCL trace, state and reward logic separate from `tau2_evolve`.
- Register `bfcl-v4` recipes in `scripts/run_experiment.sh` for smoke, baseline, cycle and final evaluation; reuse
  serving, LoRA extraction/reload checks and the experiment lifecycle. Add CPU checks for state isolation,
  tool-call conversion, split boundaries, memory snapshots and token accounting before GPU runs.
- Add correction/SFT conversion, a BFCL GRPO agent loop, router features/labels and category-aware curve analysis.
  Verify adapter activation and multi-rank checkpoint resume before committing to full training.

**Order.** Prepare the pinned harness and independent DEV cases after the TAU-2 baseline/infra smoke work. Run B0/B1
before committing to BFCL data generation and training; a teacher that fails B1 leaves a baseline/transfer result,
not an unqualified teacher loop. Run BFCL training sequentially with TAU-2 on the single 8 × A100 40 GB node. Budget the
harness/handler integration and independent TRAIN/DEV generation separately, then estimate GPU time from measured
per-category throughput; the TAU-2 schedule in Section 10 does not include this second benchmark.
