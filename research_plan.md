# TAU-2 Research Plan: RL for a 4B Student (handoff to a larger-compute team)

This document is for the people taking over these experiments with more compute. After reading it you should be able to
reproduce the existing results on your own and run the key experiments of the next phase.
General repository instructions (environments, launchers, invariants) are in `CLAUDE.md`; this document covers only the tau2 line.

Branch: `tau2-4b-rl-handoff` (based on `tau2-phase0-1-opd-27b`; contains all code and experiment scripts).

---

## 0. Status in one paragraph

On tau2-bench (airline / retail / telecom customer-service tasks), **switching the student from Qwen3.5-2B to Qwen3.5-4B is the only
significant improvement so far (VAL +10.4 pt)**. Later attempts to push further with a teacher, extra data for SFT, or routing had no
significant effect. The latest 4B GRPO ran 250 steps on 32 GB GPUs; on the frontier training tasks it mostly moved within noise.
G2, trained only on TRAIN tasks, had two checkpoints pass the bar (+5.4 / +5.6 pt), but the passing step 200 scored −2.2 pt on VAL
(not significant): the gain did not transfer (see 1.4).
**The main suspect is the length cap forced by GPU memory: about 30% of training conversations were truncated and scored 0.**
The core of the next phase is to remove truncation and enlarge the batch on large-memory GPUs, then re-test GRPO.
The whole workflow is a one-command script, `experiments/tau-2/10_4b_rl_pipeline.sh` (Section 5.0).

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
- User simulator for training and for every number in this document: local **Qwen3.5-4B**. The final EVAL is planned with a hosted
  strong model as the user simulator (needs an API key; not done yet).

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

1. **Q1 (main question)**: with truncation removed and a larger batch, does GRPO significantly improve 4B on the frontier
   (≥ +5 pt with CI lower bound > 0), and does it transfer to VAL?
2. **Q2**: do the AReaL synthetic tasks (G1) add anything over the official TRAIN tasks alone (G2)?
3. **Q3**: if Q1 is no, is the problem credit assignment (long conversations, terminal reward only)? Test with PPO + critic or per-turn advantage estimates.
4. **Q4**: report the 4B base and the best model on EVAL with a hosted user simulator.

---

## 3. Hardware and environment

- **To build the environments and download models and data from scratch, follow `experiments/tau-2/SETUP_4B_RL.md`**
  (two scripts, `scripts/setup_envs.sh` and `scripts/download_assets.sh`; written for AWS, official sources by default, optional S3 cache for models and data).
- Recommended: one node with 8 × 80 GB (A100 / H100 / A800), **NVIDIA driver ≥ 580**: the vLLM 0.29 used for evaluation ships CUDA 13
  torch; the training environment is cu128 and needs ≥ 570. Below 535, no vLLM that supports Qwen3.5 starts, so GRPO cannot run (SFT can).
- The training environment (VERL 0.8 + bundled vLLM 0.18, `.venv_qwen35`) and the serving environment (vLLM 0.29, for evaluation)
  are built by `setup_envs.sh` (background in the comments of `requirements-qwen35-cu129.txt`); tau2 runs in its own venv (`TAU2_WORKSPACE`).
- Qwen3.5's GDN layers **must use the fast path**: this needs `flash-linear-attention` and `causal-conv1d`. Without `causal-conv1d`,
  transformers silently falls back to a pure-torch implementation (training 2.5× slower or worse); search the logs for "fast path is not available".
- Models: Qwen3.5-4B (student and user simulator); optionally Qwen3.8-27B (teacher / upper-bound reference).

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

### E0 Reproduce the baseline (about 2 GPU-hours)

- 4B base on VAL (seed 300) and on the frontier should give about 34% and 58% (within ±3 pt counts as consistent).
- Use `08_phase0_eval_protocol.sh` (VAL) and `scripts/chunked_collect.sh --bucket TRAIN --trials 8 --seed 2000 --agent-temperature 1.0 --user-temperature 0.0 --max-steps 40`
  with `TAU2_PARTITION_PATH=experiments/tau-2/data/partition_frontier_only.json` (frontier).
- If the numbers differ, check first: is the user simulator Qwen3.5-4B, is `enable_thinking` off, is the tau2 commit the same.

### E1 R1 selection (about 10–20 GPU-hours, parallelizable)

- 4B base samples each valid AReaL task (1,844) and each TRAIN task (139) 4 times (temperature 1.0, 40 steps):
  `collect_traces --bucket AREAL|TRAIN --trials 4 --seed 4000 --agent-temperature 1.0 --user-temperature 0.0 --max-steps 40` (parallelize with `--shard i/n`).
- Pick tasks with 0 < p < 1 with `scripts/select_frontier.py`, then build the GRPO parquet with `tau2_evolve.prepare_grpo_data --max-steps 40 --no-balance-domains`.
- Expected: about 546 AReaL tasks and about 36 TRAIN tasks (our result; ±10% with different seeds).

### E2 GRPO without truncation (main experiment, answers Q1)

| Item | Old setting on 32 GB | New setting |
|---|---|---|
| Response cap (whole multi-turn conversation) | 5632–6144 | **16384** (target truncation rate < 5%) |
| Prompt cap | 8192 | 8192 |
| Max turns `MAX_TURNS` | 30 | 30 (raise to 40 if truncation mostly comes from the turn cap) |
| Prompts × generations per step | 4 × 8 | **16 × 8** (or 8 × 16) |
| Learning rate | 1e-5 → 3e-5 | one run each at 1e-5 and 3e-5 |
| LoRA | rank 16, attention + MLP | same; plus one run with rank 64 or full fine-tuning (if memory allows) |
| Actor precision | `fsdp_config.model_dtype=bf16` | fp32 is fine on 80 GB |
| Importance-sampling correction | VERL one-step off-policy defaults to `bypass_mode=True`: old_log_prob is the rollout log-prob, and the PPO ratio π_θ/π_rollout is clipped (implicit correction, no explicit IS weights) | **Explicit TIS**: `bypass_mode=False` (decoupled, recomputes old_log_prob) + `rollout_is=token`, `rollout_is_threshold=2.0`, `calculate_log_probs=True`; the pipeline's `ENABLE_TIS=1` sets all four |
| Steps / checkpoints | 250 / every 50 | 200 / every 25 |

- GPU layout (8 GPUs): 1 GPU for the student evaluation server + user simulator, 4 for training, 3 for rollout, e.g.
  `TRAIN_GPU=0,1,2,3 ROLLOUT_GPU=4,5,6` (`03_grpo_only.sh` accepts comma-separated GPU lists; **multi-GPU is untested**, run a 2-step smoke test first).
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
- **PPO + critic**: VERL supports GAE; a critic the size of the student (4B) is feasible on 80 GB.
- **Process reward** (early in training, annealed to 0 later): `TAU2_ACTION_RECALL_WEIGHT` and `TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS` are implemented (action recall).

---

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

- On EVAL, with a hosted strong model as the user simulator (`COMMONSTACK_API_KEY` or another OpenAI-compatible endpoint), report pass^1,
  pass^4 and 95% CIs for the 4B base and the best checkpoint, seeds 300 + 1300.
- EVAL must not be used for any selection before that.

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
| `experiments/tau-2/SETUP_4B_RL.md`, `scripts/setup_envs.sh`, `scripts/download_assets.sh` | Environments, models and data from scratch (AWS) |
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
| `experiments/tau-2/PLAN_4B_A800.md`, `PLAN_4B_RL.md` | Plans of the previous phase and their step-by-step adjustment log |

---

## 10. Time and compute estimates (8 × 80 GB)

| Step | Estimate |
|---|---|
| Environments + data (before E0) | about 1 hour (`SETUP_4B_RL.md`) |
| E0 baseline reproduction | about 2 GPU-hours |
| E1 R1 selection | 10–20 GPU-hours (a few hours in parallel) |
| E2 two GRPO runs × 200 steps | about 1 day per run (4 training + 3 rollout GPUs); frontier evaluation about 1–2 GPU-hours per checkpoint |
| E3 | concurrent with E2 |
| E4 (if needed) | 2–3 days |
| Final VAL / EVAL | half a day (EVAL needs API budget) |

## 11. Conventions

- One configuration per `RESULTS_DIR`; never mix settings in an existing directory; resume inside the original directory.
- Never train on VAL/EVAL, and never use EVAL results for any selection.
- Draw conclusions only after reloading (or merging) the adapter; every evaluation must pass the adapter-active assertion.
- `.env`, API keys, model weights and `results/` stay out of the repository.
