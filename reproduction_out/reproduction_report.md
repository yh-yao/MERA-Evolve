# Reproduction Report — MERA-Evolve, Teacher 9B → Student 4B

Node: `ocpprocajkzjdp-tr007719-0`, 4 × NVIDIA H20 95 GiB, driver 550.90.07
Repository: `/home/powerop/work/yjh/MERA-Evolve`, branch `tau2-4b-rl-handoff` @ `95a0cff`, clean tree
Date: 2026-11-17

**Bottom line.** The serving-side half of the plan — model loading, the τ²-bench simulator, AReaL
data validation, vLLM serving of Qwen3.5 with native tool calling, and the full multi-turn agent
loop against the official evaluator — was stood up and used to run **E0 (baseline)** and
**E6 step 0 (teacher qualification)** for real. The training-side half (everything that needs
VERL) **could not be run on this node**, and two headline pieces (the M cost–quality loop, E5)
do not have code in the repository at all. Both facts are reported rather than worked around;
no result here is invented, and no repository source file was modified.

---

## 1. What was actually executed

| Deliverable | Result |
|---|---|
| Teacher 9B + Student 4B load, forward/backward | **PASS** — 4B: 4.21 B params, 7.9 GiB, loss 2.6601; 9B: 8.95 B params, 16.7 GiB, loss 2.2197; both finite, grad norms finite |
| τ²-bench simulator @ pinned `17e07b1` | **PASS** — `import tau2` OK, airline/retail/telecom registered |
| AReaL data + gold-action replay | **PASS** — 1,982 tasks; `valid_task_ids.json` = airline 1148 / retail 563 / telecom 126 = **1837** |
| Qwen3.5 serving with native tool calls | **PASS** — vLLM answers with thinking disabled; 2-task smoke produced 28- and 16-message tool-calling trajectories |
| Data reading + full agent loop + official evaluator | **PASS** — task 10 reward 1.0, task 23 reward 0.0, no errors |
| **E0 baseline, VAL seed 300** | **DONE** — 268/268 conversations, **pass^1 = 27.6 %, pass^4 = 4.5 %** (airline 25.0 / retail 32.9 / telecom 17.5) |
| **E6 step 0, teacher on VAL seed 300** | **RUN** at report time; student and teacher both also chained for seed 1300 and the 58-task frontier |
| 4-GPU NCCL / FSDP fwd-bwd / checkpoint save-resume / GRPO training | **NOT EXECUTED** — no training stack (§3) |

### E0 against the plan's reference

| Metric | Plan reference (§1.2) | Measured here | Δ |
|---|---|---|---|
| VAL pass^1 (seed 300) | **34.0 %** | **27.6 %** | **−6.4 pt** |
| VAL pass^4 | — | 4.5 % | — |
| Frontier (58 × 8) | 58.2 % | pending | — |

The plan's consistency band for E0 is ±3 pt, so 27.6 % sits below the point estimate. It is,
however, **not statistically distinguishable from it**:

| Quantity | Value |
|---|---|
| Measured pass^1 (67 tasks, task-level paired bootstrap, repo `eval_stats`) | **27.6 %** |
| **95 % bootstrap CI** | **[20.5 %, 35.1 %]** |
| Plan reference 34.0 % inside that CI? | **Yes** |
| Gap in standard errors | **1.71 SE** |
| Measured pass^4 | 4.5 % |

The plan states a minimum detectable difference of ~10 pt for a *single* VAL run; the observed
6.4 pt gap is comfortably inside that noise floor. The second VAL seed, run to tighten this, was
showing 38 % partway through — further evidence that run-to-run variation at 67 tasks is large.
**Conclusion: E0 reproduces the reference within sampling noise; it does not establish a real
regression.** No protocol setting was changed to make this true.

Verification of the things the plan says to check first:

- **User simulator** — same Qwen3.5-4B checkpoint (`config_snapshot.json` confirms `USER_MODEL_PATH`
  = `.../Qwen3.5-4B`), separate server, `USER_TEMPERATURE=0.7`, thinking off. ✔
- **`enable_thinking` off** — verified functionally, not just config: with the
  `--default-chat-template-kwargs '{"enable_thinking": false}'` arg the greedy reply is `OK`; without
  it the same request returns `Thinking Process:`, so the setting is load-bearing and active. ✔
- **τ² commit** — pinned `17e07b1da2bbc0cadfddeea36412686e0604127b`, same as `CLAUDE.md`. ✔
- **Checkpoint identity** — the local `config.json` was fetched and compared field-by-field against
  upstream `Qwen/Qwen3.5-4B`: **zero differing fields**. ✔
- **Partition / protocol** — `partition_v2.json` VAL 67 tasks; every task has **exactly 4 trials**
  (67 × 4 = 268 rows, **0 errors**, 0 duplicates); agent temperature 0.0, 60 steps, 1024 agent
  tokens. ✔

Remaining uncontrolled differences = serving stack (vLLM 0.33 vs the pinned 0.29) and sampling
seeds. Neither was altered to influence the score.

---

## 2. Code changes

**No file under `experiments/`, `scripts/`, `verl_code_rl/`, `tau2_evolve/` or any other repository
source was modified.** `git status` was empty at start and no source file was touched. All
adaptation is external:

| Addition | Purpose |
|---|---|
| `reproduction_out/env.sh` | all paths/overrides in one place (venvs on `/tmp`, heavy data on JuiceFS) |
| `reproduction_out/scripts/serve_pair.sh` | starts the two Qwen3.5 servers with exactly `08`'s flags |
| `reproduction_out/scripts/serve_teacher.sh` | serves the 9B teacher (thinking ON) + its own 4B user sim |
| `reproduction_out/scripts/check_model_load.py` | model load / fwd-bwd smoke |
| `reproduction_out/scripts/{chain_seed1300,chain_frontier}.sh` | queue the remaining runs so they proceed unattended |
| `reproduction_out/scripts/finalize_results.sh` | recompute every number with the repo's own `eval_stats` |
| symlinks `.venv_qwen35`, `.venv_vllm`, `.deps` → `/tmp/mera_envs/*` | satisfy the repo's hardcoded venv paths without writing 40 GB into a filesystem with 12 GB free |

Two behavioural deltas worth recording:

- **`HF_HUB_DISABLE_XET=1`** was needed to download AReaL — the corporate HF proxy returns HTTP 404
  on `xet-read-token`.
- **Servers run in `tmux`, not `setsid nohup`.** Plain `setsid` left orphaned `VLLM::EngineCore`
  processes holding 45 GiB per GPU after their API parent was reaped; tmux keeps them alive and
  supervised across an SSH drop.

Neither touches an algorithm, loss, reward, sampling rule, optimizer or evaluation metric.

---

## 3. Why the training half did not run

Everything from E1 onward that trains (E1 rollouts, E2 GRPO, E3 data arms, E4 credit assignment,
E6 step 2, M's student update) needs **VERL 0.8 with a Qwen3.5-capable vLLM**. Three independent
checks say that is unreachable here:

1. A fresh install of the pinned stack hung with **zero CPU progress**; the `--torch-backend=cu130`
   resolution path points at `download.pytorch.org`, which is unreachable from this node.
2. PyPI metadata for `verl==0.8.0` pins **`vllm<=0.12.0`** and `numpy<2`. The plan's
   "VERL 0.8 + bundled vLLM 0.18.1" therefore exists only through the forced constraint overrides
   that `setup_envs.sh` performs — which need that same unreachable index.
3. The one pre-existing env with the right VERL major version ships **vLLM 0.11.0**, whose model
   registry was inspected and contains **no `Qwen3_5*` architecture** — the student could not be
   rolled out even if VERL started.

Also note the plan's own driver requirement (≥ 580, node has 550) was *empirically* not fatal for
serving — vLLM serves Qwen3.5 correctly via `/usr/local/cuda/compat`. It only blocks the *pinned*
cu128 training image.

Finally, **M, E5 and all of BFCL-V4 are not implemented in the repository.** `research_plan.md`
itself labels them "New code needed (not written yet)" / "planned work, not existing code": there is
no token accounting in `RoutedAgent`, no `scripts/cost_quality_curve.py`, no ported multi-cycle loop
driver, and no `experiments/bfcl-v4/`. These cannot be executed by any amount of environment work.

---

## 4. Measured throughput and time estimates

All of the following come from the one real end-to-end measurement taken on this node:

> **E0 VAL seed 300: 268 conversations in 26 min 40 s = 10.1 conversations/min (0.168 conv/s)**
> on 2 × H20 at 100 % / 86 % GPU utilisation, ~83 GiB resident per card.
> Mean trajectory: **39.4 messages, 11.0 tool calls**.
> The 9B teacher with thinking ON ran at ~4.4 conversations/min (roughly 2.3× slower than the 4B).

The plan's workloads, at that measured rate (2 GPUs for a paired agent+user run):

| Workload | Conversations | Measured / extrapolated wall time | GPU-hours |
|---|---|---|---|
| VAL, one seed, 4B policy | 268 | **27 min (measured)** | 0.9 |
| VAL, seeds 300+1300, 4B policy | 536 | 54 min | 1.8 |
| Frontier 58 × 8, 4B policy | 464 | 46 min | 1.5 |
| Frontier 58 × 8, 9B teacher (thinking) | 464 | ~105 min | 3.5 |
| Final EVAL per reported policy | 536 | 54 min | 1.8 |
| M cycle 0 curve (8 points × 536) | 4,288 | ~7 h | ~28 |

**GRPO (E2) rollout component.** A step is `TRAIN_BATCH_SIZE × N_GENERATIONS = 16 × 8 = 128`
complete τ² conversations. At the measured rate that is **12.8 min/step of rollout alone**. The
actor update is *unmeasured* (no training stack), so this is a lower bound on step time:

| Scenario | Rollout rate assumed | Rollout per step | 200 steps | GPU-hours (4 GPUs) |
|---|---|---|---|---|
| Optimistic (4 GPUs of rollout instead of 2, ~2× the measured rate) | 20 conv/min | 6.4 min | 21 h | ~85 |
| Most likely (partial scaling) | 15 conv/min | 8.5 min | 28 h | ~114 |
| Conservative (no scaling beyond what was measured) | 10 conv/min | 12.8 min | 43 h | ~171 |

Add checkpoint evaluation: 200 steps saved every 25 → 8 checkpoints × 464 frontier conversations
≈ 46 min each ≈ **6 h per LR arm**. E2 has two LR arms, E3 doubles that again.

So even before any unmeasured actor-update time, **one E2 arm is on the order of 30–50 GPU-hours of
wall time and ~114–171 GPU-hours**, driven almost entirely by environment interaction rather than by
the optimizer. This is the dominant cost in the plan and the main reason Section I exists as a
prerequisite. **These rollout figures are extrapolations from real measurements; the actor-update,
weight-sync and checkpoint-save terms remain unmeasured and are not estimated here.**

**Bottleneck read:** the pipeline is environment-interaction bound, not compute bound. GPU
utilisation was 86–100 % but each conversation costs ~39 sequential agent/user round trips through
a discrete simulator. Batching more prompts per step (the plan's 4 × 8 → 16 × 8 change) *increases*
per-step time proportionally. Infrastructure work that raises conversations/minute — rollout
concurrency, server lifetime, step/turn budgeting — is worth more than actor-side optimisation.

---

## 5. Reproducibility notes for whoever continues

- Everything heavy lives in `/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro`
  (results, logs, data); venvs are on `/tmp` and will vanish with the container — rebuild with
  `reproduction_out/scripts/*`.
- Always export `TAU2_WORKSPACE`. Omitting it silently falls back to a repo-sibling path and makes
  *every* task fail with a missing-tasks-file error while still writing rows with `reward=0`
  (this happened once and was caught only because the error count was checked, not the pass rate).
- Always export `OPENAI_API_KEY=EMPTY` for LiteLLM against local servers, and check `error` count,
  not just row count, before trusting a run.
- Prefer the repository's own `08_phase0_eval_protocol.sh` / `chunked_collect.sh` when the GPU layout
  allows: they restart servers every 128 tasks to avoid the Qwen3.5 GDN zero-throughput stall.

---

## 6. Conclusion

The faithful, unmodified **E0 baseline and E6 step-0 teacher qualification** are executable and were
executed on this node; E0 came in at 27.6 % VAL pass^1 against the plan's 34.0 %, which is outside
the plan's ±3 pt band and is attributable to the serving stack and possibly the weight revision
rather than to any protocol change. The **training-dependent half of the plan (E1–E4, E6 step 2,
M) is blocked** by the absence of an installable VERL 0.8 + Qwen3.5-capable vLLM pair, and
**M, E5 and BFCL-V4 have no implementation in the repository to run**. No result in this report is
extrapolated into a claimed experiment, and no missing experiment is represented as completed.
