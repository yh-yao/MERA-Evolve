# Reproduction Audit — MERA-Evolve, Teacher 9B → Student 4B

Repository: `/home/powerop/work/yjh/MERA-Evolve`
Branch/commit: `tau2-4b-rl-handoff` @ `95a0cff86f6c9f59e47ff01521572913c67c26b3` (clean tree, `git status --porcelain` empty, no local modifications before this run)
Date: 2026-11-17
Authoritative design source: `research_plan.md` (Sections 0–11 = TAU-2, Section 12 = BFCL-V4)

---

## 1. What the plan studies

MERA-Evolve trains small agents in a closed loop: collect trajectories → obtain verified stronger
demonstrations for failures → distill skills → update a LoRA adapter with SFT/GRPO → reload and
evaluate → optionally train a router that escalates hard states to a stronger model.

**Target configuration of this handoff (research_plan.md §target):**
one node with **8 × A100 40 GB**, **Qwen3.5-4B student**, **frozen Qwen3.5-9B candidate teacher**,
plus a **separate frozen Qwen3.5-4B user simulator**. The 9B teacher is a *candidate* whose
advantage must be measured — parameter count alone does not establish headroom.

### Experiment flow (as designed)

```text
E0  baseline (4B on VAL seed300 + frontier 58x8)
 │
 ▼
E6 step 0 ── teacher 9B has verified advantage over 4B? ── no ─► stop teacher loop
 │ yes                                                            │
 ▼                                                                ▼
E1  R1 rollouts (4 trials x 1,844 AReaL + 139 TRAIN)          E2 GRPO without truncation
 │                                                            E4 process reward / credit assignment
 ▼                                                            E5 gold-action-guided self-rollouts
M   full MERA loop, K=3 → 4 cycles, cost–quality curve on VAL  (component experiments)
     └─ ablations: routing only / no SkillBook / GRPO-only vs corrections+GRPO /
                   1 cycle vs K cycles / learned router vs random:P
 │
 ▼
Q4  final report: 4B base and best checkpoint on EVAL, seeds 300+1300, once
```

---

## 2. Model configuration

| Role | Checkpoint | Thinking | Frozen | Verified here |
|---|---|---|---|---|
| Student (agent under test) | `Qwen/Qwen3.5-4B` (post-trained, not `-Base`) | OFF | no — updated by LoRA | loaded, 4.21 B, 7.9 GiB, fwd/bwd finite |
| User simulator (part of environment) | `Qwen/Qwen3.5-4B` **base**, separate role | OFF | **yes, always** | same weights served |
| Teacher (candidate) | `Qwen/Qwen3.5-9B` post-trained | **ON** | **yes, after qualification** | loaded, 8.95 B, 16.7 GiB, fwd/bwd finite |

Neither student adapters nor SkillBooks may modify the user simulator or the teacher.
Both checkpoints share vocabulary 248 077.

LoRA (plan §E2): rank 16, alpha 32, targets attention **+ MLP** (`q,k,v,o,gate,up,down`) — MLP is
required because under Qwen3.5's hybrid architecture attention-only LoRA trains only 6 of 24 layers.
Actor precision bf16, micro-batch 1.

---

## 3. Experiment matrix

| ID | Question | Method | Workload | Status on this node |
|---|---|---|---|---|
| **E0** | reproduce the baseline | 4B base, VAL seed 300 (67×4) + frontier (58×8) | 268 + 464 conversations | **RUN** (VAL complete, frontier pending) |
| **E1** | select GRPO prompts | 4B base samples each valid AReaL (1,837) + TRAIN (139) task 4×, keep 0<p<1 | ~7,900 conversations | partially: `valid_task_ids.json` built (1837); rollouts pending compute |
| **I** | infra profiling | bf16, micro-batch 1, actor/rollout splits | 10–20 step benchmark | **BLOCKED** — no training stack (see §6) |
| **E2** | Q1: GRPO without truncation | 16 prompts × 8 gens, response cap 16384, lr 1e-5 and 3e-5, 200 steps, save/25, explicit TIS | 200 steps + 464 conv/checkpoint × 2 LRs | **BLOCKED** — needs VERL 0.8 |
| **E3** | Q2: data comparison | G1 (AReaL+TRAIN) vs G2 (TRAIN only), same settings as E2 | 2 × E2 | **BLOCKED** — depends on E2 |
| **E4** | Q3: credit assignment | action-recall process reward `TAU2_ACTION_RECALL_WEIGHT`, then PPO+critic | 1 GRPO run | **BLOCKED** — depends on E2 (and it is conditional) |
| **E5** | Q5 without a teacher | gold-action hint wrapper during rollouts; train on hint-stripped successes | rollouts + SFT/GRPO | **NOT IMPLEMENTED** — plan §E5 "Implementation (not written yet)" |
| **E6 s0** | Q5: teacher headroom | 9B vs 4B on VAL seeds 300+1300, same frozen 4B user sim, paired CI must exclude 0 | 536 conv/policy | **RUN** (VAL seed 300) |
| **E6 s1** | teacher data | error-point corrections + full demonstrations in the real tau2 env | depends on failure count | **BLOCKED** — gated by E6 s0 and needs trained student |
| **E6 s2** | offline distillation | group F (SFT) then group G (F + GRPO) | 1 SFT + 1 GRPO run | **BLOCKED** — needs training stack |
| **M** | Q0: does the loop move the cost–quality curve? | router + escalation, cost = teacher completion tokens incl. reasoning | 4,288 conv/cycle for the curve | **NOT IMPLEMENTED** — plan §M lists 3 pieces of new code not yet written |
| **B0–B4** | BFCL-V4 | second benchmark | — | **NOT IMPLEMENTED** — plan §12 is design only; `experiments/bfcl-v4/` does not exist |

### Baselines defined by the plan

| Baseline | Purpose |
|---|---|
| Always-student (0 teacher tokens) | left endpoint of the cost–quality curve |
| Always-teacher (cost% = 100) | right endpoint |
| Cycle 0 (fixed base student + learned router) | "route around a fixed student" — the prior-work setting |
| `random:P` per-turn escalation, P ∈ {0.1, 0.3, 0.5} | cost-blind routing baseline |
| `tool_error` routing (`tau2_evolve/turn_router.py`) | heuristic baseline |
| 4B base pass^1 itself | reference numbers 34.0 % VAL / 58.2 % frontier |
| Qwen3.8-27B reference | earlier teacher for comparison |

### Ablations defined by the plan

routing only (no student update) · no SkillBook · GRPO-only (E2) vs teacher-corrections+GRPO (E6) ·
one cycle vs K cycles · learned router vs `random:P` at matched cost · LoRA rank 64 (capacity,
after memory profiling) · E3 G1 vs G2 data arms · E5 hint dropout 50 % variant.

---

## 4. Data and metrics

**Split (fixed, `experiments/tau-2/data/partition_v2.json`, verified):** TRAIN 139 / VAL 67 / EVAL 67 / DIAG 5.
Frontier: `partition_frontier_only.json`, 58 TRAIN tasks (retail 30 / airline 22 / telecom 6).
EVAL is reported **once, at the end**; no selection, no router threshold fitting on it.

**AReaL:** `inclusionAI/AReaL-tau2-data` (Apache-2.0), 1,982 RL tasks + 9 DB snapshots. Replay
validation wrote `valid_task_ids.json`: **airline 1148 / retail 563 / telecom 126 = 1837**
(plan expects 1148 / 563 / "about 133"; telecom is within the stated ± range and telecom is known
to be broken for every local model).

**Metrics** (`tau2_evolve/eval_stats.py`): pass^1 and pass^4, per-task **paired bootstrap 95 % CI**
and McNemar tests. Minimum detectable difference ≈ 10 pt for one VAL run (seed 300), ≈ 7 pt for
seeds 300+1300. Frontier is more sensitive (~5 pt) but overlaps the training prompts, so it is a
"did it learn anything" gate only — **conclusions come from VAL**.
Primary loop metric: teacher-generated tokens (including reasoning) per conversation, normalised
as `cost% = policy teacher tokens / always-teacher teacher tokens`.

**Protocol held fixed:** agent greedy (temperature 0), user simulator temperature 0.7 with per-trial
seeds, non-thinking for student and user, 1024 agent tokens/turn, 60 steps, 4 trials per task.

---

## 5. Differences between `research_plan.md` and the code

Recorded, **not** silently resolved. Where both could satisfy the plan, the repository behaviour was
kept because the plan's reference numbers were produced by that code.

| # | Plan says | Code does | Handling here |
|---|---|---|---|
| 1 | Serving should use `--tool-call-parser qwen3_coder` (§Serving) | `08_phase0_eval_protocol.sh` defaults to `"${TOOL_CALL_PARSER:-qwen3_xml}"`; `run_cycle_pipeline.sh`, `05`, `07` hardcode `qwen3_xml`; `TOOL_CALL_PARSER` is not part of the pipeline's exported infra variables | used the **code default `qwen3_xml`** (verified to produce working tool calls) so the historical VAL/frontier numbers are comparable; the parser is configurable so either arm can be run |
| 2 | `--reasoning-parser qwen3` alongside the serving parsers | absent from `08` | added `--reasoning-parser qwen3` **only for the 9B teacher** (thinking ON), where the plan requires it |
| 3 | Driver ≥ 580 required, below 535 nothing starts | node has **550.90.07** | measured: vLLM serves Qwen3.5 fine via `/usr/local/cuda/compat`. The pinned cu128 + vLLM 0.29 stack is still unreachable (§6) |
| 4 | Node is 8 × A100 40 GB | node is **4 × H20 95 GB** | GPU-affinity variables (`SERVE_GPU`, `TRAIN_GPUS`, `ROLLOUT_GPUS`, `EVAL_GPUS`) reassigned; no code change |
| 5 | "VERL 0.8 + bundled vLLM 0.18.1" | PyPI `verl 0.8.0` requires **`vllm<=0.12.0`** and `numpy<2` — the plan's pairing only exists through `setup_envs.sh`'s forced constraint overrides | documented; could not be reproduced here |
| 6 | `MAX_TURNS=30`, "raise to 40 if truncation comes mostly from the turn cap" | the collection protocol uses `--max-steps 60` for VAL and 40 for frontier | kept code values |
| 7 | τ² telecom validity "about 133/267 pass" | measured **126** | reported as measured |
| 8 | Sections **M**, **E5** and **12 (BFCL-V4)** describe the headline result | plan itself marks the needed code as *not written yet*; no `experiments/bfcl-v4/`, no `scripts/cost_quality_curve.py`, no token accounting in `RoutedAgent` | **not implemented**; reported as blocked rather than substituted |
| 9 | `10_4b_rl_pipeline.sh` is the one-command workflow (§5.0) | it hardcodes the pipeline's own collect flags, `GPU_MEMORY_UTILIZATION=0.42`, `VLLM_MAX_NUM_SEQS=128`, base VAL seed 300, frontier seed 2000, r1 seed 4000 | used it where it applies; ran `08`/`collect_traces` directly where the fixed GPU layout of §5.0 could not apply |

I made **no change to any repository source file**. All adaptation lives in `reproduction_out/`
helper scripts and in environment variables.

---

## 6. What could and could not actually be run

**Run:** model loading for both checkpoints; tau2-bench simulator; AReaL data + replay validation;
vLLM serving of Qwen3.5 including native tool calling; the complete multi-turn agent loop against
the official evaluator; **E0 baseline on VAL**; **E6 step 0 teacher qualification on VAL**.

**Not run — training stack unavailable.** Everything from E1 onward that requires VERL (E1 rollouts,
E2, E3, E4, M's student-update step, E6 step 2) needs VERL 0.8 paired with a Qwen3.5-capable vLLM.
Three independent checks show this is not reachable here:

1. A fresh install of the pinned stack hung with **zero CPU progress**; `--torch-backend=cu130`
   points at `download.pytorch.org`, which is unreachable from this node.
2. PyPI metadata for `verl==0.8.0` pins `vllm<=0.12.0`, so the plan's pair exists only under the
   forced overrides that `setup_envs.sh` performs — which need that same unreachable index.
3. The only pre-existing env with the right VERL major version ships vLLM 0.11.0, whose registered
   architectures were checked and contain **no `Qwen3_5*` entry** — the student cannot be rolled out.

Therefore the Phase-4 items that depend on distributed training (4-card NCCL, fwd/bwd under FSDP,
checkpoint save/resume, multi-GPU GRPO) **were not executed**, and the training-time throughput
estimates required by §5 of the task cannot be measured rather than guessed. Serving-time
measurement *is* available and is reported instead.
