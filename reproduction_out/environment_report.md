# Environment Report — MERA-Evolve reproduction node

Run directory: `/home/powerop/work/yjh/MERA-Evolve/reproduction_out`
Heavy artifacts: `/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro`
Generated: 2026-11-17

---

## 1. Hardware as found

| Item | Value |
|---|---|
| Hostname | `ocpprocajkzjdp-tr007719-0` |
| Kernel | `5.10.134-16.4.al8.x86_64` |
| GPUs | **4 × NVIDIA H20**, 97 871 MiB (95.0 GiB) each, all free at start (`nvidia-smi --query-compute-apps` empty) |
| GPU memory total | **380 GiB** (vs the plan's target 8 × A100 40 GB = 320 GiB) |
| **Driver** | **550.90.07** |
| CPUs | 192 |
| CUDA toolkit | 13.0 (`nvcc V13.0.88`) |
| NCCL | 2.27.7 (torch-bundled) / 2.29.7 (`/usr/lib64`) |

**Driver constraint (important).** `research_plan.md` §3 and `SETUP_4B_RL.md` state that the
target node needs **driver ≥ 580** (≥ 570 for the cu128 training env, ≥ 580 for the vLLM 0.29
evaluation servers, below 535 nothing starts). This node runs **550.90.07**.

This turned out **not to be fatal in practice**: CUDA 13 works because
`LD_LIBRARY_PATH=/usr/local/cuda/compat:...` supplies the forward-compatible libcuda shim.
Measured evidence:

- `torch 2.13.0+cu130` → `torch.cuda.is_available() == True`, `device_count() == 4`
- `torch 2.14.0+cu126` → available, 4 devices
- **vLLM serves `Qwen3.5-4B` and answers requests on this node** (verified, §5)

So the driver gap is a *pinned-stack* blocker (the exact cu128/vLLM 0.29 combination cannot be
installed here), not an "impossible to run anything" blocker. Everything reported below is what
actually ran.

---

## 2. Filesystem

| Mount | Size | Avail | Used for |
|---|---|---|---|
| `/home/powerop` (nvme3n1) | 492 G | **12 G (98 % full)** | repo checkout, conda envs |
| `/` overlay | 240 G | 71 G | `/tmp` → **Python venvs, uv cache** |
| `/data/htlcrosssale-gps-share` (JuiceFS) | 1.0 P | 969 T | models, AReaL data, all results and checkpoints |

Two consequences drove layout decisions:

1. The repo's own paths for `.venv_qwen35`, `.venv_vllm`, `.deps` would have to live inside
   `/home/powerop/work/yjh/MERA-Evolve`, which has **12 GB free** — the plan's three environments
   need ~40 GB. So venvs were placed on `/tmp` (overlay) and reached through symlinks (§3).
2. JuiceFS measured **606 MB/s sequential write, 572 MB/s read**, but **~50 ms per small-file
   create** (500 files = 25 s vs 0.025 s on overlay). Therefore only bulk data and results go
   there; Python environments do not.

---

## 3. Software actually used

| Role | Environment | Key versions |
|---|---|---|
| Model load / fwd-bwd check | conda `hf-cu130-py312` | Python 3.12.13, torch 2.13.0+cu130, transformers 5.14.1, numpy 2.5.2 |
| **vLLM serving** (agent, user, teacher) | conda `fave-vllm-py311` | Python 3.11, **torch 2.14.0+cu126**, **vLLM 0.33.0** |
| **tau2-bench simulator** | `/tmp/mera_envs/router-skills-evolve/.venv_tau2` | Python 3.12.13, tau2-bench pinned `17e07b1` |
| Training (VERL) | not built — see §6 | — |

### Deviations from the pinned stack (all environment-level, no algorithm touched)

| Plan pins | Used here | Why |
|---|---|---|
| vLLM **0.29.0** in `.venv_vllm` | **vLLM 0.33.0** (pre-existing env `fave-vllm-py311`) | `download.pytorch.org` is unreachable from this node (install of a fresh 0.29 hung indefinitely); 0.33 is ≥ the `VLLM >= 0.18` that `08_phase0_eval_protocol.sh` actually requires, and it serves `Qwen3_5ForConditionalGeneration` correctly |
| torch **2.10+cu128** (training) | torch **2.14.0+cu126** (serving) / 2.13.0+cu130 (checks) | same cause; both verified working on driver 550 |
| `.venv_qwen35` inside repo | not created | see §6 |
| `.venv_vllm`, `.deps`, `.venv_qwen35` paths | symlinked to `/tmp/mera_envs/*` | 12 GB free in `/home/powerop`; the repo's `.gitignore` already ignores these names, so the repo tree stays clean |

AReaL dataset download needed `HF_HUB_DISABLE_XET=1`: the corporate HF proxy returns **404** on
`xet-read-token`. With xet disabled the download succeeded.

---

## 4. Assets verified present

| Asset | Path | Status |
|---|---|---|
| **Student / user simulator** Qwen3.5-4B | `/data/.../models/Qwen3.5-4B` | complete, 8.8 GB, 2 shards |
| **Teacher** Qwen3.5-9B | `/data/.../models/Qwen3.5-9B` | complete, 19 GB, 4 shards |
| AReaL-tau2-data | `/data/.../mera_evolve_repro/data/areal_tau2` | 1 982 RL tasks, 926 MB, 9 DB snapshots |
| `valid_task_ids.json` | produced by `check_areal_tasks.py` | **airline 1148 / retail 563 / telecom 126 = 1837** (plan expects 1148 / 563 / ~133) |
| `partition_v2.json` | repo `experiments/tau-2/data/` | TRAIN 139 / VAL 67 / EVAL 67 / DIAG 5 — matches plan §1.1 |
| `partition_frontier_only.json` | repo | TRAIN 58 tasks — matches plan §1.1 |
| tau2-bench | pinned `17e07b1da2bbc0cadfddeea36412686e0604127b` | `import tau2` OK (domains airline/retail/telecom registered) |
| router-skills-evolve | `8a2cd9ed63aa4e065a1550a0718418a54a6fed64` | OK; provides `tau2_stage2/code` and `stage2_v1/partition.json` |

**Model load verification (Phase 4 items 1–2), both on GPU:**

| Checkpoint | Params | Peak GPU | Forward/backward | Loss | Grad norm | Generate |
|---|---|---|---|---|---|---|
| Qwen3.5-4B (student) | 4.21 B | 7.9 GiB | OK | 2.6601 finite | 213.0 finite | — |
| Qwen3.5-9B (teacher) | 8.95 B | 16.7 GiB | OK | 2.2197 finite | 110.0 finite | OK |

Both share vocabulary 248 077 (relevant to any later token-level distillation check).
Both emitted the pitfall warning from `research_plan.md` §8:
`The fast path is not available because one of the required library is not installed`
— **confirming `fla` / `causal-conv1d` are absent** in the checking env (2.5× slower GDN path).

---

## 5. Serving verification (Phase 4 items 3–6, 9)

vLLM 0.33.0 serving `Qwen3.5-4B` with the protocol flags of `08_phase0_eval_protocol.sh`
(`--language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml
--default-chat-template-kwargs '{"enable_thinking": false}' --dtype bfloat16`):

- server reaches `/v1/models` in ~50–90 s from JuiceFS weights;
- greedy completion returns `OK` (thinking confirmed OFF; without the chat-template kwarg the
  model emitted `Thinking Process:`, so the kwarg is load-bearing);
- **full tau2-bench agent loop completes end-to-end**: 2 VAL airline tasks produced 28- and
  16-message multi-turn trajectories, `reward = 1.0` for task 10 and `0.0` for task 23, no errors.
  This exercises data reading, tool calling, the simulator and the official evaluator together.

Servers run inside **tmux** sessions (`mera_servers`, `teacher`); plain `setsid nohup` left
orphaned `VLLM::EngineCore` processes holding 45 GiB/GPU after the API parent was reaped.

---

## 6. Training environment status — BLOCKED

The GRPO path (E1/E2/E3, and the M/E6 student-update steps) needs **VERL 0.8 with a
Qwen3.5-capable vLLM**. Neither is satisfiable on this node:

1. `pip show verl 0.8.0` metadata requires **`vllm<=0.12.0`** and `numpy<2.0.0`. The plan's
   "VERL 0.8 + bundled vLLM 0.18.1" therefore only exists because `setup_envs.sh` force-overrides
   the constraints — a resolution that needs `download.pytorch.org`, which is unreachable here
   (the install hung with zero CPU progress until killed).
2. The one pre-existing env with the right VERL major version (`verl`, `verl 0.8.0.dev0`) ships
   **vLLM 0.11.0**, whose model registry was checked and contains **no `Qwen3_5*` architecture** —
   rollouts of the student would be impossible.
3. No reachable path to `torch 2.10+cu128` or `vllm==0.18.1` was found.

Consequence: **no multi-GPU distributed training smoke test (Phase 4 items 6–8) could be run**,
so the NCCL/4-card check, checkpoint save/resume and GRPO training experiments could not be
executed. This is reported rather than worked around: substituting a different trainer or
retraining with a different algorithm would break the "faithful reproduction" constraint.

---

## 7. Process / reliability notes

- Long jobs are run detached in **tmux** so an SSH drop cannot kill them.
- The user's instructions say don't print tokens; note that `HF_TOKEN` and an OpenAI-style key are
  present in the environment — they were not echoed into any artifact. `OPENAI_API_KEY=EMPTY` is
  passed to LiteLLM for local servers.
- No repository source file was modified. All adaptations live under `reproduction_out/`
  (scripts + symlinks), which `.gitignore` already excludes or which are untracked new files.
