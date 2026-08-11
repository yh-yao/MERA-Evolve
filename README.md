# MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale

Official code for the MERA paper. MERA improves a *small* model from shared,
verifier-grounded traces: each cycle mines failed invocations for
execution-verified teacher demonstrations, updates a SkillBook, trains a
student LoRA (SFT, optionally GRPO), and deploys the student behind a
cost-calibrated router with verifier-backed fallback.

| Setting | Main result |
|---|---|
| HumanEval+MBPP (582 held-out, 3 seeds) | Qwen2.5-Coder-1.5B: **28.7% → 49.7%** direct pass; deployed **88.3%** at **60.8%** always-Luna cost |
| TAU-2 (35 held-out) | Qwen3.5-2B GRPO: **14/35 → 18/35**, matching unadapted 4B |

Detailed machine setup, resume, and monitoring notes are in [`CLAUDE.md`](CLAUDE.md).
Recipe index: [`experiments/README.md`](experiments/README.md).

## Quick start

```bash
git clone https://github.com/yh-yao/MERA-Evolve.git
cd MERA-Evolve
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
pip install verl   # match your torch/CUDA stack

cat > .env <<'EOF'
COMMONSTACK_API_KEY=replace-me   # only if using a hosted teacher/fallback
EOF
chmod 600 .env

pytest -q
scripts/run_experiment.sh --list
```

Prerequisites: Linux, Bash 4+, Python ≥ 3.10, NVIDIA GPUs for serving + training.

## Reproduce: HumanEval + MBPP (paper main result)

Paper setup: 546 train / 582 held-out tasks, Qwen2.5-Coder-1.5B student,
GPT-5.6 Luna (or another OpenAI-compatible large model) as teacher/fallback,
four cycles, three seeds, matched SFT vs SFT+GRPO.

### 1. Serve student and large models

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct GPU=0 PORT=8000 \
  scripts/serve_vllm.sh
# Large teacher/fallback: local 3B, or point LARGE_* at an OpenAI-compatible Luna endpoint.
MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct GPU=1 PORT=8001 \
  scripts/serve_vllm.sh
```

### 2. Run the four-cycle loops

```bash
# Matched SFT-only control
SFT_GPU=2 SMALL_RELOAD_GPU=0 N_CYCLES=4 \
  scripts/run_experiment.sh humaneval_mbpp 4cycle-sft

# Paper SFT+GRPO arm
GRPO_GPU=2 SMALL_RELOAD_GPU=0 N_CYCLES=4 \
  scripts/run_experiment.sh humaneval_mbpp 4cycle-sft-grpo
```

Repeat with different `SEED` / `RESULTS_DIR` for three seeds. Useful overrides:
`LARGE_MODEL`, `LARGE_BASE_URL`, `SMALL_BASE_URL`, `TRAIN_LIMIT`, `EVAL_LIMIT`,
`WORKERS`.

### 3. What to report

Per cycle under `RESULTS_DIR/`:

- direct SLM pass (no router / no fallback)
- cascade pass + normalized cost vs always-large (1:10 small:large)
- final-cycle comparison vs always-small / always-large / RouteLLM-style /
  FrugalGPT-style baselines (see rebuttal notes under `rebuttal/` when present)

Orchestration smoke test (no GPU training):

```bash
bash scripts/run_full_pipeline.sh --mock --limit 8 --experiment-name smoke_evolve
```

## Reproduce: TAU-2 (strict adapter ablation)

Paper claim: same 35-task split, official environment outcome, no
SkillBook/router/fallback at eval; only base Qwen3.5-2B vs VERL-GRPO LoRA.
Unadapted Qwen3.5-4B is a descriptive reference.

TAU-2 needs a separate tau2 simulator env (default sibling checkout
`../router-skills-evolve`) plus a Qwen3.5 VERL training env. Bootstrap details:
[`experiments/tau-2/README.md`](experiments/tau-2/README.md) and `CLAUDE.md`.

```bash
export TAU2_WORKSPACE=/path/to/router-skills-evolve
export TRAIN_VENV=$PWD/.venv_qwen35
export BASE_MODEL=Qwen/Qwen3.5-2B
export USER_MODEL_PATH=Qwen/Qwen3.5-4B
export AGENT_GPU=0 AGENT_PORT=8260
export USER_GPU=1 USER_PORT=8261
export TRAIN_GPU=2 ROLLOUT_GPU=3
export RESULTS_DIR=$PWD/results/tau2_qwen35_$(date -u +%Y%m%d_%H%M%S)

scripts/run_experiment.sh tau2 4cycle
```

4B baseline only:

```bash
scripts/run_experiment.sh tau2 4b-baseline
```

## Repository layout

```text
scripts/run_experiment.sh     unified recipe launcher
scripts/run_full_pipeline.sh  HumanEval/MBPP closed loop
scripts/{serve,stop}_vllm.sh  model servers
experiments/humaneval_mbpp/   code recipes
experiments/tau-2/            TAU-2 adapter + recipes
verl_code_rl/                 shared code data/reward/skills/router
routerevolving-3/             paper LaTeX
data/raw/                     HumanEval + MBPP
results/                      run artifacts (gitignored)
```

```bash
scripts/run_experiment.sh --list
```

## Citation

```bibtex
@misc{mera2026,
  title        = {MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale},
  author       = {Yao, Yuhang and Wang, Zeyu and Chen, Wanyi and Yang, Tongyun and Han, Yuhang and Xiao, Jie and Bao, Chengke and Zhao, Tianyi and Ai, Lynn and Yang, Eric and Shi, Tianyu},
  year         = {2026},
  howpublished = {\url{https://github.com/yh-yao/MERA-Evolve}}
}
```

## Contact

- yuhangyao8@gmail.com
- tianyu@gradient.network
