#!/usr/bin/env bash
set -euo pipefail

# Isolates the SkillBook's own effect on tau2-bench: no training of any kind.
#   1. collect oracle traces on the 97-task TRAIN split (airline+retail+
#      telecom_small; see tau2_evolve/benchmark.py for the telecom split.
#      task_ids differ from tau2_stage2's original partition.json).
#   2. build a domain-bucketed skillbook (airline/retail/telecom) from those
#      traces' successful exemplars.
#   3. baseline eval (no skill) vs skills eval (skill text appended to the
#      agent's domain policy) on the same 35-task EVAL split.
#
# Prerequisites: agent + user-simulator OpenAI-compatible vLLM servers already
# running (plain Qwen2.5-Instruct, NOT -Coder -- see docs/experiments/ writeup
# for why). Agent needs --enable-auto-tool-choice --tool-call-parser hermes;
# user-simulator needs the same too (telecom is dual-control: the user side
# also makes tool calls).
#
# Usage: bash experiments/tau-2/01_skills_only.sh

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
export PYTHONPATH="$EXPERIMENT_DIR:${PYTHONPATH:-}"

TAU2_PYTHON="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
if [[ ! -x "$TAU2_PYTHON" ]]; then
  echo "FATAL: tau2 Python not found at $TAU2_PYTHON; set TAU2_WORKSPACE or TAU2_PYTHON" >&2
  exit 2
fi
AGENT_MODEL="${AGENT_MODEL:-openai/evol-llm-agent}"
AGENT_BASE_URL="${AGENT_BASE_URL:-http://127.0.0.1:8200/v1}"
USER_MODEL="${USER_MODEL:-openai/evol-llm-user}"
USER_BASE_URL="${USER_BASE_URL:-http://127.0.0.1:8201/v1}"
WORKERS="${WORKERS:-6}"
MAX_STEPS="${MAX_STEPS:-60}"
DISTILLER_MODEL="${DISTILLER_MODEL:-openai/gpt-5.5}"
DISTILLER_BASE_URL="${DISTILLER_BASE_URL:-https://api.commonstack.ai/v1}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/tau2_skills_only}"

mkdir -p "$RESULTS_DIR"
echo "[skills-only] results dir: $RESULTS_DIR"

echo
echo "[skills-only] step 1: collect TRAIN traces (97 tasks)"
"$TAU2_PYTHON" -m tau2_evolve.collect_traces \
  --bucket TRAIN --workers "$WORKERS" --max-steps "$MAX_STEPS" \
  --agent-model "$AGENT_MODEL" --agent-base-url "$AGENT_BASE_URL" \
  --user-model "$USER_MODEL" --user-base-url "$USER_BASE_URL" \
  --out "$RESULTS_DIR/train_traces.jsonl"

echo
echo "[skills-only] step 2: build domain-bucketed skillbook"
"$TAU2_PYTHON" -m tau2_evolve.build_skillbook \
  --traces "$RESULTS_DIR/train_traces.jsonl" \
  --output "$RESULTS_DIR/skillbook.json" \
  --distiller-model "$DISTILLER_MODEL" --distiller-base-url "$DISTILLER_BASE_URL"

echo
echo "[skills-only] step 3: baseline eval (no skill) on 35-task EVAL split"
"$TAU2_PYTHON" -m tau2_evolve.collect_traces \
  --bucket EVAL --workers "$WORKERS" --max-steps "$MAX_STEPS" \
  --agent-model "$AGENT_MODEL" --agent-base-url "$AGENT_BASE_URL" \
  --user-model "$USER_MODEL" --user-base-url "$USER_BASE_URL" \
  --out "$RESULTS_DIR/eval_baseline.jsonl"

echo
echo "[skills-only] step 4: skills eval (skill text appended to domain policy)"
"$TAU2_PYTHON" -m tau2_evolve.collect_traces \
  --bucket EVAL --workers "$WORKERS" --max-steps "$MAX_STEPS" \
  --agent-model "$AGENT_MODEL" --agent-base-url "$AGENT_BASE_URL" \
  --user-model "$USER_MODEL" --user-base-url "$USER_BASE_URL" \
  --skillbook "$RESULTS_DIR/skillbook.json" \
  --out "$RESULTS_DIR/eval_skills.jsonl"

echo
"$TAU2_PYTHON" - <<PYEOF
import json
from collections import defaultdict

def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r["passed"])
    total_passed = sum(r["passed"] for r in rows)
    print(f"  total: {total_passed}/{len(rows)} = {total_passed/len(rows):.1%}")
    for d, vals in sorted(by_domain.items()):
        print(f"  {d}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.1%}")

print("[skills-only] baseline:")
summarize("$RESULTS_DIR/eval_baseline.jsonl")
print("[skills-only] with skills:")
summarize("$RESULTS_DIR/eval_skills.jsonl")
PYEOF

echo
echo "[skills-only] done. results in $RESULTS_DIR"
