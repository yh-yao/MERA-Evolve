#!/usr/bin/env bash
# research_plan.md Step 0 analysis: paired comparison of the 9B teacher against the 4B base
# on VAL seeds 300 + 1300. Error rows (empty assistant message / context overflow) are
# excluded first, as the plan requires infra failures to be recorded separately.
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh

D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
OUT=$D/results/e6_teacher_val/clean
PY="$TAU2_WORKSPACE/.venv_tau2/bin/python"
mkdir -p "$OUT"

python3 - "$D/results/e0_base_val" "$D/results/e6_teacher_val" "$OUT" <<'PY'
import json, os, sys
base_dir, teach_dir, out = sys.argv[1:4]

def clean(src, dst):
    keep = total = 0
    with open(src) as fh, open(dst, "w") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if any(isinstance(row.get(k), str) for k in ("error", "status", "exception")):
                continue
            out.write(line + "\n"); keep += 1
    print(f"{os.path.basename(dst):46s} kept {keep}/{total} (dropped {total-keep} infra errors)")

for seed in (300, 1300):
    clean(os.path.join(base_dir, f"base_VAL_seed{seed}.jsonl"), os.path.join(out, f"base_VAL_seed{seed}.jsonl"))
    clean(os.path.join(teach_dir, f"teacher_VAL_seed{seed}_thinkon.jsonl"), os.path.join(out, f"teacher_VAL_seed{seed}.jsonl"))
PY

export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
export TAU2_PARTITION_PATH=$REPO/experiments/tau-2/data/partition_v2.json
cd "$REPO"

echo "=== overall pass (teacher vs base, seeds 300 + 1300) ==="
"$PY" -m tau2_evolve.eval_stats summary "$OUT"/base_VAL_seed300.jsonl "$OUT"/base_VAL_seed1300.jsonl \
  "$OUT"/teacher_VAL_seed300.jsonl "$OUT"/teacher_VAL_seed1300.jsonl 2>&1 | tail -12

echo "=== paired comparison (baseline=4B base, candidate=9B teacher) ==="
"$PY" -m tau2_evolve.eval_stats compare \
  --baseline "$OUT"/base_VAL_seed300.jsonl "$OUT"/base_VAL_seed1300.jsonl \
  --candidate "$OUT"/teacher_VAL_seed300.jsonl "$OUT"/teacher_VAL_seed1300.jsonl \
  --json "$OUT"/teacher_vs_base.json 2>&1 | tail -20

echo "=== per-task paired comparison (pass bar = +5 pt) ==="
python3 "$REPO/experiments/tau-2/scripts/compare_tasks.py" \
  --base "$OUT"/base_VAL_seed300.jsonl \
  --cand "$OUT"/teacher_VAL_seed300.jsonl \
  --json "$OUT"/teacher_vs_base_tasks.json 2>&1 | tail -15
