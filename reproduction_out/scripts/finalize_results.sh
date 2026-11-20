#!/usr/bin/env bash
# Collect every finished run into reproduction_out/experiment_results/ using the repository's
# own statistics code (tau2_evolve.eval_stats), so no number here is computed by a different rule.
set -uo pipefail
source /home/powerop/work/yjh/MERA-Evolve/reproduction_out/env.sh
D=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro
REPO=/home/powerop/work/yjh/MERA-Evolve
OUT=$REPO/reproduction_out/experiment_results
cd "$REPO"
mkdir -p "$OUT"
export PYTHONPATH=$REPO/experiments/tau-2:$REPO/experiments/tau-2/compat
export TAU2_WORKSPACE=$TAU2_WORKSPACE
TAU2_PY=$TAU2_WORKSPACE/.venv_tau2/bin/python

STU=$D/results/e0_base_val/base_VAL_seed300.jsonl
STU2=$D/results/e0_base_val/base_VAL_seed1300.jsonl
TCH=$D/results/e6_teacher_val/teacher_VAL_seed300.jsonl
TCH2=$D/results/e6_teacher_val/teacher_VAL_seed1300.jsonl

have() { [[ -s "$1" ]] && python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$1')]
bad=sum(1 for r in rows if r.get('error'))
sys.exit(0 if len(rows)>0 and bad==0 else 1)" 2>/dev/null; }

echo "=== per-run summary (pass^k, bootstrap CI) ===" | tee "$OUT/summary.txt"
for f in "$STU" "$STU2" "$TCH" "$TCH2"; do
  [[ -s "$f" ]] || continue
  echo "--- $(basename "$f") ---" | tee -a "$OUT/summary.txt"
  "$TAU2_PY" -m tau2_evolve.eval_stats summary "$f" 2>/dev/null | tee -a "$OUT/summary.txt"
done

echo "=== repeatability between the two seeds of one policy ===" | tee -a "$OUT/summary.txt"
if have "$STU" && have "$STU2"; then
  "$TAU2_PY" -m tau2_evolve.eval_stats repeatability "$STU" "$STU2" \
    --json "$OUT/repeatability_student_VAL.json" 2>/dev/null | tee -a "$OUT/summary.txt"
fi

echo "=== paired comparison: 9B teacher (candidate) vs 4B student (baseline) ===" | tee "$OUT/compare.txt"
BASE=(); CAND=()
have "$STU"  && BASE+=("$STU")
have "$STU2" && BASE+=("$STU2")
have "$TCH"  && CAND+=("$TCH")
have "$TCH2" && CAND+=("$TCH2")
if (( ${#BASE[@]} && ${#CAND[@]} )); then
  "$TAU2_PY" -m tau2_evolve.eval_stats compare \
    --baseline "${BASE[@]}" --candidate "${CAND[@]}" \
    --json "$OUT/compare_teacher_vs_student_VAL.json" 2>/dev/null | tee -a "$OUT/compare.txt"
fi

echo "=== frontier diagnostics ===" | tee "$OUT/frontier.txt"
for f in "$D/results/e0_base_val/base_frontier.jsonl" "$D/results/e6_teacher_val/teacher_frontier.jsonl"; do
  [[ -s "$f" ]] || continue
  echo "--- $(basename "$f") ---" | tee -a "$OUT/frontier.txt"
  "$TAU2_PY" -m tau2_evolve.eval_stats summary "$f" 2>/dev/null | tee -a "$OUT/frontier.txt"
done

echo "=== raw counts ===" | tee "$OUT/counts.txt"
for f in "$STU" "$STU2" "$TCH" "$TCH2" \
         "$D/results/e0_base_val/base_frontier.jsonl" "$D/results/e6_teacher_val/teacher_frontier.jsonl"; do
  [[ -s "$f" ]] || { echo "$(basename "$f") MISSING"; continue; }
  python3 - "$f" <<'EOF' | tee -a "$OUT/counts.txt"
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
n = len(rows); p = sum(bool(r.get('passed')) for r in rows); e = sum(1 for r in rows if r.get('error'))
dom = collections.defaultdict(lambda: [0, 0])
for r in rows:
    d = dom[r.get('domain', '?')]; d[0] += 1; d[1] += bool(r.get('passed'))
print(f"{sys.argv[1].split('/')[-1]}: n={n} passed={p} ({(p/n if n else 0):.3f}) errors={e} "
      + " ".join(f"{k}={v[1]}/{v[0]}" for k, v in sorted(dom.items())))
EOF
done
echo "results written to $OUT"
