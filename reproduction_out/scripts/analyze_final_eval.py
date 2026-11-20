#!/usr/bin/env python3
"""research_plan.md §7: final EVAL report — pass^1 and pass^4 with 95% CIs per policy.

The pipeline's `eval_stats summary` prints pass^k without CIs, and `compare` gives a paired CI
only for pass^1. The plan asks for pass^1, pass^4 *and* 95% CIs, so this script bootstraps over
tasks for both statistics (task-level resampling, the same unit the plan uses for curve CIs)
using the repository's load_runs/pass_hat_k and its paired_bootstrap_ci for the difference.

EVAL = 67 tasks x 4 trials x 2 seeds = 8 trials/task, 536 conversations per policy.
This script only reports; nothing here may be used for selection.

Usage: python3 analyze_final_eval.py [--root RESULTS/final_eval] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path("/home/powerop/work/yjh/MERA-Evolve")
sys.path.insert(0, str(REPO / "experiments" / "tau-2"))
from tau2_evolve.eval_stats import compare, load_runs, pass_hat_k  # noqa: E402

D = Path("/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro")


def policy_runs(root: Path, label: str):
    fs = [root / label / f"seed{s}" / f"{label}_EVAL_seed{s}.jsonl" for s in (300, 1300)]
    if not all(f.exists() and f.stat().st_size > 0 for f in fs):
        return None
    return load_runs(fs)


def boot_mean_ci(values: list[float], n_boot: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def report(runs: dict) -> dict:
    n = {len(v) for v in runs.values()}.pop()
    p1 = [sum(v.values()) / n for v in runs.values()]
    p4 = [pass_hat_k(n, sum(v.values()), 4) for v in runs.values()]
    lo1, hi1 = boot_mean_ci(p1)
    lo4, hi4 = boot_mean_ci(p4)
    return {
        "n_tasks": len(runs), "n_trials": n,
        "pass^1": sum(p1) / len(p1), "pass^1_ci": [lo1, hi1],
        "pass^4": sum(p4) / len(p4), "pass^4_ci": [lo4, hi4],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=D / "results" / "final_eval")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out, runs = {}, {}
    for label in ("base", "sft"):
        r = policy_runs(args.root, label)
        if r is None:
            print(f"{label}: EVAL files missing")
            continue
        runs[label] = r
        s = report(r)
        out[label] = s
        print(f"{label:5s} EVAL n={s['n_tasks']} trials/task={s['n_trials']}  "
              f"pass^1={100*s['pass^1']:.1f}% CI[{100*s['pass^1_ci'][0]:.1f},{100*s['pass^1_ci'][1]:.1f}]  "
              f"pass^4={100*s['pass^4']:.1f}% CI[{100*s['pass^4_ci'][0]:.1f},{100*s['pass^4_ci'][1]:.1f}]")

    if len(runs) == 2:
        c = compare(runs["base"], runs["sft"])
        out["base_vs_sft"] = {"delta": c["delta_pass^1"], "ci95": c["ci95"],
                              "wins": c["wins"], "losses": c["losses"], "ties": c["ties"]}
        print(f"paired base -> sft (pass^1): {100*c['delta_pass^1']:+.1f}pt "
              f"CI[{100*c['ci95'][0]:+.1f},{100*c['ci95'][1]:+.1f}] W/L/T {c['wins']}/{c['losses']}/{c['ties']}")

    if args.json:
        import json
        args.json.write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
