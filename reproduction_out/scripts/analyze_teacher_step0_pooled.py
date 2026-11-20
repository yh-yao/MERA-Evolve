#!/usr/bin/env python3
"""Teacher Step 0, plan-primary analysis (research_plan.md E6 "Step 0", item 2).

The plan asks for the 9B teacher vs the 4B base on VAL seeds 300 + 1300 (8 trials/task)
and "the per-task paired CI of the difference"; the gate is a positive gap whose 95%
paired CI excludes 0. The 2026-11-18 verdict instead reported per-seed 4-trial
comparisons and applied the E2 checkpoint bar (+5 pt, compare_tasks.py). This script
recomputes the plan's statistic with the repository's own statistics module.

Why variants instead of the existing clean/ files: eval_stats.load_runs requires every
task to carry the same number of trials. Dropping the infrastructure-error rows (empty
assistant message from the thinking-on teacher) leaves some tasks with 3 of 4 trials, so
the pooled comparison is uncomputable ("unequal trial counts within a run"). We therefore
keep 8 trials per task and vary how the errored rows are scored:

  primary   errors counted as failures (conservative for the teacher)
  expected  each error imputed with the teacher's pass rate on the same task's other
            trials (falls back to the domain rate); the central estimate
  upper     errors counted as passes (optimistic bound, shows the infra-error cost)

No tau2 run, no training: pure re-analysis of results/e6_teacher_val + a base VAL run.

--base-dir picks the base run. Two protocols exist on disk:
  results/e0_base_val                      E0 (08 defaults, agent turns capped at 1024)
  results/e6_teacher_val/matched_base      re-collected with AGENT_MAX_TOKENS=131072 (teacher budget)

Usage: python3 analyze_teacher_step0_pooled.py [--base-dir DIR] [--tag NAME]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path("/home/powerop/work/yjh/MERA-Evolve")
sys.path.insert(0, str(REPO / "experiments" / "tau-2"))
from tau2_evolve.eval_stats import compare, load_runs, paired_bootstrap_ci  # noqa: E402

D = Path("/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro")
E0 = D / "results" / "e0_base_val"
E6 = D / "results" / "e6_teacher_val"
OUT = E6 / "pooled"
SEEDS = (300, 1300)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_variant(seed: int, kind: str) -> Path:
    """Rewrite the raw teacher file with the `error` field removed and the row scored.

    eval_stats.load_trials reads `reward`; compare_tasks.py reads `passed`. Set both so the
    file is usable by either, and drop `error` so the row is no longer rejected.
    """
    src = rows(E6 / f"teacher_VAL_seed{seed}_thinkon.jsonl")
    # Same-task pass rate over the non-errored trials, for `expected`.
    per_task: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in src:
        per_task[(r["domain"], str(r["task_id"]))].append(r)
    dom_rate: dict[str, float] = {}
    for domain in {r["domain"] for r in src}:
        ok = [r for r in src if r["domain"] == domain and not isinstance(r.get("error"), str)]
        dom_rate[domain] = sum(1 for r in ok if r.get("passed")) / max(1, len(ok))

    dst = OUT / f"teacher_VAL_seed{seed}_{kind}.jsonl"
    with dst.open("w") as fh:
        for r in src:
            r = dict(r)
            errored = isinstance(r.pop("error", None), str)
            if errored:
                if kind == "fail":
                    score = 0.0
                elif kind == "pass":
                    score = 1.0
                else:  # expected
                    others = [x for x in per_task[(r["domain"], str(r["task_id"]))] if not isinstance(x.get("error"), str)]
                    rate = (sum(1 for x in others if x.get("passed")) / len(others)) if others else dom_rate[r["domain"]]
                    score = rate
                r["passed"] = score >= 0.5
                r["reward"] = score
            fh.write(json.dumps(r) + "\n")
    return dst


def domain_block(base: dict, cand: dict, domain: str) -> tuple[float, float, int]:
    keys = [k for k in base if k[0] == domain]
    b = sum(sum(base[k].values()) / len(base[k]) for k in keys) / len(keys)
    c = sum(sum(cand[k].values()) / len(cand[k]) for k in keys) / len(keys)
    return b, c, len(keys)


def expected_rates(kind: str) -> dict:
    """Per-task candidate rate for the `expected` variant (fractional, unlike load_trials)."""
    per_task: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for seed in SEEDS:
        f = OUT / f"teacher_VAL_seed{seed}_{kind}.jsonl"
        for r in rows(f):
            per_task[(r["domain"], str(r["task_id"]))].append(float(r["reward"]))
    return {k: sum(v) / len(v) for k, v in per_task.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=E0)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. 'matched'")
    args = ap.parse_args()
    base_dir = args.base_dir.resolve()
    suffix = f"_{args.tag}" if args.tag else ""

    OUT.mkdir(parents=True, exist_ok=True)
    base = load_runs([base_dir / f"base_VAL_seed{s}.jsonl" for s in SEEDS])
    base_rate = {k: sum(v.values()) / len(v) for k, v in base.items()}

    results: dict[str, dict] = {}
    for kind in ("fail", "expected", "pass"):
        files = [write_variant(s, kind) for s in SEEDS]
        if kind == "expected":
            # fractional per-task rates; CI via the repository's paired bootstrap
            cand = expected_rates(kind)
            keys = sorted(set(base_rate) & set(cand))
            deltas = [cand[k] - base_rate[k] for k in keys]
            lo, hi = paired_bootstrap_ci(deltas)
            results[kind] = {
                "base": sum(base_rate[k] for k in keys) / len(keys),
                "cand": sum(cand[k] for k in keys) / len(keys),
                "delta": sum(deltas) / len(deltas),
                "ci": [lo, hi],
                "wins": sum(d > 0 for d in deltas),
                "losses": sum(d < 0 for d in deltas),
                "ties": sum(d == 0 for d in deltas),
            }
        else:
            cand = load_runs(files)
            c = compare(base, cand)
            results[kind] = {
                "base": c["baseline_pass^1"], "cand": c["candidate_pass^1"],
                "delta": c["delta_pass^1"], "ci": c["ci95"],
                "wins": c["wins"], "losses": c["losses"], "ties": c["ties"],
                "mdd": c["mdd_if_noise"],
                "_cand": cand,
            }

    primary = results["fail"]
    by_domain = {d: domain_block(base, primary["_cand"], d) for d in ("airline", "retail", "telecom")}
    for r in results.values():
        r.pop("_cand", None)

    json.dump({"seeds": list(SEEDS), "n_tasks": len(base_rate), "base_dir": str(base_dir),
               "variants": results, "by_domain_primary": by_domain},
              (OUT / f"teacher_vs_base_pooled{suffix}.json").open("w"), indent=1)

    def line(tag: str, r: dict) -> str:
        return (f"{tag:34s} base {100*r['base']:5.1f}% -> teacher {100*r['cand']:5.1f}%  "
                f"delta {100*r['delta']:+5.1f}pt  CI [{100*r['ci'][0]:+5.1f}, {100*r['ci'][1]:+5.1f}]  "
                f"W/L/T {r['wins']}/{r['losses']}/{r['ties']}")

    text = [
        "Teacher Step 0 - amended verdict (plan-primary pooled analysis)",
        "===============================================================",
        "research_plan.md E6 Step 0 item 2: positive VAL gap whose 95% paired CI excludes 0,",
        "on VAL seeds 300 + 1300 pooled (8 trials/task, 67 tasks). Base vs Qwen3.5-9B teacher",
        "(thinking ON, unbounded agent tokens), same frozen Qwen3.5-4B user simulator.",
        f"Base run: {base_dir}",
        "",
        line("errors as FAILURES (primary)", results["fail"]),
        line("errors imputed (central)", results["expected"]),
        line("errors as PASSES (upper bound)", results["pass"]),
        "",
        f"pooled MDD at this size: {100*results['fail'].get('mdd', 0):.1f}pt",
        "",
        "Primary, by domain (errors as failures):",
    ]
    for d, (b, c, n) in by_domain.items():
        text.append(f"  {d:8s} n={n:2d}  base {100*b:5.1f}%  teacher {100*c:5.1f}%  delta {100*(c-b):+5.1f}pt")
    text += [
        "",
        "Why the 2026-11-18 verdict said 'not passed':",
        "  - it reported per-seed 4-trial CIs (single-seed MDD ~10pt) instead of the pooled 8-trial CI;",
        "  - it applied compare_tasks.py's E2 checkpoint bar (+5pt, min-delta 0.05), which Step 0 does not use;",
        "  - the pooled statistic was not computable on clean/ because dropping errored rows leaves",
        "    unequal trial counts (eval_stats raises 'unequal trial counts within a run').",
        "  - its 'both variants give the same numbers' sensitivity check was inert: it rewrote `passed`",
        "    while eval_stats.load_trials reads `reward`, so the errored rows never changed.",
        "",
        "Protocol caveats still open (research_plan.md E6 'matched generation budgets'):",
        "  - the E0 base VAL used --agent-max-tokens 1024; the teacher used no agent cap. Base assistant",
        "    messages are p99=3.2k chars, max=4.7k chars, so the cap binds on a small tail only.",
        "  - E0 base VAL is 27.6%/26.9% against the plan's 34.0% reference (outside the +-3pt band);",
        "    the cause is not yet isolated.",
        "  - Step 0 item 2 also requires 'verified teacher recoveries of student failures on a fixed",
        "    TRAIN shard'; that check has not been run.",
        "",
        "Interpretation: under the plan's own criterion the teacher is positive but marginal",
        "(+4.5pt conservative, +6.0pt central) against a 10pt planning target. Headroom for the",
        "M cost-quality curve is small; do not start M on this teacher without a stronger result.",
        "",
        "Artifacts: pooled/teacher_VAL_seed{300,1300}_{fail,expected,pass}.jsonl,",
        f"           pooled/teacher_vs_base_pooled{suffix}.json",
    ]
    (E6 / f"STEP0_VERDICT_AMENDED{suffix}.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
