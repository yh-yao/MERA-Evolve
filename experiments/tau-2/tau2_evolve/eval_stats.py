"""pass^k statistics and paired comparisons for multi-trial tau2 evaluations.

Input rows are the JSONL records written by ``collect_traces --trials K``:
one row per (domain, task_id, trial). Rows without a ``trial`` field are
treated as trial 0, so older single-run eval files still load.

Reported quantities:

* pass^1: mean per-task success rate over trials.
* pass^k: tau-bench's unbiased estimator, mean over tasks of
  C(c, k) / C(n, k) where c of n trials passed.
* Paired comparison of two runs on the same task set: per-task deltas of the
  pass^1 rate, a task-level paired bootstrap 95% CI, win/loss/tie counts,
  and exact two-sided McNemar tests on (a) each matched trial index and
  (b) the task-level all-k-pass indicator.
* Minimum detectable difference (MDD): for two runs of the SAME
  configuration, the paired difference is pure noise, so
  (z_{1-alpha/2} + z_{power}) * sd(per-task delta) / sqrt(N) is the smallest
  true pass^1 gain a comparison of this size detects with the given power.

Usage:
  python -m tau2_evolve.eval_stats summary RUN.jsonl [--k 4]
  python -m tau2_evolve.eval_stats compare --baseline A1.jsonl [A2.jsonl] --candidate B1.jsonl [B2.jsonl]
  python -m tau2_evolve.eval_stats repeatability RUN1.jsonl RUN2.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist, mean, pstdev

TaskKey = tuple[str, str]


def load_trials(path: Path) -> dict[TaskKey, dict[int, bool]]:
    """Return {(domain, task_id): {trial: passed}}; duplicate trials are errors.

    Rows carrying an ``error`` field are infrastructure failures, not
    benchmark outcomes, and are rejected so they cannot silently count as 0.
    """
    out: dict[TaskKey, dict[int, bool]] = defaultdict(dict)
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            raise ValueError(f"{path}:{line_number} is an errored row; rerun with --resume")
        key = (str(row["domain"]), str(row["task_id"]))
        trial = int(row.get("trial", 0))
        if trial in out[key]:
            raise ValueError(f"duplicate {key} trial {trial} in {path}:{line_number}")
        out[key][trial] = float(row.get("reward", 0.0) or 0.0) > 0.0
    return dict(out)


def load_runs(paths: list[Path]) -> dict[TaskKey, dict[int, bool]]:
    """Pool several runs of one configuration into one multi-trial result.

    Trials are renumbered in file order (run 0's trials first), so two
    4-trial runs with different seed blocks become one 8-trial result. All
    files must cover the same task set.
    """
    merged: dict[TaskKey, dict[int, bool]] = {}
    offset = 0
    for path in paths:
        run = load_trials(path)
        if merged and run.keys() != merged.keys():
            raise ValueError(f"{path} covers a different task set than {paths[0]}")
        width = max(len(v) for v in run.values())
        for key, trials in run.items():
            target = merged.setdefault(key, {})
            for index, (_, passed) in enumerate(sorted(trials.items())):
                target[offset + index] = passed
        offset += width
    return merged


def pass_hat_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k={k} exceeds trials={n}")
    return math.comb(c, k) / math.comb(n, k)


def summarize(trials: dict[TaskKey, dict[int, bool]], k: int | None = None) -> dict:
    n_trials = {len(v) for v in trials.values()}
    if len(n_trials) != 1:
        raise ValueError(f"tasks have unequal trial counts: {sorted(n_trials)}")
    n = n_trials.pop()
    k = n if k is None else k
    per_domain: dict[str, list[TaskKey]] = defaultdict(list)
    for key in trials:
        per_domain[key[0]].append(key)

    def block(keys: list[TaskKey]) -> dict:
        rates = [sum(trials[key].values()) / n for key in keys]
        return {
            "n_tasks": len(keys),
            "pass^1": mean(rates),
            f"pass^{k}": mean(pass_hat_k(n, sum(trials[key].values()), k) for key in keys),
        }

    result = {"n_trials": n, "k": k, "overall": block(list(trials))}
    result["domains"] = {d: block(keys) for d, keys in sorted(per_domain.items())}
    return result


def mcnemar_exact_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(
    deltas: list[float], *, n_boot: int = 10000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    boots = sorted(mean(rng.choices(deltas, k=n)) for _ in range(n_boot))
    lo = boots[int((1 - level) / 2 * n_boot)]
    hi = boots[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return lo, hi


def compare(
    baseline: dict[TaskKey, dict[int, bool]],
    candidate: dict[TaskKey, dict[int, bool]],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(f"task sets differ: missing={missing[:5]} extra={extra[:5]}")
    keys = sorted(baseline)
    nb = {len(v) for v in baseline.values()}
    nc = {len(v) for v in candidate.values()}
    if len(nb) != 1 or len(nc) != 1:
        raise ValueError("unequal trial counts within a run")
    n_base, n_cand = nb.pop(), nc.pop()

    rate_b = {key: sum(baseline[key].values()) / n_base for key in keys}
    rate_c = {key: sum(candidate[key].values()) / n_cand for key in keys}
    deltas = [rate_c[key] - rate_b[key] for key in keys]
    wins = sum(d > 0 for d in deltas)
    losses = sum(d < 0 for d in deltas)

    per_trial = []
    shared_trials = sorted(set.intersection(*(set(baseline[key]) & set(candidate[key]) for key in keys)))
    for t in shared_trials:
        w = sum(candidate[key][t] and not baseline[key][t] for key in keys)
        l = sum(baseline[key][t] and not candidate[key][t] for key in keys)
        per_trial.append({"trial": t, "wins": w, "losses": l, "p": mcnemar_exact_two_sided(w, l)})

    k = min(n_base, n_cand)
    allk_b = {key: all(baseline[key].values()) for key in keys} if n_base == k else None
    allk_c = {key: all(candidate[key].values()) for key in keys} if n_cand == k else None
    allk = None
    if allk_b is not None and allk_c is not None:
        w = sum(allk_c[key] and not allk_b[key] for key in keys)
        l = sum(allk_b[key] and not allk_c[key] for key in keys)
        allk = {"wins": w, "losses": l, "p": mcnemar_exact_two_sided(w, l)}

    lo, hi = paired_bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
    sd = pstdev(deltas) if len(deltas) > 1 else 0.0
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return {
        "n_tasks": len(keys),
        "baseline_pass^1": mean(rate_b.values()),
        "candidate_pass^1": mean(rate_c.values()),
        "delta_pass^1": mean(deltas),
        "ci95": [lo, hi],
        "wins": wins,
        "losses": losses,
        "ties": len(keys) - wins - losses,
        "mcnemar_per_trial": per_trial,
        "mcnemar_all_k": allk,
        "sd_task_delta": sd,
        "mdd_if_noise": z * sd / math.sqrt(len(keys)),
    }


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _print_summary(path: Path, k: int | None) -> None:
    s = summarize(load_trials(path), k)
    kk = s["k"]
    print(f"{path}  trials={s['n_trials']}")
    o = s["overall"]
    print(f"  overall  n={o['n_tasks']:3d}  pass^1={_fmt_pct(o['pass^1'])}  pass^{kk}={_fmt_pct(o[f'pass^{kk}'])}")
    for d, b in s["domains"].items():
        print(f"  {d:8s} n={b['n_tasks']:3d}  pass^1={_fmt_pct(b['pass^1'])}  pass^{kk}={_fmt_pct(b[f'pass^{kk}'])}")


def _print_compare(c: dict, label: str) -> None:
    print(label)
    print(f"  n_tasks={c['n_tasks']}  pass^1 {_fmt_pct(c['baseline_pass^1'])} -> {_fmt_pct(c['candidate_pass^1'])}"
          f"  delta={100 * c['delta_pass^1']:+.1f}pt  95% CI [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]")
    print(f"  per-task wins/losses/ties = {c['wins']}/{c['losses']}/{c['ties']}")
    for t in c["mcnemar_per_trial"]:
        print(f"  trial {t['trial']}: McNemar wins={t['wins']} losses={t['losses']} p={t['p']:.4g}")
    if c["mcnemar_all_k"]:
        a = c["mcnemar_all_k"]
        print(f"  all-k-pass: McNemar wins={a['wins']} losses={a['losses']} p={a['p']:.4g}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("runs", type=Path, nargs="+")
    s.add_argument("--k", type=int, default=None)
    c = sub.add_parser("compare")
    c.add_argument("--baseline", type=Path, nargs="+", required=True,
                   help="one or more runs of the baseline config (pooled)")
    c.add_argument("--candidate", type=Path, nargs="+", required=True,
                   help="one or more runs of the candidate config (pooled)")
    c.add_argument("--json", type=Path)
    r = sub.add_parser("repeatability", help="two runs of the same config -> noise level and MDD")
    r.add_argument("run1", type=Path)
    r.add_argument("run2", type=Path)
    r.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.cmd == "summary":
        for path in args.runs:
            _print_summary(path, args.k)
        return 0
    if args.cmd == "compare":
        result = compare(load_runs(args.baseline), load_runs(args.candidate))
        _print_compare(result, f"{[str(p) for p in args.baseline]} -> {[str(p) for p in args.candidate]}")
    else:
        result = compare(load_trials(args.run1), load_trials(args.run2))
        _print_compare(result, f"repeatability {args.run1} vs {args.run2}")
        lo, hi = result["ci95"]
        consistent = lo <= 0.0 <= hi
        print(f"  runs_consistent (CI covers 0) = {consistent}")
        print(f"  minimum detectable pass^1 difference (alpha=0.05, power=0.8) = {100 * result['mdd_if_noise']:.1f}pt")
        print("  EXPAND_EVAL=" + str(result["mdd_if_noise"] > 0.10))
        result["runs_consistent"] = consistent
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
