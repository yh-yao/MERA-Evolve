"""Paired per-task comparison of two multi-trial rollout files (e.g. a checkpoint vs. the base).

Each task's success rate is averaged over its trials; the difference is
bootstrapped over tasks (10k resamples). Prints one line and, with --json,
writes {"base", "cand", "delta", "ci_low", "ci_high", "wins", "losses", "ties",
"by_domain", "passes"} where passes = delta >= --min-delta and ci_low > 0.

Usage: python compare_tasks.py --base base.jsonl --cand ckpt.jsonl [--min-delta 0.05] [--json out.json]
"""
import argparse, collections, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--cand", required=True)
ap.add_argument("--min-delta", type=float, default=0.05)
ap.add_argument("--json")
args = ap.parse_args()


def load(path):
    d = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("error"):
            continue
        d[(r["domain"], str(r["task_id"]))].append(bool(r.get("passed")))
    return {k: sum(v) / len(v) for k, v in d.items()}


base, cand = load(args.base), load(args.cand)
keys = sorted(set(base) & set(cand))
if not keys:
    raise SystemExit("no common tasks")
diff = [cand[k] - base[k] for k in keys]
rng = random.Random(0)
boot = sorted(sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(10000))
by = collections.defaultdict(list)
for k in keys:
    by[k[0]].append(cand[k] - base[k])
out = {
    "n_tasks": len(keys),
    "base": sum(base[k] for k in keys) / len(keys),
    "cand": sum(cand[k] for k in keys) / len(keys),
    "delta": sum(diff) / len(diff),
    "ci_low": boot[250], "ci_high": boot[9750],
    "wins": sum(x > 0 for x in diff), "losses": sum(x < 0 for x in diff), "ties": sum(x == 0 for x in diff),
    "by_domain": {d: sum(v) / len(v) for d, v in by.items()},
}
out["passes"] = out["delta"] >= args.min_delta and out["ci_low"] > 0
print("n=%d %.3f -> %.3f delta %+.3f CI[%+.3f,%+.3f] W/L/T %d/%d/%d %s" % (
    out["n_tasks"], out["base"], out["cand"], out["delta"], out["ci_low"], out["ci_high"],
    out["wins"], out["losses"], out["ties"], "PASS" if out["passes"] else "below bar"))
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
