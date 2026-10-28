"""Pick GRPO prompts from R1 rollouts: tasks the current student sometimes solves (0 < p < 1).

Input: jsonl traces with several trials per task (collect_traces --trials k).
Output: one trace per selected task (its trial-0 rollout supplies the opening
user message for prepare_grpo_data), shuffled deterministically, plus a
summary of the p distribution per domain and source.

Usage: python select_frontier.py --traces 'r1/*.jsonl' --out frontier.jsonl [--source areal|train|all]
"""
import argparse, collections, glob, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True, help="glob")
ap.add_argument("--out", required=True)
ap.add_argument("--source", default="all", choices=["all", "areal", "train"])
ap.add_argument("--min-trials", type=int, default=4)
ap.add_argument("--max-p", type=float, default=0.999, help="keep tasks with 0 < p <= max-p")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

by_task = collections.defaultdict(list)
for path in glob.glob(args.traces):
    for line in open(path):
        r = json.loads(line)
        if r.get("error") or not r.get("messages"):
            continue
        src = "areal" if str(r["task_id"]).startswith("areal:") else "train"
        if args.source != "all" and src != args.source:
            continue
        by_task[(src, r["domain"], str(r["task_id"]))].append(r)

summary = collections.defaultdict(collections.Counter)
keep = []
for (src, dom, tid), rows in by_task.items():
    if len(rows) < args.min_trials:
        summary[(src, dom)]["incomplete"] += 1
        continue
    p = sum(bool(r["passed"]) for r in rows) / len(rows)
    bucket = "p=0" if p == 0 else ("p=1" if p == 1 else "0<p<1")
    summary[(src, dom)][bucket] += 1
    if 0 < p <= args.max_p:
        keep.append(sorted(rows, key=lambda r: r.get("trial", 0))[0])
random.Random(args.seed).shuffle(keep)
with open(args.out, "w") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for k in sorted(summary):
    print(k, dict(summary[k]))
print("selected", len(keep), collections.Counter((("areal" if str(r["task_id"]).startswith("areal:") else "train"), r["domain"]) for r in keep))
