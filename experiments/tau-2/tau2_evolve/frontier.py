"""Split TRAIN tasks by the current student's success rate (improvement plan).

Input: multi-trial TRAIN traces from ``collect_traces --trials K`` sampled
with the GRPO rollout temperature. Each task's success rate p decides its
training signal:

  p == 0      too hard: no GRPO gradient (all-zero group) -> distillation
  0 < p < 1   frontier: within-group reward variance -> GRPO
  p == 1      solved: only a small replay share

Writes, under --out-dir:
  frontier_traces.jsonl  one trace per frontier task (its first trial), the
                         input format prepare_grpo_data expects
  hard_traces.jsonl      likewise for p == 0
  solved_traces.jsonl    likewise for p == 1
  frontier_stats.json    per-task p and per-domain bucket counts, so the
                         frontier's movement can be tracked across cycles
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def bucket(p: float) -> str:
    if p <= 0.0:
        return "hard"
    if p >= 1.0:
        return "solved"
    return "frontier"


def split(rows: list[dict]) -> tuple[dict[tuple[str, str], list[dict]], dict]:
    by_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            raise ValueError(f"errored trace {row.get('domain')}/{row.get('task_id')}; rerun with --resume")
        by_task[(str(row["domain"]), str(row["task_id"]))].append(row)
    trials = {len(v) for v in by_task.values()}
    if len(trials) != 1:
        raise ValueError(f"unequal trial counts across tasks: {sorted(trials)}")
    per_task = {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for key, group in sorted(by_task.items()):
        group.sort(key=lambda r: int(r.get("trial", 0)))
        p = sum(float(r.get("reward", 0.0) or 0.0) > 0.0 for r in group) / len(group)
        per_task[f"{key[0]}/{key[1]}"] = {"p": p, "bucket": bucket(p)}
        counts[key[0]][bucket(p)] += 1
        counts["all"][bucket(p)] += 1
    stats = {
        "n_trials": trials.pop(),
        "counts": {d: dict(c) for d, c in sorted(counts.items())},
        "tasks": per_task,
    }
    return by_task, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for path in args.traces for line in path.read_text().splitlines() if line.strip()]
    by_task, stats = split(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = {name: (args.out_dir / f"{name}_traces.jsonl").open("w") for name in ("hard", "frontier", "solved")}
    for key, group in sorted(by_task.items()):
        name = stats["tasks"][f"{key[0]}/{key[1]}"]["bucket"]
        files[name].write(json.dumps(group[0], ensure_ascii=False) + "\n")
    for handle in files.values():
        handle.close()
    (args.out_dir / "frontier_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
