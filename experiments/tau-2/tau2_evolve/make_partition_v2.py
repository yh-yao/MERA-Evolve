"""Build partition v2: a usable VAL bucket and a real telecom split.

Why v2 exists (MERA-Evolve improvement plan, Phase 0):

* v1's VAL has 8 tasks -- too few to select checkpoints or calibrate a router.
* v1's telecom IDs do not resolve, so benchmark.task_ids_for() substituted
  get_tasks_small() (20 tasks, 8 of them EVAL) -- the main source of eval
  variance.

Rules:

* airline/retail TRAIN and EVAL are copied unchanged from v1 (the fixed
  splits must stay fixed). Their VAL = v1 VAL plus every task of tau2's
  "base" split that no v1 bucket uses.
* telecom uses tau2's official split of its 114 "base" tasks: EVAL = the
  official "test" split; the official "train" split is shuffled with a fixed
  seed and divided into VAL (--telecom-val) and TRAIN.
* DIAG is copied unchanged.

Usage (tau2 venv):
  PYTHONPATH=experiments/tau-2 .venv_tau2/bin/python3 -m tau2_evolve.make_partition_v2 \
      --out experiments/tau-2/data/partition_v2.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tau2_evolve import benchmark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, default=benchmark.PARTITION_PATH)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--telecom-val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    v1 = json.loads(args.v1.read_text())
    out: dict = {
        "version": 2,
        # Machine-independent provenance: path inside the tau2 workspace.
        "source_v1": str(args.v1.resolve().relative_to(benchmark.TAU2_WORKSPACE.resolve()))
        if args.v1.resolve().is_relative_to(benchmark.TAU2_WORKSPACE.resolve()) else str(args.v1),
        "seed": args.seed,
        "TRAIN": [],
        "VAL": [],
        "EVAL": [],
        "DIAG": list(v1.get("DIAG", [])),
    }
    for domain in ("airline", "retail"):
        base_ids = [str(t["id"]) for t in benchmark._adapter().load_tasks(domain)]
        used = {
            str(r["task_id"]) for bucket in ("TRAIN", "VAL", "EVAL", "DIAG")
            for r in v1.get(bucket, []) if r["domain"] == domain
        }
        unknown = used - set(base_ids)
        if unknown:
            raise SystemExit(f"{domain}: v1 IDs missing from tau2 tasks: {sorted(unknown)[:5]}")
        for bucket in ("TRAIN", "EVAL"):
            out[bucket] += [r for r in v1[bucket] if r["domain"] == domain]
        out["VAL"] += [r for r in v1["VAL"] if r["domain"] == domain]
        out["VAL"] += [{"domain": domain, "task_id": tid} for tid in base_ids if tid not in used]

    from tau2.domains.telecom.environment import get_tasks_split

    split = get_tasks_split()
    train_pool = sorted(split["train"])
    random.Random(args.seed).shuffle(train_pool)
    out["VAL"] += [{"domain": "telecom", "task_id": tid} for tid in sorted(train_pool[: args.telecom_val])]
    out["TRAIN"] += [{"domain": "telecom", "task_id": tid} for tid in sorted(train_pool[args.telecom_val :])]
    out["EVAL"] += [{"domain": "telecom", "task_id": tid} for tid in sorted(split["test"])]

    seen: dict[tuple[str, str], str] = {}
    for bucket in ("TRAIN", "VAL", "EVAL", "DIAG"):
        for r in out[bucket]:
            key = (r["domain"], str(r["task_id"]))
            if key in seen:
                raise SystemExit(f"{key} appears in both {seen[key]} and {bucket}")
            seen[key] = bucket
    for domain in benchmark.DOMAINS:
        ids = {str(t["id"]) for t in benchmark._load_all_tasks(domain)}
        missing = [r for b in ("TRAIN", "VAL", "EVAL") for r in out[b]
                   if r["domain"] == domain and str(r["task_id"]) not in ids]
        if missing:
            raise SystemExit(f"{domain}: unresolvable task IDs {missing[:3]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for bucket in ("TRAIN", "VAL", "EVAL", "DIAG"):
        print(bucket, len(out[bucket]), dict(Counter(r["domain"] for r in out[bucket])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
