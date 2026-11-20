#!/usr/bin/env python3
"""Create a retail-first, TRAIN/AReaL-only proxy plan and frozen eval partition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def task_keys(path: Path) -> set[tuple[str, str]]:
    return {(str(row.domain), str(row.task_id)) for row in pd.read_parquet(path).itertuples()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-plan", type=Path, required=True)
    p.add_argument("--selector-dir", type=Path, required=True)
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    base = json.loads(args.base_plan.read_text())
    features = [json.loads(line) for line in (args.selector_dir / "candidate_features.jsonl").read_text().splitlines()]
    contracts = []
    for item in base["cohorts"]:
        if not item["cohort"].startswith("areal|retail|"):
            continue
        source = Path(item["path"])
        contract = json.loads(source.read_text())
        contracts.append((item, source, contract))
    if len(contracts) != 3:
        raise SystemExit(f"expected exactly 3 retail contracts, found {len(contracts)}")

    proxy_fit = set()
    for _, _, contract in contracts:
        proxy_fit |= task_keys(Path(contract["runs"]["treatment"]["train_file"]))
        proxy_fit |= task_keys(Path(contract["runs"]["control"]["train_file"]))
    original = json.loads(args.partition.read_text())
    official = [
        row for row in original["TRAIN"]
        if row["domain"] == "retail" and (row["domain"], str(row["task_id"])) not in proxy_fit
    ]
    retail_areal_dev = [
        {"bucket": "UTILITY_DEV", "domain": row["domain"], "task_id": row["task_id"]}
        for row in features
        if row["split"] == "dev" and row["source"] == "areal" and row["domain"] == "retail"
        and (row["domain"], row["task_id"]) not in proxy_fit
    ]
    dev_tasks = [{**row, "bucket": "UTILITY_DEV"} for row in official] + retail_areal_dev
    if not retail_areal_dev or len({(x["domain"], x["task_id"]) for x in dev_tasks}) != len(dev_tasks):
        raise SystemExit("frozen development partition is empty or has duplicate tasks")

    args.output_dir.mkdir(parents=True)
    partition = {"version": 2, "source": "retail_utility_proxy_v1", "UTILITY_DEV": dev_tasks}
    partition_path = args.output_dir / "partition_utility_dev.json"
    partition_path.write_text(json.dumps(partition, indent=2, sort_keys=True) + "\n")
    shared = {"partition": str(partition_path), "sha256": digest(partition_path), "tasks": dev_tasks}
    plan_items = []
    for item, source, contract in contracts:
        updated = {**contract, "frozen_train_dev": shared, "parent_contract": str(source),
                   "scope": "retail-first utility-selector pilot",
                   "evaluation": {**contract["evaluation"], "seeds": [300]}}
        slug = hashlib.sha256(item["cohort"].encode()).hexdigest()[:12]
        folder = args.output_dir / "cohorts" / slug
        folder.mkdir(parents=True)
        target = folder / "proxy_contract.json"
        target.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
        plan_items.append({"cohort": item["cohort"], "n_rows": item["n_rows"], "path": str(target)})
    plan = {"schema_version": 1, "kind": "tau2_retail_utility_proxy_plan", "scope": "retail-first pilot",
            "base_plan": str(args.base_plan), "frozen_utility_dev": shared,
            "cohorts": plan_items, "prohibited": ["VAL", "EVAL", "airline proxy training", "threshold tuning"]}
    (args.output_dir / "proxy_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote retail plan: cohorts={len(plan_items)} frozen tasks={len(dev_tasks)} "
          f"(official TRAIN={len(official)}, AReaL retail={len(retail_areal_dev)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
