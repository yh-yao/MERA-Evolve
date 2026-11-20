#!/usr/bin/env python3
"""Pre-register per-cohort, paired TRAIN-only proxy-SFT contracts."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


def keys(parquet: Path) -> set[tuple[str, str]]:
    return {(str(x.domain), str(x.task_id)) for x in pd.read_parquet(parquet).itertuples()}


def control_group(row: dict) -> tuple[str, str, str]:
    size = "short" if row["content_chars"] < 24_000 else ("medium" if row["content_chars"] < 36_000 else "long")
    return row["source"], row["domain"], size


def cohort(row: dict) -> str:
    # Match tau2_evolve.data_utility.cohort_key exactly.
    source, domain, size = control_group(row)
    return f"{source}|{domain}|{row['p_bin']}|{size}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector-dir", type=Path, required=True)
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--minimum-delta", type=float, default=0.05)
    p.add_argument("--training-seed", type=int, default=1)
    p.add_argument("--min-cohort-rows", type=int, default=20)
    a = p.parse_args()
    if a.output_dir.exists(): raise SystemExit(f"refusing to overwrite {a.output_dir}")
    selector = a.selector_dir / "selection_manifest.json"
    features_file = a.selector_dir / "candidate_features.jsonl"
    fit_t, fit_c = a.selector_dir / "selected_fit_sft.parquet", a.selector_dir / "matched_p1_control_fit_sft.parquet"
    for path in (selector, features_file, fit_t, fit_c, a.partition):
        if not path.is_file(): raise SystemExit(f"missing {path}")
    sm = json.loads(selector.read_text())
    if sm.get("schema_version") != 2 or sm.get("control_overlap_with_selected") != 0:
        raise SystemExit("selector manifest is not strict v2 matched selection")
    features = [json.loads(x) for x in features_file.read_text().splitlines() if x.strip()]
    selected = [x for x in features if x["split"] == "fit" and x["selected"]]
    treatment_rows = pd.read_parquet(fit_t).to_dict("records")
    control_rows = pd.read_parquet(fit_c).to_dict("records")
    treatment = {(str(x["domain"]), str(x["task_id"])): x for x in treatment_rows}
    control = {(str(x["domain"]), str(x["task_id"])): x for x in control_rows}
    partition = json.loads(a.partition.read_text())
    fit_keys = set(treatment) | set(control)
    frozen = [r for r in partition["TRAIN"] if (r["domain"], str(r["task_id"])) not in fit_keys]
    if not frozen: raise SystemExit("no official TRAIN tasks remain for frozen proxy evaluation")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in selected: groups[cohort(row)].append(row)
    a.output_dir.mkdir(parents=True)
    contracts = []
    skipped = []
    for name, cohort_features in sorted(groups.items()):
        if len(cohort_features) < a.min_cohort_rows:
            skipped.append({"cohort": name, "n_rows": len(cohort_features), "reason": "below_min_cohort_rows"})
            continue
        group = control_group(cohort_features[0])
        need = len(cohort_features)
        candidates = [row for row in control_rows if control_group({
            "source": "areal" if str(row["task_id"]).startswith("areal:") else "train",
            "domain": str(row["domain"]), "content_chars": sum(len(str(m.get("content") or "")) for m in row["messages"]),
        }) == group]
        candidates.sort(key=lambda row: (str(row["domain"]), str(row["task_id"])))
        if len(candidates) < need: raise SystemExit(f"cohort {name} has insufficient matched controls")
        t_rows = [treatment[(str(row["domain"]), str(row["task_id"]))] for row in cohort_features]
        c_rows = candidates[:need]
        slug = hashlib.sha256(name.encode()).hexdigest()[:12]
        directory = a.output_dir / "cohorts" / slug; directory.mkdir(parents=True)
        tp, cp = directory / "treatment_fit_sft.parquet", directory / "control_fit_sft.parquet"
        pd.DataFrame(t_rows).to_parquet(tp); pd.DataFrame(c_rows).to_parquet(cp)
        contract = {"schema_version": 1, "kind": "tau2_data_utility_proxy_contract", "synthetic_validation_only": False,
          "cohort": name, "selector_manifest": {"path": str(selector), "sha256": digest(selector)},
          "frozen_train_dev": {"partition": str(a.partition), "tasks": frozen, "sha256": digest(a.partition)},
          "runs": {"treatment": {"train_file": str(tp), "sha256": digest(tp)}, "control": {"train_file": str(cp), "sha256": digest(cp)}},
          "training": {"seed": a.training_seed, "sft_lr": 1e-7, "epochs": 1, "max_length": 24576, "lora": "r16-alpha32-attn+MLP"},
          "evaluation": {"trials": a.trials, "agent_temperature": 0.0, "user_temperature": 0.7, "agent_max_tokens": 1024, "thinking": False},
          "label_gate": {"minimum_delta": a.minimum_delta, "ci_lower_must_exceed": 0.0},
          "prohibited": ["VAL", "EVAL", "threshold tuning on frozen_train_dev"]}
        contract_path = directory / "proxy_contract.json"; contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        contracts.append({"cohort": name, "path": str(contract_path), "n_rows": need})
    plan = {"schema_version": 1, "kind": "tau2_data_utility_proxy_plan", "frozen_train_dev_tasks": len(frozen),
            "selector_manifest": str(selector), "min_cohort_rows": a.min_cohort_rows,
            "cohorts": contracts, "skipped_cohorts": skipped,
            "note": "Contracts are pre-registered CPU artifacts; no proxy SFT/evaluation was run."}
    (a.output_dir / "proxy_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(contracts)} cohort contracts; frozen official TRAIN dev tasks={len(frozen)}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
