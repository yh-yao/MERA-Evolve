#!/usr/bin/env python3
"""Turn a completed paired proxy evaluation into one auditable utility label."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tau2_evolve.data_utility import load_proxy_contract, load_trials, paired_summary, utility_label  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--control-eval", type=Path, required=True)
    p.add_argument("--treatment-eval", type=Path, required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise SystemExit(f"refusing to overwrite {a.output}")
    contract = load_proxy_contract(a.contract)
    allowed = {(str(x["domain"]), str(x["task_id"])) for x in contract["frozen_train_dev"]["tasks"]}
    control, treatment = (
        load_trials(a.control_eval, allowed, expected_trials=contract["evaluation"]["trials"]),
        load_trials(a.treatment_eval, allowed, expected_trials=contract["evaluation"]["trials"]),
    )
    summary = paired_summary(control, treatment)
    out = {"schema_version": 1, "kind": "tau2_data_utility_label", "contract": str(a.contract),
           "cohort": a.cohort, "label": utility_label(summary, contract["label_gate"]),
           "summary": summary, "label_gate": contract["label_gate"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
