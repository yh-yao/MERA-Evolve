#!/usr/bin/env python3
"""Build auditable TRAIN/AReaL-only SFT selections.

The default ``p_lt_1_matched`` arm pairs p<1 candidates with distinct p=1
controls in the same source/domain/length stratum.  This is a rule baseline,
not a learned usefulness router.  It accepts neither VAL/EVAL official tasks
nor AReaL tasks outside the supplied immutable allow-list.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tau2_evolve.data_selection import (  # noqa: E402
    SELECTORS, assign_splits, enrich_candidates, load_areal_allowlist,
    load_partition_membership, load_r1_rates, matched_p_lt_1_selection,
    matched_random, selector_accepts, sha256_file, task_key,
)


def _rows(path: Path) -> list[dict]:
    return pd.read_parquet(path).to_dict("records")


def _counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(f"{r['source']}/{r['domain']}" for r in rows).items()))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_parquet(path: Path, rows: list[dict], lookup: dict[tuple[str, str], dict]) -> None:
    # The named artifact is a training input. Never silently include dev rows.
    pd.DataFrame([lookup[task_key(row)] for row in rows]).to_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-traces", type=Path, action="append", required=True)
    parser.add_argument("--sft-pairs", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--areal-allowlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector", choices=SELECTORS, default="p_lt_1_matched")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--min-r1-trials", type=int, default=4)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    membership = load_partition_membership(args.partition)
    areal_allowed = load_areal_allowlist(args.areal_allowlist)
    rates = load_r1_rates(args.r1_traces, min_trials=args.min_r1_trials,
                          membership=membership, areal_allowed=areal_allowed)
    original_rows = _rows(args.sft_pairs)
    features = enrich_candidates(original_rows, rates, membership, areal_allowed)
    splits = assign_splits(features, dev_fraction=args.dev_fraction, seed=args.seed)
    for row in features:
        row["split"] = splits[task_key(row)]
        row["selected"] = selector_accepts(row, args.selector)

    selected = ([] if args.selector == "p_lt_1_matched" else [row for row in features if row["selected"]])
    if args.selector == "p_lt_1_matched":
        selected = [
            row
            for split in ("fit", "dev")
            for row in matched_p_lt_1_selection(
                [item for item in features if item["split"] == split], seed=args.seed
            )
        ]
    selected_keys = {task_key(row) for row in selected}
    for row in features:
        row["selected"] = task_key(row) in selected_keys
    selected_fit = [row for row in selected if row["split"] == "fit"]
    selected_dev = [row for row in selected if row["split"] == "dev"]
    lookup = {task_key(row): row for row in original_rows}

    controls: list[dict] = []
    if args.selector == "p_lt_1_matched":
        controls_fit = matched_random([row for row in features if row["split"] == "fit"], selected_fit, seed=args.seed)
        controls_dev = matched_random([row for row in features if row["split"] == "dev"], selected_dev, seed=args.seed)
        controls = controls_fit + controls_dev
        _write_parquet(output / "matched_p1_control_fit_sft.parquet", controls_fit, lookup)
        _write_parquet(output / "matched_p1_control_dev_sft.parquet", controls_dev, lookup)

    _write_jsonl(output / "candidate_features.jsonl", features)
    _write_parquet(output / "selected_fit_sft.parquet", selected_fit, lookup)
    _write_parquet(output / "selected_dev_sft.parquet", selected_dev, lookup)
    manifest = {
        "schema_version": 2,
        "purpose": "TRAIN/AReaL-only rule-based SFT selection baseline; not a utility router",
        "selector": args.selector, "seed": args.seed, "dev_fraction": args.dev_fraction,
        "inputs": {"partition": {"path": str(args.partition), "sha256": sha256_file(args.partition)},
                   "areal_allowlist": {"path": str(args.areal_allowlist), "sha256": sha256_file(args.areal_allowlist)},
                   "sft_pairs": {"path": str(args.sft_pairs), "sha256": sha256_file(args.sft_pairs)},
                   "r1_traces": [{"path": str(path), "sha256": sha256_file(path)} for path in args.r1_traces]},
        "n_candidates": len(features), "n_selected_fit": len(selected_fit), "n_selected_dev": len(selected_dev),
        "n_matched_p1_control_fit": sum(row["split"] == "fit" for row in controls),
        "n_matched_p1_control_dev": sum(row["split"] == "dev" for row in controls),
        "candidate_counts": _counts(features), "selected_counts": _counts(selected),
        "control_counts": _counts(controls), "split_counts": dict(Counter(row["split"] for row in features)),
        "control_overlap_with_selected": len({task_key(row) for row in controls} & selected_keys),
        "official_partition_policy": "only TRAIN accepted; VAL/EVAL/DIAG rejected",
        "selection_policy": "p_lt_1_matched: p<1 rows only when a distinct p=1 control exists in the same source/domain/length stratum",
        "training_artifacts": ["selected_fit_sft.parquet", "matched_p1_control_fit_sft.parquet"],
        "development_artifacts": ["selected_dev_sft.parquet", "matched_p1_control_dev_sft.parquet"],
    }
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
