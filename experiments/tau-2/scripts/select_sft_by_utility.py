#!/usr/bin/env python3
"""Apply a trained utility router to selector features; emit fit-only SFT data."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tau2_evolve.data_utility import feature_vector, task_key  # noqa: E402


def score(row: dict, model: dict) -> float:
    x = feature_vector(row)
    z = model["bias"] + sum(w * ((v - m) / s) for v, m, s, w in zip(x, model["means"], model["scales"], model["weights"]))
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--source-sft", type=Path, required=True)
    p.add_argument("--router", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--include-neutral", action="store_true")
    a = p.parse_args()
    if not 0.0 < a.threshold < 1.0: p.error("--threshold must be in (0,1)")
    if a.output_dir.exists(): raise SystemExit(f"refusing to overwrite {a.output_dir}")
    model = json.loads(a.router.read_text())
    if model.get("kind") != "tau2_data_utility_router": raise SystemExit("not a utility router")
    features = [json.loads(x) for x in a.features.read_text().splitlines() if x.strip()]
    selected = []
    for row in features:
        if row.get("split") != "fit": continue
        probability = score(row, model)
        if probability >= a.threshold or (a.include_neutral and probability >= 0.5):
            selected.append({**row, "utility_probability": probability})
    source = {(str(r.domain), str(r.task_id)): r._asdict() for r in pd.read_parquet(a.source_sft).itertuples()}
    if not all(task_key(row) in source for row in selected): raise SystemExit("selected feature missing source SFT row")
    a.output_dir.mkdir(parents=True)
    with (a.output_dir / "selected_features.jsonl").open("w") as f:
        for row in selected: f.write(json.dumps(row, sort_keys=True) + "\n")
    pd.DataFrame([source[task_key(row)] for row in selected]).to_parquet(a.output_dir / "utility_selected_fit_sft.parquet")
    manifest = {"schema_version": 1, "kind": "tau2_data_utility_selection", "router": str(a.router),
                "features": str(a.features), "threshold": a.threshold, "include_neutral": a.include_neutral,
                "n_selected_fit": len(selected), "prohibited": ["VAL", "EVAL", "dev training"]}
    (a.output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
