#!/usr/bin/env python3
"""Fit a task-ID-free, interpretable utility router from cohort labels."""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tau2_evolve.data_utility import cohort_key, feature_vector  # noqa: E402

FEATURE_NAMES = ["r1_p", "message_count", "assistant_turns", "tool_calls", "content_chars_scaled"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--labels", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=0.01)
    a = p.parse_args()
    if a.output.exists(): raise SystemExit(f"refusing to overwrite {a.output}")
    labels = {}
    for path in a.labels:
        item = json.loads(path.read_text())
        if item.get("kind") != "tau2_data_utility_label": raise SystemExit(f"not a utility label: {path}")
        if item["label"] == "neutral": continue
        if item["cohort"] in labels: raise SystemExit(f"duplicate cohort label: {item['cohort']}")
        labels[item["cohort"]] = int(item["label"] == "helpful")
    rows = [json.loads(x) for x in a.features.read_text().splitlines() if x.strip()]
    train = [(feature_vector(row), labels[cohort_key(row)]) for row in rows if row.get("split") == "fit" and cohort_key(row) in labels]
    if len({y for _, y in train}) != 2: raise SystemExit("router requires both helpful and harmful non-neutral cohorts")
    means = [sum(x[i] for x, _ in train) / len(train) for i in range(len(FEATURE_NAMES))]
    scales = [max(math.sqrt(sum((x[i] - means[i]) ** 2 for x, _ in train) / len(train)), 1e-6) for i in range(len(FEATURE_NAMES))]
    vectors = [[(v - means[i]) / scales[i] for i, v in enumerate(x)] for x, _ in train]
    weights, bias = [0.0] * len(FEATURE_NAMES), 0.0
    for _ in range(a.epochs):
        dw, db = [0.0] * len(weights), 0.0
        for x, (_, y) in zip(vectors, train):
            prob = 1 / (1 + math.exp(-max(min(bias + sum(w * v for w, v in zip(weights, x)), 30), -30)))
            err = prob - y; db += err
            for i, v in enumerate(x): dw[i] += err * v
        for i in range(len(weights)): weights[i] -= a.learning_rate * (dw[i] / len(train) + a.l2 * weights[i])
        bias -= a.learning_rate * db / len(train)
    out = {"schema_version": 1, "kind": "tau2_data_utility_router", "feature_names": FEATURE_NAMES,
           "means": means, "scales": scales, "weights": weights, "bias": bias,
           "training_rows": len(train), "cohort_labels": dict(sorted(labels.items())),
           "class_counts": dict(Counter("helpful" if y else "harmful" for _, y in train))}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
