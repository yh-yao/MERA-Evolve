"""Offline ablation metrics from MERA trace rows and a learned router."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_traces(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def labels_for(traces: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    kept = []
    labels = []
    for row in traces:
        small_success = row.get("small_success")
        if isinstance(small_success, bool):
            kept.append(row)
            labels.append(0 if small_success else 1)
    return kept, labels


def score(labels: list[int], preds: list[int]) -> dict[str, float | int]:
    if not labels:
        return {"routing_acc": 0.0, "large_f1": 0.0, "fallback": 0.0, "cost_vs_large": 0.0, "n_eval": 0}
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    large_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(labels)
    small_cost = 0.001
    large_cost = 0.01
    predicted_large = sum(preds)
    cost = predicted_large * large_cost + (n - predicted_large) * small_cost
    return {
        "routing_acc": sum(int(a == b) for a, b in zip(labels, preds)) / n,
        "large_f1": large_f1,
        "fallback": sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0) / n,
        "cost_vs_large": cost / (n * large_cost),
        "n_eval": n,
    }


def task_pass_rate(traces: list[dict[str, Any]], preds: list[int]) -> float:
    if not traces:
        return 0.0
    ok = 0
    for row, pred in zip(traces, preds):
        if pred == 0:
            passed = bool(row.get("small_success"))
        else:
            passed = bool(row.get("large_success")) if not row.get("large_skipped", False) else bool(row.get("final_success"))
        ok += int(passed)
    return ok / len(traces)


def router_predictions(traces: list[dict[str, Any]], router_dir: Path, threshold: float) -> list[int]:
    router_path = router_dir / "router.joblib" if router_dir.is_dir() else router_dir
    if not router_path.exists():
        return [0 for _ in traces]
    import joblib

    router = joblib.load(router_path)
    prompts = [str(row.get("prompt") or "") for row in traces]
    if hasattr(router, "predict_proba"):
        return [1 if float(prob[1]) >= threshold else 0 for prob in router.predict_proba(prompts)]
    return [int(x) for x in router.predict(prompts)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    traces, labels = labels_for(load_traces(args.traces))
    if not traces:
        print("[ablation] no labeled traces", file=sys.stderr)
        return 2

    preds = {
        "large": [1 for _ in traces],
        "skills": [0 for _ in traces],
        "router": router_predictions(traces, args.router_dir, args.router_threshold),
    }
    preds["full"] = list(preds["router"])

    variants = {}
    for name, pred in preds.items():
        variants[name] = score(labels, pred)
        variants[name]["task_pass"] = task_pass_rate(traces, pred)

    out = {
        "variants": variants,
        "label_distribution": dict(Counter(labels)),
        "n_traces_total": len(traces),
        "router_threshold": args.router_threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    if args.markdown_output:
        lines = [
            "| Variant | Routing Acc | Large F1 | Fallback | Cost vs Large | Task Pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in ("large", "skills", "router", "full"):
            item = variants[name]
            lines.append(
                f"| {name} | {item['routing_acc']:.2%} | {item['large_f1']:.2%} | "
                f"{item['fallback']:.2%} | {item['cost_vs_large']:.2%} | {item['task_pass']:.2%} |"
            )
        args.markdown_output.write_text("\n".join(lines) + "\n")

    print(f"[ablation] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
