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
            large_success = row.get("large_success")
            need_large = (not small_success) and (
                bool(large_success) if isinstance(large_success, bool) else True
            )
            labels.append(int(need_large))
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
    return {
        "routing_acc": sum(int(a == b) for a, b in zip(labels, preds)) / n,
        "large_f1": large_f1,
        "fallback": sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0) / n,
        # Kept for JSON compatibility. Real cascade cost is filled by
        # policy_metrics() from measured trace usage.
        "cost_vs_large": 0.0,
        "n_eval": n,
    }


def policy_metrics(
    traces: list[dict[str, Any]],
    preds: list[int],
    fallback_small_cost: float = 0.001,
    fallback_large_cost: float = 0.01,
) -> dict[str, float | int]:
    """Evaluate the actual deployment policy, including small→large fallback."""
    if not traces:
        return {"task_pass": 0.0, "avg_cost": 0.0, "cost_vs_large": 0.0,
                "fallback_rate": 0.0, "direct_large_rate": 0.0, "unknown_rate": 0.0}
    passed = 0
    total_cost = 0.0
    always_large_cost = 0.0
    fallbacks = direct_large = unknown = 0
    for row, pred in zip(traces, preds):
        small_cost = float(row.get("small_cost") or fallback_small_cost)
        large_cost = float(row.get("large_cost") or fallback_large_cost)
        always_large_cost += large_cost
        large_known = not row.get("large_skipped", False)
        if pred == 1:
            direct_large += 1
            if not large_known:
                unknown += 1
                continue
            passed += int(bool(row.get("large_success")))
            total_cost += large_cost
        elif row.get("small_success"):
            passed += 1
            total_cost += small_cost
        elif not large_known:
            unknown += 1
        else:
            fallbacks += 1
            passed += int(bool(row.get("large_success")))
            total_cost += small_cost + large_cost
    known = len(traces) - unknown
    return {
        "task_pass": passed / known if known else 0.0,
        "avg_cost": total_cost / known if known else 0.0,
        "cost_vs_large": total_cost / always_large_cost if always_large_cost else 0.0,
        "fallback_rate": fallbacks / len(traces),
        "direct_large_rate": direct_large / len(traces),
        "unknown_rate": unknown / len(traces),
    }


def task_pass_rate(traces: list[dict[str, Any]], preds: list[int]) -> float:
    """Backward-compatible wrapper for callers/tests."""
    return float(policy_metrics(traces, preds)["task_pass"])


def router_predictions(traces: list[dict[str, Any]], router_dir: Path, threshold: float) -> list[int]:
    router_path = router_dir / "router.joblib" if router_dir.is_dir() else router_dir
    if not router_path.exists():
        return [0 for _ in traces]
    import joblib

    from verl_code_rl.embedding import DEFAULT_EMBED_MODEL, embed

    router = joblib.load(router_path)
    meta_path = (router_dir if router_dir.is_dir() else router_dir.parent) / "router_meta.json"
    embed_model = DEFAULT_EMBED_MODEL
    if meta_path.exists():
        try:
            embed_model = json.loads(meta_path.read_text()).get("embedding_model", DEFAULT_EMBED_MODEL)
        except (json.JSONDecodeError, OSError):
            pass
    features = embed([str(row.get("prompt") or "") for row in traces], embed_model)
    if hasattr(router, "predict_proba"):
        return [1 if float(prob[1]) >= threshold else 0 for prob in router.predict_proba(features)]
    return [int(x) for x in router.predict(features)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--fallback-small-cost", type=float, default=0.001)
    parser.add_argument("--fallback-large-cost", type=float, default=0.01)
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
        variants[name].update(policy_metrics(
            traces, pred, args.fallback_small_cost, args.fallback_large_cost,
        ))

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
            "| Variant | Routing Acc | Large F1 | Policy Fallback | Direct Large | Cost vs Large | Task Pass |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ("large", "skills", "router", "full"):
            item = variants[name]
            lines.append(
                f"| {name} | {item['routing_acc']:.2%} | {item['large_f1']:.2%} | "
                f"{item['fallback_rate']:.2%} | {item['direct_large_rate']:.2%} | "
                f"{item['cost_vs_large']:.2%} | {item['task_pass']:.2%} |"
            )
        args.markdown_output.write_text("\n".join(lines) + "\n")

    print(f"[ablation] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
