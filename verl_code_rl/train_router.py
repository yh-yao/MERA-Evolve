"""Train a prompt router from MERA trace rows.

The router predicts P(small model fails) from a frozen text-embedding model
(see embedding.py) plus a lightweight LogisticRegression head -- no TF-IDF /
bag-of-words featurizer. The classifier is what gets joblib-dumped; the
embedding model itself is referenced by id in router_meta.json and reloaded
(once, cached) by any caller that needs to score new prompts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from verl_code_rl.embedding import DEFAULT_EMBED_MODEL, embed


def load_examples(path: Path) -> tuple[list[str], list[int], int]:
    prompts: list[str] = []
    labels: list[int] = []
    skipped = 0
    with path.open() as fh:
        for line in fh:
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            prompt = str(row.get("prompt") or "")
            small_success = row.get("small_success")
            if not prompt or not isinstance(small_success, bool):
                skipped += 1
                continue
            prompts.append(prompt)
            labels.append(0 if small_success else 1)
    return prompts, labels, skipped


def load_binomial_examples(path: Path) -> tuple[list[str], list[int], list[str], int]:
    """Expand K small rollouts and retain task ids for leakage-free grouped CV."""
    prompts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    skipped = 0
    with path.open() as fh:
        for line_no, line in enumerate(fh):
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            prompt = str(row.get("prompt") or "")
            if not prompt:
                skipped += 1
                continue
            n = int(row.get("small_samples", 1) or 1)
            passed = row.get("small_pass_count")
            if not isinstance(passed, int):
                success = row.get("small_success")
                if not isinstance(success, bool):
                    skipped += 1
                    continue
                passed = int(success)
            passed = max(0, min(passed, n))
            task_id = str(row.get("task_id") or f"line-{line_no}")
            prompts.extend([prompt] * n)
            labels.extend([0] * passed + [1] * (n - passed))
            groups.extend([task_id] * n)
    return prompts, labels, groups, skipped


def train(
    prompts: list[str],
    labels: list[int],
    seed: int = 42,
    groups: list[str] | None = None,
    embed_model: str = "",
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

    if len(prompts) < 2:
        raise ValueError("need at least 2 examples to train router")
    if len(set(labels)) < 2:
        print(f"[router] WARN only one class {set(labels)}; adding synthetic complement", file=sys.stderr)
        prompts = prompts + ["__synthetic_complementary_router_class__"]
        labels = labels + [1 - labels[0]]
        if groups is not None:
            groups = groups + ["__synthetic_complementary_router_class__"]

    embed_model = embed_model or DEFAULT_EMBED_MODEL
    features = embed(prompts, embed_model)

    metrics: dict[str, float | int] = {}
    min_class = min(labels.count(0), labels.count(1))
    n_groups = len(set(groups)) if groups is not None else len(labels)
    if len(labels) >= 6 and min_class >= 2 and n_groups >= 2:
        splits = min(5, min_class, n_groups)
        cv = GroupKFold(n_splits=splits) if groups is not None else StratifiedKFold(
            n_splits=splits, shuffle=True, random_state=seed,
        )
        try:
            probe = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            probs = cross_val_predict(
                probe, features, labels, cv=cv, groups=groups, method="predict_proba",
            )[:, 1]
            preds = [1 if p >= 0.5 else 0 for p in probs]
            metrics = {
                "auc": float(roc_auc_score(labels, probs)),
                "acc": float(accuracy_score(labels, preds)),
                "f1_large": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
                "brier": float(brier_score_loss(labels, probs)),
                "cv_splits": splits,
                "cv_grouped_by_task": groups is not None,
            }
        except ValueError as exc:
            print(f"[router] WARN grouped CV unavailable for this label layout: {exc}", file=sys.stderr)
            metrics = {
                "auc": 0.0, "acc": 0.0, "f1_large": 0.0, "brier": 0.0,
                "cv_splits": 0, "cv_grouped_by_task": groups is not None,
            }
    else:
        metrics = {
            "auc": 0.0, "acc": 0.0, "f1_large": 0.0, "brier": 0.0,
            "cv_splits": 0, "cv_grouped_by_task": groups is not None,
        }

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(features, labels)
    meta = {
        "chosen_featurizer": "qwen3-embedding",
        "embedding_model": embed_model,
        "metrics": metrics,
        "n_examples_total": len(prompts),
        "label_distribution": {
            "0_small_ok": labels.count(0),
            "1_need_large": labels.count(1),
        },
        "threshold_default": 0.5,
        "seed": seed,
    }
    return clf, meta


def select_threshold(
    router,
    calibration_path: Path,
    target_pass_rate: float | None,
    fallback_small_cost: float,
    fallback_large_cost: float,
    embed_model: str = "",
) -> tuple[float, dict[str, Any]]:
    """Select a cost-minimal cascade threshold on disjoint calibration traces."""
    rows: list[dict[str, Any]] = []
    with calibration_path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("small_success"), bool) and isinstance(row.get("large_success"), bool):
                rows.append(row)
    if not rows:
        raise ValueError("calibration traces must contain small_success and large_success")
    features = embed([str(row.get("prompt") or "") for row in rows], embed_model or DEFAULT_EMBED_MODEL)
    probabilities = router.predict_proba(features)[:, 1]
    curve: list[dict[str, float]] = []
    for i in range(1, 100):
        threshold = i / 100
        passed = total_cost = 0.0
        fallback_count = 0
        for row, probability in zip(rows, probabilities):
            small_cost = float(row.get("small_cost") or fallback_small_cost)
            large_cost = float(row.get("large_cost") or fallback_large_cost)
            if probability >= threshold:
                passed += int(bool(row["large_success"]))
                total_cost += large_cost
            elif row["small_success"]:
                passed += 1
                total_cost += small_cost
            else:
                passed += int(bool(row["large_success"]))
                total_cost += small_cost + large_cost
                fallback_count += 1
        curve.append({"threshold": threshold, "task_pass": passed / len(rows),
                      "avg_cost": total_cost / len(rows), "fallback_rate": fallback_count / len(rows)})
    feasible = [x for x in curve if target_pass_rate is None or x["task_pass"] >= target_pass_rate]
    chosen = min(feasible, key=lambda x: (x["avg_cost"], -x["task_pass"])) if feasible else max(
        curve, key=lambda x: (x["task_pass"], -x["avg_cost"]),
    )
    return float(chosen["threshold"]), {"chosen": chosen, "curve": curve}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-traces", type=Path, default=None,
                        help="Disjoint traces used only for policy-threshold selection.")
    parser.add_argument("--target-pass-rate", type=float, default=None)
    parser.add_argument("--fallback-small-cost", type=float, default=0.001)
    parser.add_argument("--fallback-large-cost", type=float, default=0.01)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help="Frozen text-embedding model used as the router featurizer.")
    args = parser.parse_args()

    prompts, labels, groups, skipped = load_binomial_examples(args.traces)
    print(
        f"[router] loaded {len(prompts)} examples "
        f"({sum(labels)} need-large, {len(labels) - sum(labels)} small-ok); skipped={skipped}",
        file=sys.stderr,
    )
    clf, meta = train(prompts, labels, seed=args.seed, groups=groups, embed_model=args.embed_model)

    if args.calibration_traces:
        threshold, calibration = select_threshold(
            clf, args.calibration_traces, args.target_pass_rate,
            args.fallback_small_cost, args.fallback_large_cost,
            embed_model=args.embed_model,
        )
        meta["threshold_default"] = threshold
        meta["threshold_selection"] = calibration

    import joblib

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.output_dir / "router.joblib")
    (args.output_dir / "router_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[router] wrote {args.output_dir / 'router.joblib'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
