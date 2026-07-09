"""Train a prompt router from MERA trace rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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


def train(prompts: list[str], labels: list[int], seed: int = 42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    if len(prompts) < 2:
        raise ValueError("need at least 2 examples to train router")
    if len(set(labels)) < 2:
        print(f"[router] WARN only one class {set(labels)}; adding synthetic complement", file=sys.stderr)
        prompts = prompts + ["__synthetic_complementary_router_class__"]
        labels = labels + [1 - labels[0]]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=4096, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])

    metrics: dict[str, float | int] = {}
    min_class = min(labels.count(0), labels.count(1))
    if len(labels) >= 6 and min_class >= 2:
        splits = min(5, min_class)
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        probs = cross_val_predict(pipe, prompts, labels, cv=cv, method="predict_proba")[:, 1]
        preds = [1 if p >= 0.5 else 0 for p in probs]
        metrics = {
            "auc": float(roc_auc_score(labels, probs)),
            "acc": float(accuracy_score(labels, preds)),
            "f1_large": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
            "cv_splits": splits,
        }
    else:
        metrics = {"auc": 0.0, "acc": 0.0, "f1_large": 0.0, "cv_splits": 0}

    pipe.fit(prompts, labels)
    meta = {
        "chosen_featurizer": "tfidf",
        "metrics": metrics,
        "n_examples_total": len(prompts),
        "label_distribution": {
            "0_small_ok": labels.count(0),
            "1_need_large": labels.count(1),
        },
        "threshold_default": 0.5,
        "seed": seed,
    }
    return pipe, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts, labels, skipped = load_examples(args.traces)
    print(
        f"[router] loaded {len(prompts)} examples "
        f"({sum(labels)} need-large, {len(labels) - sum(labels)} small-ok); skipped={skipped}",
        file=sys.stderr,
    )
    pipe, meta = train(prompts, labels, seed=args.seed)

    import joblib

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.output_dir / "router.joblib")
    (args.output_dir / "router_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[router] wrote {args.output_dir / 'router.joblib'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
