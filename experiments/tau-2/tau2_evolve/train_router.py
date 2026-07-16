"""Train the lightweight tau2 escalation router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2_evolve.router import RouterModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    rows = [
        json.loads(line)
        for path in args.data
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    model = RouterModel.fit(rows, threshold=args.threshold)
    model.save(args.output)
    accuracy = sum(
        model.should_escalate(row["features"]) == bool(row["label"])
        for row in rows
    ) / len(rows)
    print(f"[router] rows={len(rows)} train_accuracy={accuracy:.1%} output={args.output}")


if __name__ == "__main__":
    main()
