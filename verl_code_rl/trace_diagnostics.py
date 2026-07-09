"""Report reward-variance and routing-label health from collected traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(rows: list[dict]) -> dict[str, float | int]:
    n = len(rows)
    if not n:
        return {"n_tasks": 0, "all_small_pass_rate": 0.0, "all_small_fail_rate": 0.0,
                "mixed_small_reward_rate": 0.0, "mean_small_pass_rate": 0.0}
    rates = []
    all_pass = all_fail = 0
    for row in rows:
        k = max(1, int(row.get("small_samples", 1) or 1))
        passed = int(row.get("small_pass_count", int(bool(row.get("small_success")))))
        rate = max(0.0, min(1.0, passed / k))
        rates.append(rate)
        all_pass += int(rate == 1.0)
        all_fail += int(rate == 0.0)
    return {
        "n_tasks": n,
        "all_small_pass_rate": all_pass / n,
        "all_small_fail_rate": all_fail / n,
        "mixed_small_reward_rate": (n - all_pass - all_fail) / n,
        "mean_small_pass_rate": sum(rates) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.traces.open() as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    report = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
