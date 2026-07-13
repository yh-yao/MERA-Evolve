"""Compare two tau2 eval JSONL files with an exact paired test."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_results(path: Path) -> dict[tuple[str, str], bool]:
    results: dict[tuple[str, str], bool] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        row = json.loads(line)
        key = (str(row["domain"]), str(row["task_id"]))
        if key in results:
            raise ValueError(f"duplicate task {key} in {path}:{line_number}")
        results[key] = float(row.get("reward", 0.0) or 0.0) > 0.0
    return results


def upper_binomial_tail(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2**trials)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    candidate = load_results(args.candidate)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(f"task sets differ: missing={missing}, extra={extra}")

    wins = sum(candidate[k] and not baseline[k] for k in baseline)
    losses = sum(baseline[k] and not candidate[k] for k in baseline)
    discordant = wins + losses
    one_sided_p = upper_binomial_tail(wins, discordant)
    baseline_passes = sum(baseline.values())
    candidate_passes = sum(candidate.values())
    total = len(baseline)

    domains: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for key, baseline_passed in baseline.items():
        domain = key[0]
        domains[domain][0] += int(baseline_passed)
        domains[domain][1] += int(candidate[key])
        domains[domain][2] += 1

    print(f"baseline:  {baseline_passes}/{total} = {baseline_passes / total:.1%}")
    print(f"candidate: {candidate_passes}/{total} = {candidate_passes / total:.1%}")
    print(f"delta:     {(candidate_passes - baseline_passes) / total:+.1%}")
    print(f"paired:    wins={wins} losses={losses} ties={total - discordant}")
    print(f"McNemar exact one-sided p={one_sided_p:.6g} (alpha={args.alpha})")
    for domain in sorted(domains):
        base_count, candidate_count, count = domains[domain]
        print(f"{domain}: {base_count}/{count} -> {candidate_count}/{count}")
    print("SIGNIFICANT_IMPROVEMENT=" + str(candidate_passes > baseline_passes and one_sided_p < args.alpha))


if __name__ == "__main__":
    main()
