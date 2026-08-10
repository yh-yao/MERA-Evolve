"""Evaluate matched routing baselines and deployment stress tests from traces.

The learned baselines are adaptations to executable code outcomes:

* ``preference_router`` is RouteLLM-style: a query-only classifier predicts
  whether the large model is preferred before either endpoint is called.
* ``response_cascade`` is FrugalGPT-style: the small answer is generated
  first and a learned response scorer decides whether to escalate.

These names describe the decision pattern; they are not official checkpoints
or exact reproductions of either external system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SMALL_DEFAULT_COST = 0.001
LARGE_DEFAULT_COST = 0.01


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("small_success"), bool) and isinstance(row.get("large_success"), bool):
                rows.append(row)
    return rows


def costs(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row.get("small_cost") or SMALL_DEFAULT_COST),
        float(row.get("large_cost") or LARGE_DEFAULT_COST),
    )


def summarize(outcomes: Iterable[tuple[bool, float]], n: int) -> dict[str, float | int]:
    values = list(outcomes)
    passed = sum(int(ok) for ok, _ in values)
    total_cost = sum(cost for _, cost in values)
    return {
        "n": n,
        "passes": passed,
        "pass_rate": passed / n if n else 0.0,
        "avg_cost": total_cost / n if n else 0.0,
        "cost_vs_large": total_cost / (n * LARGE_DEFAULT_COST) if n else 0.0,
    }


def pre_route(
    rows: list[dict[str, Any]],
    route_large: list[bool],
    verifier_fallback: bool,
) -> dict[str, float | int]:
    outcomes: list[tuple[bool, float]] = []
    direct_large = fallback = 0
    for row, use_large in zip(rows, route_large):
        small_cost, large_cost = costs(row)
        if use_large:
            direct_large += 1
            outcomes.append((bool(row["large_success"]), large_cost))
        elif row["small_success"]:
            outcomes.append((True, small_cost))
        elif verifier_fallback:
            fallback += 1
            outcomes.append((bool(row["large_success"]), small_cost + large_cost))
        else:
            outcomes.append((False, small_cost))
    result = summarize(outcomes, len(rows))
    result.update({
        "direct_large_rate": direct_large / len(rows) if rows else 0.0,
        "fallback_rate": fallback / len(rows) if rows else 0.0,
    })
    return result


def response_cascade(
    rows: list[dict[str, Any]],
    escalate: list[bool],
) -> dict[str, float | int]:
    outcomes: list[tuple[bool, float]] = []
    escalations = 0
    for row, should_escalate in zip(rows, escalate):
        small_cost, large_cost = costs(row)
        if should_escalate:
            escalations += 1
            outcomes.append((bool(row["large_success"]), small_cost + large_cost))
        else:
            outcomes.append((bool(row["small_success"]), small_cost))
    result = summarize(outcomes, len(rows))
    result.update({
        "direct_large_rate": 0.0,
        "fallback_rate": escalations / len(rows) if rows else 0.0,
    })
    return result


def fit_text_router(
    rows: list[dict[str, Any]],
    include_response: bool,
    seed: int,
):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def text(row: dict[str, Any]) -> str:
        prompt = str(row.get("prompt") or "")
        if not include_response:
            return prompt
        return prompt + "\n<SMALL_RESPONSE>\n" + str(row.get("small_completion") or "")

    # Prefer/escalate to large only when doing so changes failure to success.
    labels = [int(not row["small_success"] and row["large_success"]) for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("text-router training rows contain only one preference class")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40_000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit([text(row) for row in rows], labels)
    return model, text


def probabilities(model, text, rows: list[dict[str, Any]]) -> list[float]:
    return [float(value) for value in model.predict_proba([text(row) for row in rows])[:, 1]]


def select_threshold(
    rows: list[dict[str, Any]],
    probs: list[float],
    post_response: bool,
) -> tuple[float, list[dict[str, float | int]]]:
    target = pre_route(rows, [True] * len(rows), verifier_fallback=False)["pass_rate"]
    curve: list[dict[str, float | int]] = []
    for index in range(101):
        threshold = index / 100
        decisions = [probability >= threshold for probability in probs]
        metrics = response_cascade(rows, decisions) if post_response else pre_route(
            rows, decisions, verifier_fallback=False,
        )
        curve.append({"threshold": threshold, **metrics})
    feasible = [point for point in curve if point["pass_rate"] >= target]
    chosen = min(feasible, key=lambda point: (point["avg_cost"], -point["pass_rate"])) if feasible else max(
        curve, key=lambda point: (point["pass_rate"], -point["avg_cost"]),
    )
    return float(chosen["threshold"]), curve


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def exact_cache_metrics(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    successful = {str(row.get("prompt") or "") for row in train_rows if row["small_success"]}
    hits = [str(row.get("prompt") or "") in successful for row in eval_rows]
    outcomes = []
    for row, hit in zip(eval_rows, hits):
        small_cost, _ = costs(row)
        outcomes.append((True, 0.0) if hit else (bool(row["small_success"]), small_cost))
    result = summarize(outcomes, len(eval_rows))
    result.update({
        "cache_hits": sum(hits),
        "cache_coverage": sum(hits) / len(hits) if hits else 0.0,
        "direct_large_rate": 0.0,
        "fallback_rate": 0.0,
    })
    return result


def verifier_coverage_curve(
    rows: list[dict[str, Any]], mera_large: list[bool], seeds: int = 20,
) -> list[dict[str, float]]:
    points = []
    for coverage in (1.0, 0.75, 0.5, 0.25, 0.0):
        samples = []
        for seed in range(seeds):
            # If verification is unavailable for a task class, conservatively
            # bypass the cheap path rather than serving an unchecked answer.
            route_large = [
                requested_large or stable_fraction(str(row.get("task_id")), seed) >= coverage
                for row, requested_large in zip(rows, mera_large)
            ]
            samples.append(pre_route(rows, route_large, verifier_fallback=True))
        points.append({
            "coverage": coverage,
            "pass_rate_mean": sum(float(x["pass_rate"]) for x in samples) / seeds,
            "cost_vs_large_mean": sum(float(x["cost_vs_large"]) for x in samples) / seeds,
            "fallback_rate_mean": sum(float(x["fallback_rate"]) for x in samples) / seeds,
        })
    return points


def small_model_drift_curve(
    rows: list[dict[str, Any]], mera_large: list[bool], seeds: int = 20,
) -> list[dict[str, float]]:
    points = []
    for regression in (0.0, 0.1, 0.25, 0.5):
        samples = []
        for seed in range(seeds):
            changed = []
            for row in rows:
                copy = dict(row)
                if copy["small_success"] and stable_fraction(str(row.get("task_id")), seed) < regression:
                    copy["small_success"] = False
                changed.append(copy)
            samples.append(pre_route(changed, mera_large, verifier_fallback=True))
        points.append({
            "regressed_fraction_of_small_successes": regression,
            "pass_rate_mean": sum(float(x["pass_rate"]) for x in samples) / seeds,
            "cost_vs_large_mean": sum(float(x["cost_vs_large"]) for x in samples) / seeds,
            "fallback_rate_mean": sum(float(x["fallback_rate"]) for x in samples) / seeds,
        })
    return points


def signature_support_curve(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    mera_large: list[bool],
) -> list[dict[str, float | int]]:
    support = Counter(str(row.get("dataset") or "unknown") for row in train_rows)
    points = []
    for minimum in (0, 10, 25, 50, 100, 200):
        route_large = [
            requested_large or support[str(row.get("dataset") or "unknown")] < minimum
            for row, requested_large in zip(eval_rows, mera_large)
        ]
        points.append({
            "minimum_signature_support": minimum,
            "abstained_signatures": sorted(name for name, count in support.items() if count < minimum),
            **pre_route(eval_rows, route_large, verifier_fallback=True),
        })
    return points


def domain_mix_curve(
    rows: list[dict[str, Any]], mera_large: list[bool],
) -> list[dict[str, float]]:
    domains = sorted({str(row.get("dataset") or "unknown") for row in rows})
    if len(domains) != 2:
        return []
    metrics = {}
    for domain in domains:
        indices = [index for index, row in enumerate(rows) if str(row.get("dataset") or "unknown") == domain]
        metrics[domain] = pre_route(
            [rows[index] for index in indices],
            [mera_large[index] for index in indices],
            verifier_fallback=True,
        )
    points = []
    for first_share in (0.0, 0.25, 0.5, 0.75, 1.0):
        second_share = 1.0 - first_share
        points.append({
            f"{domains[0]}_share": first_share,
            f"{domains[1]}_share": second_share,
            "pass_rate": first_share * float(metrics[domains[0]]["pass_rate"])
            + second_share * float(metrics[domains[1]]["pass_rate"]),
            "cost_vs_large": first_share * float(metrics[domains[0]]["cost_vs_large"])
            + second_share * float(metrics[domains[1]]["cost_vs_large"]),
        })
    return points


def evaluate(
    train_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    query_model, query_text = fit_text_router(train_rows, include_response=False, seed=seed)
    response_model, response_text = fit_text_router(train_rows, include_response=True, seed=seed)

    query_calibration = probabilities(query_model, query_text, calibration_rows)
    response_calibration = probabilities(response_model, response_text, calibration_rows)
    query_threshold, query_curve = select_threshold(calibration_rows, query_calibration, post_response=False)
    response_threshold, response_curve = select_threshold(
        calibration_rows, response_calibration, post_response=True,
    )
    query_eval = probabilities(query_model, query_text, eval_rows)
    response_eval = probabilities(response_model, response_text, eval_rows)
    preference_large = [value >= query_threshold for value in query_eval]
    response_escalate = [value >= response_threshold for value in response_eval]
    mera_large = [str(row.get("policy_route") or "small") == "large" for row in eval_rows]
    random_rate = sum(mera_large) / len(mera_large) if mera_large else 0.0
    random_large = [
        stable_fraction(str(row.get("task_id")), seed) < random_rate for row in eval_rows
    ]
    oracle_large = [not row["small_success"] and row["large_success"] for row in eval_rows]

    methods = {
        "always_small": pre_route(eval_rows, [False] * len(eval_rows), False),
        "always_large": pre_route(eval_rows, [True] * len(eval_rows), False),
        "verifier_cascade": pre_route(eval_rows, [False] * len(eval_rows), True),
        "random_budget_matched_no_fallback": pre_route(eval_rows, random_large, False),
        "random_budget_matched_with_fallback": pre_route(eval_rows, random_large, True),
        "preference_router_no_fallback": pre_route(eval_rows, preference_large, False),
        "preference_router_with_fallback": pre_route(eval_rows, preference_large, True),
        "response_cascade": response_cascade(eval_rows, response_escalate),
        "mera_router_no_fallback": pre_route(eval_rows, mera_large, False),
        "mera_router_with_fallback": pre_route(eval_rows, mera_large, True),
        "oracle_router": pre_route(eval_rows, oracle_large, False),
        "exact_cache_plus_small": exact_cache_metrics(train_rows, eval_rows),
    }
    domains = {}
    for domain in sorted({str(row.get("dataset") or "unknown") for row in eval_rows}):
        indices = [index for index, row in enumerate(eval_rows) if str(row.get("dataset") or "unknown") == domain]
        subset = [eval_rows[index] for index in indices]
        decisions = [mera_large[index] for index in indices]
        domains[domain] = pre_route(subset, decisions, True)

    return {
        "metadata": {
            "train_rows": len(train_rows),
            "calibration_rows": len(calibration_rows),
            "eval_rows": len(eval_rows),
            "seed": seed,
            "baseline_scope": "matched adaptations, not official external checkpoints",
        },
        "thresholds": {
            "preference_router": query_threshold,
            "response_cascade": response_threshold,
        },
        "methods": methods,
        "domain_breakdown": domains,
        "stress_tests": {
            "verifier_coverage": verifier_coverage_curve(eval_rows, mera_large),
            "small_model_drift": small_model_drift_curve(eval_rows, mera_large),
            "signature_support": signature_support_curve(train_rows, eval_rows, mera_large),
            "domain_mix": domain_mix_curve(eval_rows, mera_large),
        },
        "calibration_curves": {
            "preference_router": query_curve,
            "response_cascade": response_curve,
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Routing Baselines and Robustness",
        "",
        "Matched adaptations only; `preference_router` and `response_cascade` are not official external checkpoints.",
        "",
        "| Method | Pass | Cost vs large | Direct large | Fallback |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in result["methods"].items():
        lines.append(
            f"| {name} | {item['passes']}/{item['n']} ({item['pass_rate']:.1%}) | "
            f"{item['cost_vs_large']:.1%} | {item['direct_large_rate']:.1%} | {item['fallback_rate']:.1%} |"
        )
    lines.extend(["", "## Domain Breakdown", "", "| Domain | Pass | Cost vs large |", "|---|---:|---:|"])
    for name, item in result["domain_breakdown"].items():
        lines.append(f"| {name} | {item['passes']}/{item['n']} ({item['pass_rate']:.1%}) | {item['cost_vs_large']:.1%} |")
    lines.extend(["", "## Verifier Coverage", "", "| Coverage | Pass | Cost vs large | Fallback |", "|---:|---:|---:|---:|"])
    for item in result["stress_tests"]["verifier_coverage"]:
        lines.append(
            f"| {item['coverage']:.0%} | {item['pass_rate_mean']:.1%} | "
            f"{item['cost_vs_large_mean']:.1%} | {item['fallback_rate_mean']:.1%} |"
        )
    lines.extend(["", "## Simulated Small-Model Drift", "", "| Regressed successes | Pass | Cost vs large | Fallback |", "|---:|---:|---:|---:|"])
    for item in result["stress_tests"]["small_model_drift"]:
        lines.append(
            f"| {item['regressed_fraction_of_small_successes']:.0%} | {item['pass_rate_mean']:.1%} | "
            f"{item['cost_vs_large_mean']:.1%} | {item['fallback_rate_mean']:.1%} |"
        )
    lines.extend(["", "## Minimum Signature Support", "", "| Minimum support | Abstained signatures | Pass | Cost vs large |", "|---:|---|---:|---:|"])
    for item in result["stress_tests"]["signature_support"]:
        abstained = ", ".join(item["abstained_signatures"]) or "none"
        lines.append(
            f"| {item['minimum_signature_support']} | {abstained} | "
            f"{item['pass_rate']:.1%} | {item['cost_vs_large']:.1%} |"
        )
    domain_mix = result["stress_tests"]["domain_mix"]
    if domain_mix:
        share_keys = [key for key in domain_mix[0] if key.endswith("_share")]
        lines.extend(["", "## Domain-Mix Shift", "", f"| {share_keys[0]} | {share_keys[1]} | Pass | Cost vs large |", "|---:|---:|---:|---:|"])
        for item in domain_mix:
            lines.append(
                f"| {item[share_keys[0]]:.0%} | {item[share_keys[1]]:.0%} | "
                f"{item['pass_rate']:.1%} | {item['cost_vs_large']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-traces", type=Path, required=True)
    parser.add_argument("--calibration-traces", type=Path, required=True)
    parser.add_argument("--eval-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = evaluate(
        load_rows(args.train_traces),
        load_rows(args.calibration_traces),
        load_rows(args.eval_traces),
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.write_text(render_markdown(result))
    print(f"[routing-baselines] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
