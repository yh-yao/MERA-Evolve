import pytest

from verl_code_rl.evaluate_routing_baselines import (
    exact_cache_metrics,
    pre_route,
    response_cascade,
    signature_support_curve,
    verifier_coverage_curve,
)


def _row(task_id: str, small: bool, large: bool, prompt: str = "prompt") -> dict:
    return {
        "task_id": task_id,
        "dataset": "test",
        "prompt": prompt,
        "small_success": small,
        "large_success": large,
        "small_cost": 0.001,
        "large_cost": 0.01,
    }


def test_pre_route_separates_direct_routing_and_verifier_fallback():
    rows = [_row("easy", True, False), _row("hard", False, True)]

    direct = pre_route(rows, [False, False], verifier_fallback=False)
    cascade = pre_route(rows, [False, False], verifier_fallback=True)

    assert direct["passes"] == 1
    assert cascade["passes"] == 2
    assert cascade["fallback_rate"] == 0.5
    assert cascade["cost_vs_large"] == 0.6


def test_response_cascade_pays_for_small_before_large():
    rows = [_row("hard", False, True)]
    metrics = response_cascade(rows, [True])
    assert metrics["passes"] == 1
    assert metrics["cost_vs_large"] == pytest.approx(1.1)
    assert metrics["fallback_rate"] == 1.0


def test_exact_cache_reports_held_out_prompt_coverage():
    train = [_row("train", True, True, prompt="same")]
    evaluation = [
        _row("hit", False, True, prompt="same"),
        _row("miss", True, True, prompt="new"),
    ]
    metrics = exact_cache_metrics(train, evaluation)
    assert metrics["cache_hits"] == 1
    assert metrics["cache_coverage"] == 0.5
    assert metrics["passes"] == 2


def test_zero_verifier_coverage_conservatively_routes_everything_large():
    rows = [_row("easy", True, False), _row("hard", False, True)]
    curve = verifier_coverage_curve(rows, [False, False], seeds=2)
    zero_coverage = curve[-1]
    assert zero_coverage["coverage"] == 0.0
    assert zero_coverage["pass_rate_mean"] == 0.5
    assert zero_coverage["cost_vs_large_mean"] == 1.0


def test_sparse_signature_abstention_routes_unsupported_domain_large():
    train = [_row("train", True, True)]
    rows = [_row("easy", True, False), _row("hard", False, True)]
    curve = signature_support_curve(train, rows, [False, False])
    high_support = next(point for point in curve if point["minimum_signature_support"] == 10)
    assert high_support["abstained_signatures"] == ["test"]
    assert high_support["direct_large_rate"] == 1.0
