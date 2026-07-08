"""verl custom reward entrypoint for HumanEval/MBPP code tasks."""

from __future__ import annotations

from typing import Any

from verl_code_rl.code_eval import score_solution


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Score a rollout by executing its generated Python against hidden tests."""
    _ = data_source
    return score_solution(solution_str, ground_truth, extra_info=extra_info)
