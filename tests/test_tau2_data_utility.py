from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TAU2 = Path(__file__).parents[1] / "experiments" / "tau-2"
sys.path.insert(0, str(TAU2))
from tau2_evolve import data_utility as utility  # noqa: E402


def _rows(passed: list[bool]) -> dict:
    return {("retail", "1"): {i: value for i, value in enumerate(passed)},
            ("retail", "2"): {i: value for i, value in enumerate(passed)}}


def test_paired_summary_and_utility_label() -> None:
    control = _rows([False, False, False, False])
    treatment = _rows([True, True, True, True])
    summary = utility.paired_summary(control, treatment, n_boot=100, seed=0)
    assert summary["delta_pass1"] == 1.0
    assert utility.utility_label(summary, {"minimum_delta": 0.05}) == "helpful"
    negative = utility.paired_summary(treatment, control, n_boot=100, seed=0)
    assert utility.utility_label(negative, {"minimum_delta": 0.05}) == "harmful"


def test_load_trials_rejects_outside_duplicate_and_malformed(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    allowed = {("retail", "1")}
    rows = [{"domain": "retail", "task_id": "1", "trial": i, "passed": bool(i % 2)} for i in range(4)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(utility.load_trials(path, allowed)[("retail", "1")]) == 4
    path.write_text(json.dumps({**rows[0], "task_id": "99"}) + "\n")
    with pytest.raises(ValueError, match="outside frozen"):
        utility.load_trials(path, allowed)
    path.write_text(json.dumps({**rows[0], "passed": "false"}) + "\n")
    with pytest.raises(ValueError, match="must be boolean"):
        utility.load_trials(path, allowed)


def test_contract_rejects_synthetic_and_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": "tau2_data_utility_proxy_contract",
                                "synthetic_validation_only": True, "selector_manifest": {}, "frozen_train_dev": {},
                                "runs": {}, "label_gate": {}}))
    with pytest.raises(ValueError, match="synthetic"):
        utility.load_proxy_contract(path)


def test_features_exclude_task_identity() -> None:
    row = {"r1_p": 0.5, "message_count": 10, "assistant_turns": 4, "tool_calls": 3, "content_chars": 1000}
    assert utility.feature_vector(row) == [0.5, 10.0, 4.0, 3.0, 0.02]
