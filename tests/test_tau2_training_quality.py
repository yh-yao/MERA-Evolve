from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


TAU2_EXPERIMENT = Path(__file__).parents[1] / "experiments" / "tau-2"
sys.path.insert(0, str(TAU2_EXPERIMENT))

import lib_tau2  # noqa: E402


def test_action_completion_requires_repeated_golden_calls() -> None:
    task = {
        "evaluation_criteria": {
            "actions": [
                {"name": "lookup"},
                {"name": "lookup"},
                {"name": "update"},
            ]
        }
    }
    messages = [
        {"tool_calls": [{"name": "lookup"}, {"function": {"name": "update"}}]},
    ]

    _, _, recall, complete = lib_tau2.action_completion(task, messages)

    assert recall == 2 / 3
    assert complete is False


def test_action_completion_accepts_tau2_style_message_objects() -> None:
    task = SimpleNamespace(
        evaluation_criteria=SimpleNamespace(
            actions=[SimpleNamespace(name="lookup"), SimpleNamespace(name="update")]
        )
    )
    messages = [
        SimpleNamespace(
            tool_calls=[SimpleNamespace(name="lookup"), SimpleNamespace(name="update")]
        ),
        SimpleNamespace(content="done"),
    ]

    expected, observed, recall, complete = lib_tau2.action_completion(task, messages)

    assert expected == ["lookup", "update"]
    assert observed == expected
    assert recall == 1.0
    assert complete is True
