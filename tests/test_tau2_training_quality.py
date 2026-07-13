from __future__ import annotations

import sys
import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


TAU2_EXPERIMENT = Path(__file__).parents[1] / "experiments" / "tau-2"
sys.path.insert(0, str(TAU2_EXPERIMENT))

import lib_tau2  # noqa: E402
import traces_to_sft  # noqa: E402
from tau2_verl_interaction import Tau2Interaction  # noqa: E402


def test_action_completion_accepts_shorter_valid_read_path() -> None:
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

    assert recall == 1.0
    assert complete is True


def test_action_completion_rejects_missing_action_type() -> None:
    task = {
        "evaluation_criteria": {
            "actions": [{"name": "lookup"}, {"name": "update"}]
        }
    }
    messages = [{"tool_calls": [{"name": "lookup"}]}]

    _, _, recall, complete = lib_tau2.action_completion(task, messages)

    assert recall == 0.5
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


def test_trace_with_explicitly_empty_action_requirements_is_complete() -> None:
    assert lib_tau2.trace_action_complete(
        {"expected_tool_calls": [], "observed_tool_calls": []}
    )


def test_verl_state_evaluator_is_an_instance_method() -> None:
    parameters = list(inspect.signature(Tau2Interaction._evaluate_state).parameters)
    assert parameters[0] == "self"


def test_sft_tool_calls_survive_parquet_without_argument_union(tmp_path: Path) -> None:
    messages = [
        traces_to_sft._clean_message(
            {
                "role": "assistant",
                "tool_calls": [{"name": "lookup", "arguments": {"user_id": "123"}}],
            }
        ),
        traces_to_sft._clean_message(
            {
                "role": "assistant",
                "tool_calls": [{"name": "cancel", "arguments": {"order_id": "456"}}],
            }
        ),
    ]
    output = tmp_path / "sft.parquet"

    pd.DataFrame([{"messages": messages}]).to_parquet(output)
    round_tripped = pd.read_parquet(output).iloc[0].messages

    assert "tool_calls" not in round_tripped[0]
    assert round_tripped[0]["content"] == (
        '<tool_call>\n{"name":"lookup","arguments":{"user_id":"123"}}\n</tool_call>'
    )
    assert "order_id" not in round_tripped[0]["content"]
