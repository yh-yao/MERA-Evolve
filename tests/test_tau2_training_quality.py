from __future__ import annotations

import sys
import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


TAU2_EXPERIMENT = Path(__file__).parents[1] / "experiments" / "tau-2"
sys.path.insert(0, str(TAU2_EXPERIMENT))

from tau2_evolve import benchmark, prepare_grpo_data, traces_to_sft  # noqa: E402
from tau2_evolve.interaction import Tau2Interaction  # noqa: E402
from tau2_evolve.skills import SkillBook  # noqa: E402
from verl_code_rl.extract_sft_lora_adapter import _normalize_checkpoint_keys


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

    _, _, recall, complete = benchmark.action_completion(task, messages)

    assert recall == 1.0
    assert complete is True


def test_action_completion_rejects_missing_action_type() -> None:
    task = {
        "evaluation_criteria": {
            "actions": [{"name": "lookup"}, {"name": "update"}]
        }
    }
    messages = [{"tool_calls": [{"name": "lookup"}]}]

    _, _, recall, complete = benchmark.action_completion(task, messages)

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

    expected, observed, recall, complete = benchmark.action_completion(task, messages)

    assert expected == ["lookup", "update"]
    assert observed == expected
    assert recall == 1.0
    assert complete is True


def test_trace_with_explicitly_empty_action_requirements_is_complete() -> None:
    assert benchmark.trace_action_complete(
        {"expected_tool_calls": [], "observed_tool_calls": []}
    )


def test_qwen_thinking_flag_is_forwarded_to_chat_template() -> None:
    spec = benchmark.make_llm_spec(
        "openai/qwen3", "http://localhost:8000/v1", enable_thinking=False
    )

    assert spec.args["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


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


def test_sft_domain_balancing_is_deterministic() -> None:
    rows = [
        {"domain": "airline", "task_id": "a1"},
        {"domain": "retail", "task_id": "r1"},
        {"domain": "retail", "task_id": "r2"},
        {"domain": "telecom", "task_id": "t1"},
    ]

    first = traces_to_sft._balance_domains(rows)
    second = traces_to_sft._balance_domains(rows)

    assert first == second
    assert len(first) == 6
    assert {domain: sum(r["domain"] == domain for r in first) for domain in {"airline", "retail", "telecom"}} == {
        "airline": 2,
        "retail": 2,
        "telecom": 2,
    }


def test_grpo_domain_balancing_is_interleaved() -> None:
    rows = [
        {"ability": "retail", "id": "r0"},
        {"ability": "airline", "id": "a0"},
        {"ability": "airline", "id": "a1"},
        {"ability": "telecom", "id": "t0"},
    ]

    balanced = prepare_grpo_data.balance_and_interleave_domains(rows)

    assert [row["ability"] for row in balanced] == [
        "airline", "retail", "telecom", "airline", "retail", "telecom"
    ]
    assert [row["id"] for row in balanced] == ["a0", "r0", "t0", "a1", "r0", "t0"]


def test_sft_drops_only_assistant_messages_before_first_user() -> None:
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "generic greeting"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "decision"},
    ]

    cleaned = traces_to_sft._drop_pre_user_assistant_messages(messages)

    assert [message["content"] for message in cleaned] == ["policy", "task", "decision"]


def test_sft_requires_a_nonempty_assistant_target() -> None:
    assert not traces_to_sft._has_assistant_target([
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "task"},
    ])
    assert traces_to_sft._has_assistant_target([
        {"role": "assistant", "content": "decision"},
    ])


def test_sft_adapter_export_maps_qwen35_language_model_keys() -> None:
    source_key = (
        "base_model.model.model.language_model.layers.0.self_attn.q_proj."
        "lora_A.default.weight"
    )
    target_key = (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    )

    normalized, remapped = _normalize_checkpoint_keys(
        {source_key: "weight"}, {target_key}
    )

    assert normalized == {target_key: "weight"}
    assert remapped == 1


def test_telecom_skill_covers_policy_mobile_data_branches() -> None:
    rendered = SkillBook().skills["telecom"].render()

    for action in (
        "toggle_data",
        "toggle_roaming",
        "toggle_data_saver_mode",
        "disconnect_vpn",
        "set_network_mode_preference",
    ):
        assert action in rendered
    assert "actions the USER performs" in rendered
