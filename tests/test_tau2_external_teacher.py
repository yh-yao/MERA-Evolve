from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

TAU2_EXPERIMENT = Path(__file__).parents[1] / "experiments" / "tau-2"
sys.path.insert(0, str(TAU2_EXPERIMENT))

from tau2_evolve import external_teacher  # noqa: E402


def _vllm_rows(prompt_ids: list[int], topk: int) -> list:
    """Shape of vLLM's /v1/completions prompt_logprobs (keys are str ids)."""
    rows = [None]
    for position in range(1, len(prompt_ids)):
        actual = prompt_ids[position]
        entry = {str(actual): {"logprob": -0.1 * position, "rank": 3, "decoded_token": "x"}}
        for rank in range(1, topk + 1):
            token = 1000 + 10 * position + rank
            entry[str(token)] = {"logprob": -float(rank), "rank": rank, "decoded_token": "y"}
        rows.append(entry)
    return rows


def test_estimator_rows_hold_actual_token_logprob_and_dummy_tail() -> None:
    prompt = [5, 6, 7, 8]
    ids, lps = external_teacher.parse_prompt_logprobs(_vllm_rows(prompt, 0), prompt, 0)
    assert ids == [[6], [7], [8], [0]]
    assert [row[0] for row in lps] == pytest.approx([-0.1, -0.2, -0.3, 0.0])
    assert len(ids) == len(prompt)


def test_topk_rows_are_rank_ordered_and_skip_out_of_topk_actual_token() -> None:
    prompt = [5, 6, 7]
    ids, lps = external_teacher.parse_prompt_logprobs(_vllm_rows(prompt, 2), prompt, 2)
    assert ids[0] == [1011, 1012]
    assert lps[0] == [-1.0, -2.0]
    assert ids[-1] == [0, 0] and lps[-1] == [0.0, 0.0]
    assert len(ids) == len(prompt)


def test_missing_actual_token_raises() -> None:
    with pytest.raises(ValueError, match="no logprob"):
        external_teacher.parse_prompt_logprobs([None, {"99": {"logprob": -1.0, "rank": 1}}], [1, 2], 0)
    with pytest.raises(ValueError, match="rows"):
        external_teacher.parse_prompt_logprobs([None], [1, 2], 0)


def test_client_posts_token_prompt_and_parses_response(monkeypatch) -> None:
    prompt = [3, 4, 5]
    seen = {}

    async def fake_post(self, payload):
        seen.update(payload)
        return {"choices": [{"prompt_logprobs": _vllm_rows(prompt, 0)}]}

    monkeypatch.setattr(external_teacher.ExternalTeacherClient, "_post", fake_post)
    client = external_teacher.ExternalTeacherClient("http://t/v1/", "teacher")
    out = asyncio.run(client.generate("rid", prompt_ids=prompt, sampling_params={"prompt_logprobs": 0, "temperature": 1.0}))
    assert seen["prompt"] == prompt and seen["max_tokens"] == 1 and seen["prompt_logprobs"] == 0
    assert seen["model"] == "teacher" and client.base_url == "http://t/v1"
    assert out.extra_fields["prompt_ids"] == [[4], [5], [0]]


def test_action_recall_weight_anneals_to_official_only(monkeypatch) -> None:
    pytest.importorskip("verl")
    agent_loop = importlib.import_module("tau2_evolve.agent_loop")
    monkeypatch.setenv("TAU2_ACTION_RECALL_WEIGHT", "0.5")
    monkeypatch.delenv("TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS", raising=False)
    assert agent_loop.official_reward_weight(10_000) == 0.5
    monkeypatch.setenv("TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS", "100")
    assert agent_loop.official_reward_weight(0) == 0.5
    assert agent_loop.official_reward_weight(50) == pytest.approx(0.75)
    assert agent_loop.official_reward_weight(500) == 1.0
    monkeypatch.setenv("TAU2_ACTION_RECALL_WEIGHT", "0")
    assert agent_loop.official_reward_weight(0) == 1.0
