import json
from pathlib import Path

import numpy as np

from verl_code_rl.collect_traces import (
    _router_routes,
    large_cache_key,
    load_large_cache,
    shard_tasks,
    shard_tasks_many,
)
from verl_code_rl.prepare_data import convert_rows
from verl_code_rl.run_ablation import labels_for, policy_metrics, score
from verl_code_rl.skills import SkillBook
from verl_code_rl.traces_to_sft import extract_pairs
from verl_code_rl.train_router import load_binomial_examples, load_examples, train


def _trace(task_id: str, prompt: str, entry_point: str, ok: bool) -> dict:
    return {
        "task_id": task_id,
        "dataset": "humaneval",
        "prompt": prompt,
        "entry_point": entry_point,
        "test": f"def check(candidate):\n    assert candidate() == {1 if ok else 2}\n",
        "small_success": ok,
        "large_success": True,
        "small_completion": f"def {entry_point}():\n    return 1\n",
        "large_completion": f"def {entry_point}():\n    return 2\n",
    }


def test_skillbook_distills_procedure_from_successes():
    skillbook = SkillBook()
    row = _trace("HumanEval/0", "def f():\n", "f", True)
    skillbook.update_from_trace(row, small_model="small", large_model="large")
    assert skillbook.distill_all() == 1
    procedure = skillbook.get_procedure("def f():\n", "humaneval")
    assert "Procedure" in procedure
    assert "Worked examples" not in procedure


def test_convert_rows_can_prepend_skillbook_procedure():
    skillbook = SkillBook()
    row = _trace("HumanEval/0", "def f():\n", "f", True)
    skillbook.update_from_trace(row, small_model="small", large_model="large")
    skillbook.distill_all()

    rows = [{
        "task_id": "HumanEval/0",
        "prompt": "def f():\n",
        "entry_point": "f",
        "test": "def check(candidate):\n    assert candidate() == 1\n",
        "split": "train",
        "_src": "humaneval",
    }]
    out = convert_rows(rows, "train", skillbook=skillbook)
    assert out[0]["extra_info"]["has_procedure"] is True
    assert "Procedure" in out[0]["prompt"][1]["content"]


def test_router_and_ablation_helpers(tmp_path: Path):
    traces = [
        _trace("HumanEval/0", "def add():\n", "add", True),
        _trace("HumanEval/1", "def hard():\n", "hard", False),
        _trace("HumanEval/2", "def easy():\n", "easy", True),
        _trace("HumanEval/3", "def tricky():\n", "tricky", False),
    ]
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text("\n".join(json.dumps(row) for row in traces) + "\n")

    prompts, labels, skipped = load_examples(trace_path)
    assert skipped == 0
    assert labels == [0, 1, 0, 1]
    router, meta = train(prompts, labels)
    assert hasattr(router, "predict_proba")
    assert meta["label_distribution"]["1_need_large"] == 2

    kept, ablation_labels = labels_for(traces)
    assert len(kept) == 4
    assert score(ablation_labels, [0, 1, 0, 1])["routing_acc"] == 1.0


def test_binomial_router_examples_keep_task_groups(tmp_path: Path):
    traces = [
        {**_trace("HumanEval/0", "def f():\n", "f", True), "small_samples": 3, "small_pass_count": 2},
        {**_trace("HumanEval/1", "def g():\n", "g", False), "small_samples": 2, "small_pass_count": 0},
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in traces))
    prompts, labels, groups, skipped = load_binomial_examples(path)
    assert skipped == 0
    assert labels == [0, 0, 1, 1, 1]
    assert groups == ["HumanEval/0"] * 3 + ["HumanEval/1"] * 2
    assert len(prompts) == 5


def test_router_labels_only_large_rescues(tmp_path: Path):
    traces = [
        {**_trace("HumanEval/0", "def f():\n", "f", False), "large_success": True},
        {**_trace("HumanEval/1", "def g():\n", "g", False), "large_success": False},
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in traces))
    _, labels, _, skipped = load_binomial_examples(path)
    assert skipped == 0
    assert labels == [1, 0]
    _, ablation_labels = labels_for(traces)
    assert ablation_labels == [1, 0]


def test_multiple_router_shards_match_union_of_single_shards():
    tasks = [{"task_id": f"task-{index}"} for index in range(50)]
    expected = shard_tasks(tasks, 5, 0) + shard_tasks(tasks, 5, 2)
    actual = shard_tasks_many(tasks, 5, [0, 2])
    assert {row["task_id"] for row in actual} == {row["task_id"] for row in expected}
    assert [tasks.index(row) for row in actual] == sorted(tasks.index(row) for row in actual)


def test_policy_metrics_include_fallback_cost():
    traces = [
        {"small_success": True, "large_success": True, "large_skipped": False,
         "small_cost": 1.0, "large_cost": 10.0},
        {"small_success": False, "large_success": True, "large_skipped": False,
         "small_cost": 1.0, "large_cost": 10.0},
    ]
    metrics = policy_metrics(traces, [0, 0])
    assert metrics["task_pass"] == 1.0
    assert metrics["avg_cost"] == 6.0
    assert metrics["fallback_rate"] == 0.5


def test_sft_pairs_use_teacher_and_repair_successes():
    teacher = _trace("HumanEval/0", "def f():\n", "f", False)
    teacher["large_success"] = True
    repair = _trace("HumanEval/1", "def g():\n", "g", True)
    repair["small_turns"] = [
        {"success": False, "completion": "def g():\n    return 0", "code": "def g():\n    return 0"},
        {"success": True, "completion": "def g():\n    return 1", "code": "def g():\n    return 1"},
    ]
    pairs = extract_pairs([teacher, repair])
    assert {pair["source"] for pair in pairs} == {"teacher", "self_repair"}


def test_large_cache_key_is_config_sensitive_and_loadable(tmp_path: Path):
    task = {"task_id": "HumanEval/0", "prompt": "def f():\n"}
    key = large_cache_key(task, "teacher", 0.0, 768, 0)
    assert key != large_cache_key(task, "teacher", 0.2, 768, 0)
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"cache_key": key, "result": {"success": True}}) + "\n")
    assert load_large_cache(path)[key]["success"] is True


def test_router_routes_batches_all_prompts(monkeypatch):
    calls = []

    def fake_embed(prompts, model):
        calls.append((prompts, model))
        return np.array([[0.0], [1.0]])

    class Classifier:
        def predict_proba(self, features):
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    monkeypatch.setattr("verl_code_rl.embedding.embed", fake_embed)
    routes = _router_routes((Classifier(), "embed-model"), ["easy", "hard"], 0.5)
    assert calls == [(["easy", "hard"], "embed-model")]
    assert routes == [("small", 0.2), ("large", 0.9)]
