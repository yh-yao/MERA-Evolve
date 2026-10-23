from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

TAU2_EXPERIMENT = Path(__file__).parents[1] / "experiments" / "tau-2"
sys.path.insert(0, str(TAU2_EXPERIMENT))

from tau2_evolve import eval_stats  # noqa: E402


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _rows(outcomes: dict[str, list[int]], domain: str = "retail") -> list[dict]:
    return [
        {"domain": domain, "task_id": tid, "trial": t, "reward": float(r)}
        for tid, results in outcomes.items()
        for t, r in enumerate(results)
    ]


def test_pass_hat_k_matches_tau_bench_estimator() -> None:
    assert eval_stats.pass_hat_k(4, 4, 4) == 1.0
    assert eval_stats.pass_hat_k(4, 3, 4) == 0.0
    assert eval_stats.pass_hat_k(4, 2, 1) == 0.5
    assert eval_stats.pass_hat_k(4, 3, 2) == pytest.approx(math.comb(3, 2) / math.comb(4, 2))


def test_summary_reports_pass1_and_passk(tmp_path: Path) -> None:
    path = _write(tmp_path / "run.jsonl", _rows({"a": [1, 1, 1, 1], "b": [1, 0, 0, 0]}))
    s = eval_stats.summarize(eval_stats.load_trials(path))
    assert s["n_trials"] == 4
    assert s["overall"]["pass^1"] == pytest.approx((1.0 + 0.25) / 2)
    assert s["overall"]["pass^4"] == pytest.approx(0.5)


def test_rows_without_trial_field_load_as_single_run(tmp_path: Path) -> None:
    path = _write(tmp_path / "old.jsonl", [{"domain": "airline", "task_id": "1", "reward": 1.0}])
    assert eval_stats.load_trials(path) == {("airline", "1"): {0: True}}


def test_errored_and_duplicate_rows_are_rejected(tmp_path: Path) -> None:
    errored = _write(tmp_path / "e.jsonl", [{"domain": "a", "task_id": "1", "error": "boom"}])
    with pytest.raises(ValueError, match="errored"):
        eval_stats.load_trials(errored)
    dup = _write(tmp_path / "d.jsonl", _rows({"x": [1]}) + _rows({"x": [0]}))
    with pytest.raises(ValueError, match="duplicate"):
        eval_stats.load_trials(dup)


def test_compare_counts_and_mcnemar(tmp_path: Path) -> None:
    base = _write(tmp_path / "b.jsonl", _rows({"a": [0, 0], "b": [1, 1], "c": [1, 0]}))
    cand = _write(tmp_path / "c.jsonl", _rows({"a": [1, 1], "b": [1, 1], "c": [0, 0]}))
    c = eval_stats.compare(eval_stats.load_trials(base), eval_stats.load_trials(cand), n_boot=200)
    assert (c["wins"], c["losses"], c["ties"]) == (1, 1, 1)
    assert c["delta_pass^1"] == pytest.approx((1.0 + 0.0 - 0.5) / 3)
    assert [t["wins"] for t in c["mcnemar_per_trial"]] == [1, 1]
    assert c["mcnemar_all_k"] == {"wins": 1, "losses": 0, "p": 1.0}
    lo, hi = c["ci95"]
    assert lo <= c["delta_pass^1"] <= hi


def test_compare_rejects_mismatched_task_sets(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.jsonl", _rows({"x": [1]}))
    b = _write(tmp_path / "b.jsonl", _rows({"y": [1]}))
    with pytest.raises(ValueError, match="task sets differ"):
        eval_stats.compare(eval_stats.load_trials(a), eval_stats.load_trials(b))


def test_identical_runs_have_zero_noise() -> None:
    trials = {("retail", str(i)): {0: bool(i % 2), 1: bool(i % 3)} for i in range(30)}
    c = eval_stats.compare(trials, trials, n_boot=100)
    assert c["delta_pass^1"] == 0.0
    assert c["mdd_if_noise"] == 0.0
    assert c["ci95"] == [0.0, 0.0]


def test_mcnemar_exact_two_sided() -> None:
    assert eval_stats.mcnemar_exact_two_sided(0, 0) == 1.0
    assert eval_stats.mcnemar_exact_two_sided(7, 3) == pytest.approx(2 * sum(math.comb(10, i) for i in range(4)) / 2**10)
    assert eval_stats.mcnemar_exact_two_sided(10, 0) == pytest.approx(2 / 2**10)


def test_partition_v2_is_disjoint_and_keeps_v1_airline_retail_eval() -> None:
    v2 = json.loads((TAU2_EXPERIMENT / "data" / "partition_v2.json").read_text())
    assert v2["version"] == 2
    keys = [(r["domain"], r["task_id"]) for b in ("TRAIN", "VAL", "EVAL", "DIAG") for r in v2[b]]
    assert len(keys) == len(set(keys))
    eval_domains = {r["domain"] for r in v2["EVAL"]}
    assert eval_domains == {"airline", "retail", "telecom"}
    # Phase 0 acceptance: VAL large enough for checkpoint selection, telecom EVAL > 8.
    assert len(v2["VAL"]) >= 40
    assert sum(r["domain"] == "telecom" for r in v2["EVAL"]) >= 30
