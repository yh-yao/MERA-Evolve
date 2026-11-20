from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TAU2_EXPERIMENT = Path(__file__).parents[1] / "experiments" / "tau-2"
sys.path.insert(0, str(TAU2_EXPERIMENT))

from tau2_evolve import data_selection as select  # noqa: E402


def _partition(tmp_path: Path) -> Path:
    path = tmp_path / "partition.json"
    path.write_text(json.dumps({"version": 2,
        "TRAIN": [{"domain": "retail", "task_id": "1"}, {"domain": "airline", "task_id": "2"}],
        "VAL": [{"domain": "retail", "task_id": "3"}],
        "EVAL": [{"domain": "retail", "task_id": "4"}], "DIAG": []}))
    return path


def _allowlist(tmp_path: Path) -> Path:
    path = tmp_path / "areal.json"
    path.write_text(json.dumps(["areal:retail_1", "areal:retail_2"]))
    return path


def _r1(domain: str, task_id: str, passed: list[bool]) -> list[dict]:
    return [{"domain": domain, "task_id": task_id, "trial": trial, "passed": ok}
            for trial, ok in enumerate(passed)]


def test_load_r1_rates_validates_trials_types_and_membership(tmp_path: Path) -> None:
    partition = select.load_partition_membership(_partition(tmp_path))
    allowlist = select.load_areal_allowlist(_allowlist(tmp_path))
    path = tmp_path / "r1.jsonl"
    rows = _r1("retail", "1", [True, False, True, False])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rates = select.load_r1_rates([path], min_trials=4, membership=partition, areal_allowed=allowlist)
    assert rates[("retail", "1")]["p"] == 0.5

    bad = [*rows]
    bad[0] = {**bad[0], "passed": "false"}
    path.write_text("".join(json.dumps(row) + "\n" for row in bad))
    with pytest.raises(ValueError, match="must be boolean"):
        select.load_r1_rates([path], min_trials=4, membership=partition, areal_allowed=allowlist)
    path.write_text("".join(json.dumps({**row, "trial": row["trial"] + 1}) + "\n" for row in rows))
    with pytest.raises(ValueError, match="trials must equal"):
        select.load_r1_rates([path], min_trials=4, membership=partition, areal_allowed=allowlist)


def test_enrich_candidates_rejects_heldout_and_unlisted_areal(tmp_path: Path) -> None:
    membership = select.load_partition_membership(_partition(tmp_path))
    allowed = select.load_areal_allowlist(_allowlist(tmp_path))
    rates = {("retail", "1"): {"source": "train", "trials": 4, "passed": 2, "p": 0.5},
             ("retail", "3"): {"source": "train", "trials": 4, "passed": 2, "p": 0.5},
             ("retail", "areal:retail_99"): {"source": "areal", "trials": 4, "passed": 2, "p": 0.5}}
    row = {"domain": "retail", "task_id": "1", "messages": [{"role": "assistant", "content": "<tool_call>"}]}
    assert select.enrich_candidates([row], rates, membership, allowed)[0]["tool_calls"] == 1
    with pytest.raises(ValueError, match="official TRAIN"):
        select.enrich_candidates([{**row, "task_id": "3"}], rates, membership, allowed)
    with pytest.raises(ValueError, match="allow-list"):
        select.enrich_candidates([{**row, "task_id": "areal:retail_99"}], rates, membership, allowed)


def test_split_and_matched_controls_are_disjoint_and_deterministic() -> None:
    population = [{"domain": "retail", "task_id": str(index), "source": "train", "r1_p": p,
                   "p_bin": select.p_bin(p), "content_chars": 10_000}
                  for index, p in enumerate((0.25, 0.5, 0.75, 1.0, 1.0, 1.0))]
    selection = select.matched_p_lt_1_selection(population, seed=2)
    control = select.matched_random(population, selection, seed=2)
    assert len(selection) == len(control) == 3
    assert not {select.task_key(row) for row in selection} & {select.task_key(row) for row in control}
    splits = select.assign_splits(population, dev_fraction=0.4, seed=7)
    assert splits == select.assign_splits(population, dev_fraction=0.4, seed=7)
    assert set(splits.values()) == {"fit", "dev"}
