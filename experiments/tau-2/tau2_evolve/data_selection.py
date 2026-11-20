"""Auditable TRAIN/AReaL-only data-selection primitives for tau2 SFT.

This module intentionally implements rules, not a learned utility predictor.  A
future utility router may consume its immutable task splits and feature table,
but its labels must come from TRAIN-only proxy-update measurements.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

TaskKey = tuple[str, str]
SELECTORS = ("all", "p_lt_1", "mixed", "p_lt_1_matched")


def task_source(task_id: str) -> str:
    return "areal" if str(task_id).startswith("areal:") else "train"


def task_key(row: dict[str, Any]) -> TaskKey:
    return str(row["domain"]), str(row["task_id"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_partition_membership(path: Path) -> dict[TaskKey, str]:
    payload = json.loads(path.read_text())
    membership: dict[TaskKey, str] = {}
    for bucket in ("TRAIN", "VAL", "EVAL", "DIAG"):
        for row in payload.get(bucket, []):
            key = str(row["domain"]), str(row["task_id"])
            if key in membership:
                raise ValueError(f"partition task appears twice: {key}")
            membership[key] = bucket
    return membership


def load_areal_allowlist(path: Path) -> set[TaskKey]:
    """Load a separately generated set of allowed ``(domain, areal:<id>)`` keys."""
    payload = json.loads(path.read_text())
    values = payload.get("task_keys", payload) if isinstance(payload, dict) else payload
    allowed: set[TaskKey] = set()
    for value in values:
        if isinstance(value, dict):
            key = str(value["domain"]), str(value["task_id"])
        else:
            task_id = str(value)
            key = task_id.removeprefix("areal:").split("_", 1)[0], task_id
        if task_source(key[1]) != "areal":
            raise ValueError(f"AReaL allow-list contains non-AReaL task {key}")
        validate_areal_key(key)
        allowed.add(key)
    if not allowed:
        raise ValueError("AReaL allow-list is empty")
    return allowed


def validate_areal_key(key: TaskKey) -> None:
    domain, task_id = key
    prefix = f"areal:{domain}_"
    if domain not in {"airline", "retail", "telecom"} or not task_id.startswith(prefix):
        raise ValueError(f"invalid AReaL domain/task pairing: {key}")


def validate_allowed_key(
    key: TaskKey, membership: dict[TaskKey, str], areal_allowed: set[TaskKey]
) -> None:
    if task_source(key[1]) == "areal":
        validate_areal_key(key)
        if key not in areal_allowed:
            raise ValueError(f"AReaL task is not in the immutable allow-list: {key}")
        return
    if membership.get(key) != "TRAIN":
        raise ValueError(f"task must be an official TRAIN task, got {key} in {membership.get(key)}")


def load_r1_rates(
    paths: Iterable[Path], *, min_trials: int, membership: dict[TaskKey, str], areal_allowed: set[TaskKey]
) -> dict[TaskKey, dict[str, Any]]:
    """Aggregate validated R1 attempts into per-task success-rate records."""
    grouped: dict[TaskKey, dict[int, dict[str, Any]]] = defaultdict(dict)
    expected_trials = set(range(min_trials))
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                raise ValueError(f"R1 error row: {path}:{line_number}")
            if not isinstance(row.get("passed"), bool):
                raise ValueError(f"R1 passed must be boolean: {path}:{line_number}")
            key = task_key(row)
            validate_allowed_key(key, membership, areal_allowed)
            trial = row.get("trial")
            if not isinstance(trial, int) or isinstance(trial, bool):
                raise ValueError(f"R1 trial must be an integer: {path}:{line_number}")
            if trial in grouped[key]:
                raise ValueError(f"duplicate R1 trial {key} trial={trial}")
            grouped[key][trial] = row
    out: dict[TaskKey, dict[str, Any]] = {}
    for key, trials in grouped.items():
        if set(trials) != expected_trials:
            raise ValueError(f"R1 trials must equal {sorted(expected_trials)} for {key}, got {sorted(trials)}")
        source = task_source(key[1])
        passed = sum(row["passed"] for row in trials.values())
        out[key] = {"source": source, "trials": min_trials, "passed": passed, "p": passed / min_trials}
    return out


def p_bin(rate: float) -> str:
    if rate <= 0.0:
        return "p0"
    if rate >= 1.0:
        return "p1"
    return f"p{rate:.2f}"


def _messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return []
    return list(value)


def message_features(messages: Any) -> dict[str, int]:
    rows = _messages(messages)
    assistant = [row for row in rows if row.get("role") == "assistant"]
    tool_calls = sum(str(row.get("content") or "").count("<tool_call>") for row in assistant)
    chars = sum(len(str(row.get("content") or "")) for row in rows)
    return {"message_count": len(rows), "assistant_turns": len(assistant), "tool_calls": tool_calls, "content_chars": chars}


def validate_candidate(
    key: TaskKey, membership: dict[TaskKey, str], areal_allowed: set[TaskKey]
) -> None:
    validate_allowed_key(key, membership, areal_allowed)


def enrich_candidates(
    rows: Iterable[dict[str, Any]],
    rates: dict[TaskKey, dict[str, Any]],
    membership: dict[TaskKey, str],
    areal_allowed: set[TaskKey],
) -> list[dict[str, Any]]:
    """Join SFT rows with R1 rates and stable trace-shape features."""
    out: list[dict[str, Any]] = []
    seen: set[TaskKey] = set()
    for row in rows:
        key = task_key(row)
        if key in seen:
            raise ValueError(f"duplicate SFT task {key}")
        seen.add(key)
        validate_candidate(key, membership, areal_allowed)
        if key not in rates:
            raise ValueError(f"SFT task missing R1 trials: {key}")
        rate = rates[key]
        features = message_features(row["messages"])
        out.append({
            "domain": key[0], "task_id": key[1], "source": rate["source"],
            "r1_p": rate["p"], "r1_trials": rate["trials"], "r1_passed": rate["passed"],
            "p_bin": p_bin(rate["p"]), **features,
        })
    return out


def selector_accepts(row: dict[str, Any], selector: str) -> bool:
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector {selector!r}; expected {SELECTORS}")
    p = float(row["r1_p"])
    return selector == "all" or (selector in {"p_lt_1", "p_lt_1_matched"} and p < 1.0) or (selector == "mixed" and 0.0 < p < 1.0)


def stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    chars = int(row["content_chars"])
    size = "short" if chars < 24_000 else ("medium" if chars < 36_000 else "long")
    return str(row["source"]), str(row["domain"]), f"{row['p_bin']}:{size}"


def assign_splits(rows: Iterable[dict[str, Any]], *, dev_fraction: float, seed: int) -> dict[TaskKey, str]:
    """Deterministically reserve roughly dev_fraction in each stratum by task."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stratum(row)].append(row)
    assignments: dict[TaskKey, str] = {}
    for group, values in grouped.items():
        ranked = sorted(values, key=lambda row: hashlib.sha256(
            f"{seed}|{row['domain']}|{row['task_id']}".encode()
        ).hexdigest())
        n_dev = round(len(ranked) * dev_fraction)
        if len(ranked) > 1:
            n_dev = max(1, min(len(ranked) - 1, n_dev))
        for index, row in enumerate(ranked):
            assignments[task_key(row)] = "dev" if index < n_dev else "fit"
    return assignments


def control_stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    """Match source/domain/length while allowing p to remain the treatment."""
    source, domain, p_and_size = stratum(row)
    return source, domain, p_and_size.split(":", maxsplit=1)[1]


def matched_p_lt_1_selection(population: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Choose p<1 rows only where a distinct p=1 control exists in its stratum."""
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in population:
        by_group[control_stratum(row)].append(row)
    out: list[dict[str, Any]] = []
    for group, rows in sorted(by_group.items()):
        treatment = [row for row in rows if float(row["r1_p"]) < 1.0]
        capacity = sum(float(row["r1_p"]) == 1.0 for row in rows)
        treatment = sorted(treatment, key=lambda row: hashlib.sha256(
            f"{seed}|treatment|{row['domain']}|{row['task_id']}".encode()
        ).hexdigest())
        out.extend(treatment[:capacity])
    return out


def matched_random(
    population: list[dict[str, Any]], selected: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    """Distinct p=1 control matched on source, domain, and trace size."""
    quotas = Counter(control_stratum(row) for row in selected)
    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in population:
        if float(row["r1_p"]) == 1.0:
            pools[control_stratum(row)].append(row)
    out: list[dict[str, Any]] = []
    for group, quota in sorted(quotas.items()):
        pool = sorted(pools[group], key=lambda row: hashlib.sha256(
            f"{seed}|control|{row['domain']}|{row['task_id']}".encode()
        ).hexdigest())
        if len(pool) < quota:
            raise ValueError(f"insufficient distinct p=1 controls in stratum {group}: {len(pool)} < {quota}")
        out.extend(pool[:quota])
    return out
