"""TRAIN-only proxy-result validation and utility-label primitives.

A utility label is valid only when two pre-registered proxy adapters were
measured on the same frozen TRAIN-dev task/trial grid.  This module never reads
VAL/EVAL and refuses manifests marked as synthetic for scientific selection.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["domain"]), str(row["task_id"])


def load_proxy_contract(path: Path) -> dict[str, Any]:
    data = load_json(path)
    required = {"schema_version", "kind", "selector_manifest", "frozen_train_dev", "runs", "label_gate"}
    if data.get("schema_version") != 1 or data.get("kind") != "tau2_data_utility_proxy_contract":
        raise ValueError("not a data-utility proxy contract v1")
    if missing := required - data.keys():
        raise ValueError(f"proxy contract missing {sorted(missing)}")
    if data.get("synthetic_validation_only"):
        raise ValueError("synthetic proxy contract cannot label scientific data")
    return data


def load_trials(path: Path, allowed: set[tuple[str, str]], *, expected_trials: int | None = None) -> dict[tuple[str, str], dict[int, bool]]:
    out: dict[tuple[str, str], dict[int, bool]] = defaultdict(dict)
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            raise ValueError(f"proxy evaluation has error row: {path}:{n}")
        if not isinstance(row.get("passed"), bool):
            raise ValueError(f"proxy evaluation passed must be boolean: {path}:{n}")
        key = task_key(row)
        if key not in allowed:
            raise ValueError(f"proxy evaluation task is outside frozen TRAIN-dev: {key}")
        trial = row.get("trial")
        if not isinstance(trial, int) or isinstance(trial, bool):
            raise ValueError(f"proxy evaluation trial must be integer: {path}:{n}")
        if trial in out[key]:
            raise ValueError(f"duplicate proxy evaluation task/trial: {key}/{trial}")
        out[key][trial] = bool(row.get("passed"))
    if set(out) != allowed:
        raise ValueError("proxy evaluation does not cover exactly frozen TRAIN-dev tasks")
    widths = {tuple(sorted(v)) for v in out.values()}
    if len(widths) != 1:
        raise ValueError("proxy evaluation tasks have inconsistent trial IDs")
    if expected_trials is not None and widths != {tuple(range(expected_trials))}:
        raise ValueError(f"proxy evaluation trials must equal {list(range(expected_trials))}")
    return dict(out)


def paired_summary(control: dict, treatment: dict, *, n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    if control.keys() != treatment.keys():
        raise ValueError("control/treatment tasks differ")
    deltas = []
    domains: dict[str, list[float]] = defaultdict(list)
    for key in sorted(control):
        if set(control[key]) != set(treatment[key]):
            raise ValueError(f"control/treatment trials differ for {key}")
        delta = mean(treatment[key].values()) - mean(control[key].values())
        deltas.append(delta); domains[key[0]].append(delta)
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(n_boot))
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    return {"n_tasks": len(deltas), "delta_pass1": mean(deltas), "ci95": [lo, hi],
            "by_domain": {k: mean(v) for k, v in sorted(domains.items())}}


def utility_label(summary: dict[str, Any], gate: dict[str, float]) -> str:
    delta, lo, hi = summary["delta_pass1"], *summary["ci95"]
    minimum = float(gate["minimum_delta"])
    if delta >= minimum and lo > 0.0:
        return "helpful"
    if delta <= -minimum and hi < 0.0:
        return "harmful"
    return "neutral"


def cohort_key(row: dict[str, Any]) -> str:
    """Predeclared, task-ID-free cohort used for proxy labels and routing."""
    chars = int(row["content_chars"])
    size = "short" if chars < 24_000 else ("medium" if chars < 36_000 else "long")
    return f"{row['source']}|{row['domain']}|{row['p_bin']}|{size}"


def feature_vector(row: dict[str, Any]) -> list[float]:
    """Stable task-ID-free features for a future learned utility router."""
    return [float(row["r1_p"]), float(row["message_count"]), float(row["assistant_turns"]),
            float(row["tool_calls"]), min(float(row["content_chars"]) / 50_000.0, 2.0)]
