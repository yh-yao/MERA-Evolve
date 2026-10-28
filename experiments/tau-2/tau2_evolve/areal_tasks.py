"""Run inclusionAI/AReaL-tau2-data RL tasks inside tau2.

Each AReaL task names its own database snapshot (``db_path``). tau2 builds
environments through ``registry.get_env_constructor(domain)`` both when it
starts a simulation and when it evaluates one (gold replay). ``install()``
wraps that lookup so a constructor obtained while ``use_db(domain, path)`` is
active passes the snapshot as ``db``. The selection lives in a contextvar, so
concurrent rollouts (threads or asyncio tasks) never see each other's DB.

Task ids are ``areal:<id>`` (e.g. ``areal:airline_1``). Telecom tasks whose
fault combination equals a VAL/EVAL task of our partition are excluded
(``heldout_telecom_ids``); telecom uses tau2's default database, which the
release ships unchanged.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PREFIX = "areal:"
AREAL_DIR = Path(os.environ.get("MERA_AREAL_DIR", "/mnt/data/users/luzheng/agent_rl/tmp_transfer/areal_tau2"))
_CURRENT: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("areal_db", default=None)
_INSTALLED = False


def is_areal(task_id: str) -> bool:
    return str(task_id).startswith(PREFIX)


@lru_cache(maxsize=1)
def _rows() -> dict[str, dict[str, Any]]:
    out = {}
    with (AREAL_DIR / "tau2_rl_train.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def task_dict(task_id: str) -> dict[str, Any]:
    """tau2 Task dict for ``areal:<id>`` (evaluation_criteria decoded, db_path dropped)."""
    r = dict(_rows()[str(task_id)[len(PREFIX):]])
    ec = r["evaluation_criteria"]
    if isinstance(ec, str):
        ec = json.loads(ec)
    r["evaluation_criteria"] = ec
    r["id"] = str(task_id)
    r.pop("db_path", None)
    return r


def db_path(task_id: str) -> str:
    return str(AREAL_DIR / _rows()[str(task_id)[len(PREFIX):]]["db_path"])


def domain_of(task_id: str) -> str:
    return str(task_id)[len(PREFIX):].split("_")[0]


def _init_signature(init) -> tuple | None:
    if not init:
        return None
    acts = (init.get("initialization_actions") if isinstance(init, dict) else init) or []
    return tuple(sorted(json.dumps({"f": a.get("func_name") or a.get("name"), "a": a.get("arguments")},
                                   sort_keys=True) for a in acts))


@lru_cache(maxsize=1)
def heldout_telecom_ids() -> frozenset[str]:
    """AReaL telecom tasks whose initial_state equals a VAL/EVAL telecom task."""
    from tau2_evolve import benchmark

    held = {tid for b in ("VAL", "EVAL") for d, tid in benchmark.task_ids_for(b, "telecom")}
    sigs = set()
    for t in benchmark._load_all_tasks("telecom"):
        if str(t["id"]) in held:
            s = _init_signature(t.get("initial_state"))
            if s:
                sigs.add(s)
    return frozenset(PREFIX + rid for rid, r in _rows().items()
                     if rid.startswith("telecom") and _init_signature(r.get("initial_state")) in sigs)


def valid_ids() -> set[str] | None:
    """Tasks whose gold replay scores 1 in our tau2 (scripts/check_areal_tasks.py --write-valid).

    About 45% of the release's telecom tasks do not reach their asserted device
    state with our pinned tau2; airline/retail replay cleanly.
    """
    path = AREAL_DIR / "valid_task_ids.json"
    return set(json.loads(path.read_text())) if path.exists() else None


def task_ids(domain: str | None = None) -> list[tuple[str, str]]:
    held = heldout_telecom_ids()
    valid = valid_ids()
    return [(rid.split("_")[0], PREFIX + rid) for rid in _rows()
            if (domain is None or rid.startswith(domain)) and PREFIX + rid not in held
            and (valid is None or PREFIX + rid in valid)]


@lru_cache(maxsize=16)
def _db_template(domain: str, path: str):
    if domain == "airline":
        from tau2.domains.airline.data_model import FlightDB
        return FlightDB.load(path)
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB
        return RetailDB.load(path)
    return None  # telecom: the release ships tau2's default DB


def install() -> None:
    """Wrap tau2's registry so constructors honour ``use_db`` (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    from tau2.registry import registry

    original = registry.get_env_constructor

    def get_env_constructor(name: str):
        ctor = original(name)
        current = _CURRENT.get()
        if current is None or current[0] != name:
            return ctor
        template = _db_template(*current)
        if template is None:
            return ctor

        def build(*args, **kwargs):
            kwargs.setdefault("db", template.model_copy(deep=True))
            return ctor(*args, **kwargs)

        return build

    registry.get_env_constructor = get_env_constructor
    _INSTALLED = True


@contextlib.contextmanager
def use_db(task_id: str):
    """Activate the task's DB snapshot for environments built inside this block."""
    if not is_areal(task_id):
        yield
        return
    install()
    token = _CURRENT.set((domain_of(task_id), db_path(task_id)))
    try:
        yield
    finally:
        _CURRENT.reset(token)
