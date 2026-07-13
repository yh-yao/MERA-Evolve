"""Shared tau2-bench helpers for MERA-Evolve's experiments/tau-2/ scripts.

Works from EITHER the tau2_stage2 venv (has `tau2` + `core.schemas`
importable via its .pth-installed editable package) or MERA-Evolve's own
venv (has torch/transformers/verl) -- both need `tau2`
itself, which is a vendored SOURCE tree, not a real pip-installed package
(confirmed via .venv_tau2's .pth file: it just adds vendor/tau2-bench/src to
sys.path), so adding that same path here works from any interpreter.

Writes its own task-runner instead of calling Tau2BenchAdapter.run_task()
directly, because skill injection needs to mutate `orch.agent.domain_policy`
between `build_text_orchestrator()` and `orch.run()` -- a hook the adapter
doesn't expose. The rest (loaders, NL-judge routing, step extraction) is
reused via the adapter module's own helpers to stay in sync with it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TAU2_STAGE2_CODE = Path("/shared_home/yuhang.yao/router-skills-evolve/tau2_stage2/code")
VENDOR_ROOT = TAU2_STAGE2_CODE / "vendor" / "tau2-bench"
PARTITION_PATH = Path(
    "/shared_home/yuhang.yao/router-skills-evolve/tau2_stage2/data_processed/stage2_v1/partition.json"
)

for _p in (TAU2_STAGE2_CODE, VENDOR_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# MERA's venv intentionally owns torch/transformers/verl. Append tau2's
# site-packages only as a fallback for optional simulation dependencies.
TAU2_SITE_PACKAGES = Path(
    "/shared_home/yuhang.yao/router-skills-evolve/.venv_tau2/lib/python3.12/site-packages"
)
if str(TAU2_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(TAU2_SITE_PACKAGES))

DOMAINS = ["airline", "retail", "telecom"]


def load_partition() -> dict[str, list[dict[str, str]]]:
    return json.loads(PARTITION_PATH.read_text())


def task_ids_for(bucket: str, domain: str | None = None) -> list[tuple[str, str]]:
    """Return [(domain, task_id), ...] for a partition bucket (TRAIN/EVAL/DIAG/VAL).

    telecom's partition.json task_ids ("mob_E_002" etc.) don't resolve to any
    loadable tau2 task (their mapping script isn't shipped -- see
    _load_all_tasks) -- substituted here with a fresh split of tau2's own
    get_tasks_small() (20 real tasks): first 8 for EVAL, remaining 12 for
    TRAIN. Not comparable to tau2_stage2's historical telecom numbers, but a
    real, loadable telecom set for our own skills_only/sft_only baselines.
    """
    if bucket in ("TRAIN", "EVAL") and (domain is None or domain == "telecom"):
        real_telecom_ids = telecom_small_task_ids()
        telecom_eval_ids = real_telecom_ids[:8]
        telecom_train_ids = real_telecom_ids[8:]
        telecom_rows = [
            ("telecom", tid) for tid in (telecom_eval_ids if bucket == "EVAL" else telecom_train_ids)
        ]
    else:
        telecom_rows = []

    rows = load_partition()[bucket]
    if domain:
        rows = [r for r in rows if r["domain"] == domain]
    out = [(r["domain"], r["task_id"]) for r in rows if r["domain"] != "telecom"]
    if domain is None or domain == "telecom":
        out.extend(telecom_rows)
    return out


def _adapter():
    from adapters.tau2_bench.adapter import Tau2BenchAdapter

    return Tau2BenchAdapter(vendor_root=VENDOR_ROOT)


_nl_judge_primed = False


def prime_nl_judge_routing(user_spec) -> None:
    """Route tau2's module-level NL judge to the active user-model endpoint.

    The stage2 adapter defaults the judge to ``openai/gpt-5.2`` while reusing
    the caller's API URL. Our caller URL is a local vLLM server that only
    serves ``evol-llm-user``, so retail evaluations requiring NL assertions
    otherwise fail with a 404. Patch both copied globals directly and only
    once per process, before concurrent collection or rollout work starts.
    """
    global _nl_judge_primed
    if _nl_judge_primed:
        return
    from tau2 import config as tau2_config
    from tau2.evaluator import evaluator_nl_assertions as nl_mod

    model = user_spec.model
    args = {
        **dict(user_spec.args),
        "temperature": tau2_config.DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE,
        "response_format": {"type": "json_object"},
    }
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = model
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    nl_mod.DEFAULT_LLM_NL_ASSERTIONS = model
    nl_mod.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    _nl_judge_primed = True


_TASKS_CACHE: dict[str, list[dict[str, Any]]] = {}


def _load_all_tasks(domain: str) -> list[dict[str, Any]]:
    """airline/retail load from tasks.json via the adapter; telecom's tasks.json
    (2285 auto-generated entries) doesn't match tau2_stage2's curated
    partition.json task_ids (e.g. "mob_E_002") -- that ID scheme's mapping
    script isn't shipped in the bundle, so telecom uses tau2's own
    get_tasks_small() (20 real, natively-IDed tasks) as a fresh substitute
    instead of trying to reverse-engineer the lost mapping.
    """
    if domain == "telecom":
        from tau2.domains.telecom.environment import get_tasks_small

        return [t.model_dump(mode="json") for t in get_tasks_small()]
    return _adapter().load_tasks(domain)


def load_task(domain: str, task_id: str) -> dict[str, Any]:
    if domain not in _TASKS_CACHE:
        _TASKS_CACHE[domain] = _load_all_tasks(domain)
    return next(t for t in _TASKS_CACHE[domain] if str(t["id"]) == str(task_id))


def telecom_small_task_ids() -> list[str]:
    return [t["id"] for t in _load_all_tasks("telecom")]


def make_llm_spec(model: str, api_base: str, api_key: str = "EMPTY"):
    from core.schemas.artifacts import LLMSpec

    return LLMSpec(model=model, args={"api_base": api_base, "api_key": api_key})


def build_agent_context(
    *,
    domain: str,
    task_id: str,
    user_spec,
    seed: int = 300,
    max_steps: int = 40,
    skill_text: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Build the exact tau2 agent prompt/tools without running a simulation."""
    from adapters.tau2_bench.adapter import _capture_agent_context
    from core.schemas.artifacts import RunTaskConfig
    from tau2.data_model.simulation import TextRunConfig
    from tau2.data_model.tasks import Task
    from tau2.runner import build_text_orchestrator

    agent_spec = make_llm_spec("openai/evol-llm-agent", "http://127.0.0.1:1/v1")
    config = RunTaskConfig(
        agent=agent_spec, user=user_spec, seed=seed, max_steps=max_steps, max_errors=10
    )
    text_cfg = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        llm_agent=config.agent.model,
        llm_args_agent=dict(config.agent.args),
        user="user_simulator",
        llm_user=config.user.model,
        llm_args_user=dict(config.user.args),
        seed=seed,
        max_steps=max_steps,
        max_errors=10,
        num_trials=1,
    )
    task = Task.model_validate(load_task(domain, task_id))
    prime_nl_judge_routing(user_spec)
    orch = build_text_orchestrator(text_cfg, task, seed=seed)
    if skill_text:
        orch.agent.domain_policy = f"{orch.agent.domain_policy}\n\n{skill_text}"
    return _capture_agent_context(orch)


def run_task(
    *,
    domain: str,
    task_id: str,
    agent_spec,
    user_spec,
    seed: int = 300,
    max_steps: int = 60,
    max_errors: int = 10,
    skill_text: str = "",
) -> dict[str, Any]:
    """Run one tau2 task end-to-end; optionally inject skill text into the
    agent's domain policy. Returns a plain-dict trace record (JSON-safe).
    """
    from core.schemas.artifacts import RunTaskConfig
    from tau2.data_model.tasks import Task
    from tau2.evaluator.evaluator import evaluate_simulation
    from tau2.runner import build_text_orchestrator
    from tau2.data_model.simulation import TextRunConfig

    from adapters.tau2_bench.adapter import (
        _capture_agent_context,
        _communication_mode,
        _evaluation_type_for,
    )

    task_dict = load_task(domain, task_id)
    task_obj = Task.model_validate(task_dict)
    config = RunTaskConfig(agent=agent_spec, user=user_spec, seed=seed, max_steps=max_steps, max_errors=max_errors)

    text_cfg = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        llm_agent=config.agent.model,
        llm_args_agent=dict(config.agent.args),
        user="user_simulator",
        llm_user=config.user.model,
        llm_args_user=dict(config.user.args),
        seed=config.seed,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        num_trials=1,
    )

    prime_nl_judge_routing(user_spec)
    orch = build_text_orchestrator(text_cfg, task_obj, seed=config.seed)
    if skill_text:
        orch.agent.domain_policy = f"{orch.agent.domain_policy}\n\n{skill_text}"
    system_prompt, tools = _capture_agent_context(orch)

    sim = orch.run()
    sim.policy = orch.environment.get_policy()
    try:
        sim.reward_info = evaluate_simulation(
            simulation=sim,
            task=task_obj,
            evaluation_type=_evaluation_type_for(task_obj),
            solo_mode=getattr(orch, "solo_mode", False),
            domain=domain,
            mode=_communication_mode(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[lib_tau2] evaluation failed for {domain}/{task_id}: {e}", file=sys.stderr)
        sim.reward_info = None

    reward = float(sim.reward_info.reward) if sim.reward_info else 0.0
    messages = []
    for m in (sim.messages or []):
        messages.append(m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m))

    return {
        "domain": domain,
        "task_id": task_id,
        "passed": reward > 0.0,
        "reward": reward,
        "termination_reason": str(sim.termination_reason),
        "n_steps": len(messages),
        "agent_cost_usd": float(getattr(sim, "agent_cost", 0.0) or 0.0),
        "user_cost_usd": float(getattr(sim, "user_cost", 0.0) or 0.0),
        "system_prompt": system_prompt,
        "tools": tools,
        "messages": messages,
        "used_skill": bool(skill_text),
    }
