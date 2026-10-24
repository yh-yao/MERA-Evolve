"""On-policy distillation from an externally served teacher (Phase 1).

VERL 0.8 ships token-level distillation (``distillation.*`` config,
``verl.trainer.distillation.losses``): after each rollout, the agent-loop
worker asks a teacher for per-token logprobs of the full sequence and the
actor adds a reverse-KL term (k1/k3 estimator, or top-k forward KL).
VERL normally launches the teacher itself inside the Ray job, but

* the tau2 recipe trains with ``one_step_off_policy`` (separate actor and
  rollout GPUs), whose trainer never builds a teacher manager, and
* the 27B teacher is better served by the node's own vLLM (TP=4 on one PCIe
  group, possibly on another node) than by VERL's internal vLLM 0.18.

This module keeps VERL's loss and data path untouched and only replaces the
teacher *transport*: ``ExternalTeacherClient`` implements the
``LLMServerClient.generate`` contract that
``AsyncTeacherLLMServerManager.compute_teacher_logprobs_single`` relies on,
by calling an OpenAI-compatible ``/v1/completions`` endpoint with
``prompt_logprobs``. ``install_opd_patches`` wires it into the one-step
off-policy trainer.

Configuration comes from the environment so it reaches Ray workers:
  MERA_TEACHER_BASE_URL   e.g. http://172.30.16.6:8270/v1   (required)
  MERA_TEACHER_MODEL      served model name (default: evol-llm-teacher)
  MERA_TEACHER_TIMEOUT_S  per-request timeout (default: 600)
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional


def parse_prompt_logprobs(
    prompt_logprobs: list[Optional[dict[str, Any]]],
    prompt_ids: list[int],
    num_prompt_logprobs: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Mirror verl.workers.rollout.vllm_rollout.utils.extract_prompt_logprobs.

    ``prompt_logprobs[i]`` describes token i given tokens < i (entry 0 is
    None). VERL's layout drops entry 0 and appends one dummy row, so row j
    holds the teacher's distribution for predicting token j+1.

    With ``num_prompt_logprobs == 0`` each row is the logprob of the actual
    next token (estimator losses). Otherwise each row holds the teacher's
    top-k ids/logprobs ordered by rank (top-k losses).
    """
    if len(prompt_logprobs) != len(prompt_ids):
        raise ValueError(f"prompt_logprobs has {len(prompt_logprobs)} rows for {len(prompt_ids)} tokens")
    ids_rows: list[list[int]] = []
    lp_rows: list[list[float]] = []
    for position in range(1, len(prompt_logprobs)):
        entry = prompt_logprobs[position] or {}
        actual = prompt_ids[position]
        if num_prompt_logprobs == 0:
            info = entry.get(str(actual), entry.get(actual))
            if info is None:
                raise ValueError(f"teacher returned no logprob for actual token {actual} at {position}")
            lp_rows.append([float(_logprob(info))])
            ids_rows.append([int(actual)])
            continue
        ids = [0] * num_prompt_logprobs
        lps = [float("-inf")] * num_prompt_logprobs
        for token, info in entry.items():
            rank = int(info["rank"]) if isinstance(info, dict) else int(info.rank)
            if 1 <= rank <= num_prompt_logprobs:
                ids[rank - 1] = int(token)
                lps[rank - 1] = float(_logprob(info))
        ids_rows.append(ids)
        lp_rows.append(lps)
    width = max(num_prompt_logprobs, 1)
    ids_rows.append([0] * width)
    lp_rows.append([0.0] * width)
    return ids_rows, lp_rows


def _logprob(info: Any) -> float:
    return info["logprob"] if isinstance(info, dict) else info.logprob


class _TeacherOutput:
    """Duck-typed stand-in for verl's TokenOutput (only extra_fields is read)."""

    def __init__(self, prompt_ids: list[list[int]], prompt_logprobs: list[list[float]]):
        self.token_ids: list[int] = []
        self.log_probs = None
        self.extra_fields = {"prompt_ids": prompt_ids, "prompt_logprobs": prompt_logprobs}


class ExternalTeacherClient:
    """``LLMServerClient.generate`` over an OpenAI-compatible completions API."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 600.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._session = None

    async def _post(self, payload: dict) -> dict:
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s))
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._session.post(f"{self.base_url}/completions", json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    body = (await response.text())[:500]
                    error = RuntimeError(f"teacher HTTP {response.status}: {body}")
                    if response.status < 500:
                        raise error
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                error = exc
            if attempt == self.max_retries:
                raise RuntimeError(f"teacher request failed after {attempt} attempts: {error}")
            await asyncio.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        **_: Any,
    ) -> _TeacherOutput:
        num = int(sampling_params.get("prompt_logprobs") or 0)
        payload = {
            "model": self.model,
            "prompt": list(prompt_ids),
            "max_tokens": 1,
            "temperature": float(sampling_params.get("temperature", 1.0)),
            "prompt_logprobs": num,
            "request_id": request_id,
        }
        data = await self._post(payload)
        rows = data["choices"][0].get("prompt_logprobs")
        if rows is None:
            raise RuntimeError("teacher response has no prompt_logprobs; is the server vLLM?")
        ids, lps = parse_prompt_logprobs(rows, list(prompt_ids), num)
        return _TeacherOutput(ids, lps)


def external_teacher_clients(config) -> dict[str, ExternalTeacherClient]:
    base_url = os.environ.get("MERA_TEACHER_BASE_URL")
    if not base_url:
        raise RuntimeError("distillation is enabled but MERA_TEACHER_BASE_URL is not set")
    model = os.environ.get("MERA_TEACHER_MODEL", "evol-llm-teacher")
    timeout_s = float(os.environ.get("MERA_TEACHER_TIMEOUT_S", "600"))
    # Clients are looked up by the *resolved* teacher key ("default" for a
    # single teacher), not by the YAML entry name ("teacher_model").
    from verl.utils.config import omega_conf_to_dataclass

    keys = list(omega_conf_to_dataclass(config.distillation).teacher_models.keys())
    return {key: ExternalTeacherClient(base_url, model, timeout_s) for key in keys}


def install_opd_patches() -> None:
    """Teach OneStepOffRayTrainer to pass distillation through to VERL's code.

    Idempotent; safe to call from every Ray worker's setup hook.
    """
    from verl.experimental.one_step_off_policy import ray_trainer as one_step
    from verl.single_controller.ray import RayClassWithInitArgs
    from verl.trainer.distillation.losses import is_distillation_enabled
    from verl.trainer.ppo.utils import Role
    from verl.utils.config import omega_conf_to_dataclass

    trainer_cls = one_step.OneStepOffRayTrainer
    if getattr(trainer_cls, "_mera_opd_patched", False):
        return
    original_init_rollout = trainer_cls._init_async_rollout_manager

    def _create_actor_rollout_classes(self):
        # Upstream omits distillation_config, so the actor would never build
        # the distillation loss even with distillation.enabled=true.
        for role in [Role.Actor]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_async_rollout_manager(self):
        if not is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config = None
            self.teacher_model_manager = None
            return original_init_rollout(self)
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            from verl.utils.import_utils import load_class_from_fqn

            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        self.llm_server_manager = one_step.LLMServerManager.create(config=self.config)
        self.async_rollout_mode = True
        self.teacher_model_manager = None  # external server: nothing to sleep/wake
        self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=external_teacher_clients(self.config),
            reward_loop_worker_handles=reward_handles,
        )

    trainer_cls._create_actor_rollout_classes = _create_actor_rollout_classes
    trainer_cls._init_async_rollout_manager = _init_async_rollout_manager
    trainer_cls._mera_opd_patched = True


def install_worker_patches() -> None:
    """Ray ``worker_process_setup_hook``: Qwen3.5 patches plus OPD wiring."""
    from tau2_evolve.qwen35_worker_setup import install_qwen35_worker_patches

    install_qwen35_worker_patches()
    install_opd_patches()
