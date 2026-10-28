"""verl multi-turn interaction backed by the real tau2 orchestrator."""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Optional
from uuid import uuid4

# Append (do not prepend) tau2's venv so optional voice dependencies missing
# from MERA's venv resolve without overriding verl's transformers/tokenizers.
tau2_site_packages = os.environ.get("TAU2_SITE_PACKAGES")
if tau2_site_packages and tau2_site_packages not in sys.path:
    sys.path.append(tau2_site_packages)

from tau2_evolve import benchmark


class Tau2Interaction:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._instances: dict[str, dict[str, Any]] = {}
        self._official_reward_weight = float(config.get("official_reward_weight", 0.5))

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        domain: str = "",
        task_id: str = "",
        seed: int = 300,
        max_steps: int = 40,
        user_model: str = "openai/evol-llm-user",
        user_base_url: str = "http://127.0.0.1:8211/v1",
        user_thinking: bool | None = None,
        skill_text: str = "",
        expected_initial_user: str = "",
        **_: Any,
    ) -> str:
        from core.schemas.artifacts import RunTaskConfig
        from tau2.data_model.message import UserMessage
        from tau2.data_model.simulation import TextRunConfig
        from tau2.data_model.tasks import Task
        from tau2.orchestrator.orchestrator import Role
        from tau2.runner import build_text_orchestrator
        from tau2.utils.utils import get_now

        instance_id = instance_id or uuid4().hex
        task = Task.model_validate(benchmark.load_task(domain, str(task_id)))
        user_spec = benchmark.make_llm_spec(
            user_model, user_base_url, enable_thinking=user_thinking
        )
        # The agent object is only used to construct policy/tools and is never
        # asked to generate; verl owns every assistant generation.
        agent_spec = benchmark.make_llm_spec("openai/evol-llm-agent", "http://127.0.0.1:1/v1")
        cfg = RunTaskConfig(agent=agent_spec, user=user_spec, seed=seed, max_steps=max_steps, max_errors=10)
        text_cfg = TextRunConfig(
            domain=domain, agent="llm_agent", llm_agent=cfg.agent.model,
            llm_args_agent=dict(cfg.agent.args), user="user_simulator",
            llm_user=cfg.user.model, llm_args_user=dict(cfg.user.args),
            seed=seed, max_steps=max_steps, max_errors=10, num_trials=1,
        )
        benchmark.prime_nl_judge_routing(user_spec)
        from tau2_evolve import areal_tasks

        with areal_tasks.use_db(str(task_id)):  # AReaL tasks bind tools to their own DB snapshot
            orch = build_text_orchestrator(text_cfg, task, seed=seed, simulation_id=instance_id)
        if skill_text:
            orch.agent.domain_policy = f"{orch.agent.domain_policy}\n\n{skill_text}"
        orch._run_start_time = get_now()
        orch._run_start_perf = time.perf_counter()
        orch.initialize()
        if not expected_initial_user:
            orch._cleanup()
            raise ValueError("expected_initial_user is required for deterministic tau2 initialization")
        # The user LLM is sampled and is not deterministic even with a fixed
        # seed. Seed the exact parquet request into both the official tau2
        # trajectory and the simulator's private history instead.
        initial_user = UserMessage(
            role="user", content=expected_initial_user, timestamp=get_now()
        )
        initial_user.validate()
        greeting = orch.trajectory[-1]
        orch.user_state.messages.extend([greeting, initial_user])
        orch.trajectory.append(initial_user)
        orch.message = initial_user
        orch.from_role = Role.USER
        orch.to_role = Role.AGENT
        self._instances[instance_id] = {"orch": orch, "task": task, "score": 0.0, "finalized": False}
        return instance_id

    @staticmethod
    def _parse_agent_message(content: str):
        from tau2.utils.tools import parse_action_string

        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL)
        action = match.group(1) if match else content
        # Hermes emits {"name": ..., "arguments": ...}; tau2's ToolCall
        # accepts that JSON directly.
        return parse_action_string(action, requestor="assistant")

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **_: Any
    ) -> tuple[bool, str, float, dict[str, Any]]:
        from adapters.tau2_bench.adapter import _communication_mode, _evaluation_type_for
        from tau2.data_model.message import ToolMessage
        from tau2.data_model.simulation import TerminationReason
        from tau2.orchestrator.orchestrator import Role

        state = self._instances[instance_id]
        orch = state["orch"]
        content = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        try:
            agent_msg = self._parse_agent_message(content)
            agent_msg.validate()
            orch.trajectory.append(agent_msg)
            orch.message = agent_msg
            orch.from_role = Role.AGENT
            orch.to_role = Role.ENV if agent_msg.is_tool_call() else Role.USER
            if orch.agent.is_stop(agent_msg):
                orch.done = True
                orch.termination_reason = TerminationReason.AGENT_STOP
            if not orch.done:
                orch.step()
                orch._check_termination()
        except Exception as exc:
            orch.done = True
            orch.termination_reason = TerminationReason.AGENT_ERROR
            state["score"] = 0.0
            state["finalized"] = True
            orch._cleanup()
            print(f"[tau2-verl] interaction failed for {instance_id}: {exc}", file=sys.stderr)
            return True, "", 0.0, {
                "tau2_reward": 0.0,
                "observation_role": "user",
                "interaction_error": str(exc),
            }

        observation = orch.message
        if isinstance(observation, ToolMessage):
            response = observation.content or (f"Error: {observation.error}" if observation.error else "")
            observation_role = "tool"
        else:
            response = getattr(observation, "content", "") or ""
            observation_role = "user"

        reward = 0.0
        if orch.done:
            try:
                reward = self._evaluate_state(state, _evaluation_type_for, _communication_mode)
            except Exception as exc:
                print(f"[tau2-verl] evaluation failed for {instance_id}: {exc}", file=sys.stderr)
                state["score"] = 0.0
                state["finalized"] = True
                orch._cleanup()
        return orch.done, response, reward, {"tau2_reward": reward, "observation_role": observation_role}

    async def step_native(
        self, instance_id: str, assistant: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]], float, dict[str, Any]]:
        """Advance tau2 with a parsed, native-format assistant turn.

        ``assistant`` is an OpenAI-format message (content + tool_calls) built
        from the model's own XML tool calls, so the environment receives what
        evaluation's vLLM parser would have produced -- including several
        calls in one turn. Returns (done, new messages in the litellm format
        tau2 sends to the model, reward, extra).
        """
        import json as _json

        from adapters.tau2_bench.adapter import _communication_mode, _evaluation_type_for
        from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
        from tau2.data_model.simulation import TerminationReason
        from tau2.orchestrator.orchestrator import Role

        from tau2_evolve import native_chat

        state = self._instances[instance_id]
        orch = state["orch"]
        try:
            calls = []
            for call in assistant.get("tool_calls") or []:
                fn = call["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = _json.loads(args)
                calls.append(ToolCall(id=call["id"], name=fn["name"], arguments=args, requestor="assistant"))
            agent_msg = AssistantMessage(
                role="assistant", content=assistant.get("content") or None, tool_calls=calls or None
            )
            agent_msg.validate()
            orch.trajectory.append(agent_msg)
            orch.message = agent_msg
            orch.from_role = Role.AGENT
            orch.to_role = Role.ENV if agent_msg.is_tool_call() else Role.USER
            if orch.agent.is_stop(agent_msg):
                orch.done = True
                orch.termination_reason = TerminationReason.AGENT_STOP
            if not orch.done:
                orch.step()
                orch._check_termination()
        except Exception as exc:
            orch.done = True
            orch.termination_reason = TerminationReason.AGENT_ERROR
            state["score"] = 0.0
            state["finalized"] = True
            orch._cleanup()
            print(f"[tau2-verl] native interaction failed for {instance_id}: {exc}", file=sys.stderr)
            return True, [], 0.0, {"tau2_reward": 0.0, "interaction_error": str(exc)}

        observation = orch.message
        if isinstance(observation, MultiToolMessage):
            new = native_chat.litellm_messages(list(observation.tool_messages))
        elif observation is not None and observation is not agent_msg:
            new = native_chat.litellm_messages([observation])
        else:
            new = []

        reward = 0.0
        if orch.done:
            try:
                reward = self._evaluate_state(state, _evaluation_type_for, _communication_mode)
            except Exception as exc:
                print(f"[tau2-verl] evaluation failed for {instance_id}: {exc}", file=sys.stderr)
                state["score"] = 0.0
                state["finalized"] = True
                orch._cleanup()
        return orch.done, new, reward, {"tau2_reward": reward}

    def _evaluate_state(self, state, evaluation_type_for=None, communication_mode=None) -> float:
        if state["finalized"]:
            return float(state["score"])
        if evaluation_type_for is None or communication_mode is None:
            from adapters.tau2_bench.adapter import _communication_mode, _evaluation_type_for
            evaluation_type_for, communication_mode = _evaluation_type_for, _communication_mode
        from tau2.evaluator.evaluator import evaluate_simulation

        orch = state["orch"]
        sim = orch._finalize()
        sim.policy = orch.environment.get_policy()
        from tau2_evolve import areal_tasks

        with areal_tasks.use_db(str(state["task"].id)):  # gold replay must use the same DB snapshot
            info = evaluate_simulation(
                simulation=sim, task=state["task"], evaluation_type=evaluation_type_for(state["task"]),
                solo_mode=getattr(orch, "solo_mode", False), domain=orch.domain, mode=communication_mode(),
            )
        official_reward = float(info.reward)
        _, _, action_recall, _ = benchmark.action_completion(state["task"], sim.messages or [])
        state["score"] = (
            self._official_reward_weight * official_reward
            + (1.0 - self._official_reward_weight) * action_recall
        )
        state["finalized"] = True
        return float(state["score"])

    async def force_terminate(self, instance_id: str, reason: str = "verl_limit") -> float:
        from tau2.data_model.simulation import TerminationReason

        state = self._instances[instance_id]
        orch = state["orch"]
        if not orch.done:
            orch.done = True
            orch.termination_reason = TerminationReason.MAX_STEPS
        try:
            return self._evaluate_state(state)
        except Exception as exc:
            print(f"[tau2-verl] evaluation failed for {instance_id}: {exc}", file=sys.stderr)
            state["score"] = 0.0
            state["finalized"] = True
            orch._cleanup()
            return 0.0

    async def calculate_score(self, instance_id: str, **_: Any) -> float:
        return float(self._instances[instance_id]["score"])

    async def finalize_interaction(self, instance_id: str, **_: Any) -> None:
        state = self._instances.pop(instance_id, None)
        if state and not state["finalized"]:
            state["orch"]._cleanup()
