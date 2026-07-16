"""Tau2 specialization of verl's native multi-turn ToolAgentLoop."""
import json
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

from tau2_evolve.interaction import Tau2Interaction


_INTERACTING = object()


def _bounded_sampling_params(
    sampling_params: dict,
    *,
    prompt_tokens: int,
    response_tokens: int,
    max_model_length: int,
    max_response_length: int,
) -> dict | None:
    """Limit one generation to the context and rollout space still available."""
    available = min(
        max_model_length - prompt_tokens,
        max_response_length - response_tokens,
    )
    if available <= 0:
        return None

    bounded = dict(sampling_params)
    configured = bounded.get("max_tokens")
    if configured is None or int(configured) > available:
        bounded["max_tokens"] = available
    return bounded


@register("tau2_agent")
class Tau2AgentLoop(ToolAgentLoop):
    """Drive TAU-2 interactions on VERL versions without ``verl.interactions``.

    VERL 0.8 removed its generic interaction registry from ToolAgentLoop. TAU-2
    still needs an environment turn after every assistant generation, so this
    subclass restores that one state while retaining VERL's current rollout,
    token accounting, and tool parser implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction = Tau2Interaction({"official_reward_weight": 0.5})
        configured_model_length = getattr(self.rollout_config, "max_model_len", None)
        self.max_model_length = int(
            configured_model_length or self.prompt_length + self.response_length
        )

    async def run(self, sampling_params: dict, **kwargs):
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        request_id = uuid4().hex
        interaction_kwargs = (kwargs.get("extra_info") or {}).get("interaction_kwargs", {})
        await self.interaction.start_interaction(request_id, **interaction_kwargs)

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=self._get_mm_processor_kwargs(audios),
            metrics={},
            request_id=request_id,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )
        agent_data.interaction = self.interaction
        agent_data.interaction_kwargs = interaction_kwargs

        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                for tool in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state is _INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                reward = await self.interaction.force_terminate(request_id)
                agent_data.turn_scores.append(reward)
                await self.interaction.finalize_interaction(request_id)
                state = AgentState.TERMINATED

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        output_multi_modal_data = {}
        if agent_data.image_data is not None:
            output_multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            output_multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            output_multi_modal_data["audios"] = agent_data.audio_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=output_multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs
                else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update(
            {"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards}
        )
        output.reward_score = float(sum(agent_data.turn_scores))
        return output

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        bounded_sampling_params = _bounded_sampling_params(
            sampling_params,
            prompt_tokens=len(agent_data.prompt_ids),
            response_tokens=len(agent_data.response_mask),
            max_model_length=self.max_model_length,
            max_response_length=self.response_length,
        )
        if bounded_sampling_params is None:
            agent_data.metrics["context_limit_terminated"] = 1
            reward = await self.interaction.force_terminate(agent_data.request_id)
            agent_data.turn_scores.append(reward)
            await self.interaction.finalize_interaction(agent_data.request_id)
            return AgentState.TERMINATED

        state = await super()._handle_generating_state(
            agent_data, bounded_sampling_params, ignore_termination=ignore_termination
        )
        if agent_data.tool_calls:
            call = agent_data.tool_calls[0]
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = call.arguments
            assistant_message = "<tool_call>\n" + json.dumps(
                {"name": call.name, "arguments": arguments}, ensure_ascii=False
            ) + "\n</tool_call>"
        else:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
        agent_data.messages.append({"role": "assistant", "content": assistant_message})

        reached_limit = (
            len(agent_data.response_mask) >= self.response_length
            or bool(self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
            or bool(self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        )
        if reached_limit:
            reward = await self.interaction.force_terminate(agent_data.request_id)
            agent_data.turn_scores.append(reward)
            await self.interaction.finalize_interaction(agent_data.request_id)
            return AgentState.TERMINATED

        # Tau2Interaction owns task-specific tool execution. Tool schemas vary
        # by domain, so every generated assistant turn goes to the environment.
        agent_data.tool_calls = []
        return _INTERACTING

    async def _handle_interacting_state(self, agent_data):
        terminate, response, reward, extra = await self.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1
        role = extra.get("observation_role", "user")
        if reward is not None:
            agent_data.turn_scores.append(reward)
        if response:
            add_messages = [{"role": role, "content": response}]
            agent_data.messages.extend(add_messages)
            response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)
        if terminate:
            await self.interaction.finalize_interaction(agent_data.request_id)
            return AgentState.TERMINATED
        return AgentState.GENERATING
