"""Tau2 specialization of verl's native multi-turn ToolAgentLoop."""
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


@register("tau2_agent")
class Tau2AgentLoop(ToolAgentLoop):
    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        if state == AgentState.TERMINATED and agent_data.interaction is not None:
            reward = await agent_data.interaction.force_terminate(agent_data.request_id)
            agent_data.turn_scores.append(reward)
            await agent_data.interaction.finalize_interaction(agent_data.request_id)
        if state == AgentState.PROCESSING_TOOLS:
            # Tau2Interaction owns task-specific tool execution. The schemas
            # vary by domain, so do not route calls through verl's static tool
            # registry; pass the decoded assistant turn to the interaction.
            agent_data.tool_calls = []
            return AgentState.INTERACTING
        return state

    async def _handle_interacting_state(self, agent_data):
        terminate, response, reward, extra = await agent_data.interaction.generate_response(
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
            await agent_data.interaction.finalize_interaction(agent_data.request_id)
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def run(self, sampling_params: dict, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        scores = output.extra_fields.get("turn_scores") or []
        # Tau2Interaction emits zero for intermediate turns and the official
        # terminal environment score on the final turn.
        output.reward_score = float(sum(scores))
        return output
