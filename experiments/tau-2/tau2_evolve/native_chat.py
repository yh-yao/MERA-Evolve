"""Render tau2 conversations exactly as the evaluation endpoint sees them.

Evaluation sends tau2's own OpenAI-format messages and tool schemas to vLLM,
which applies the model's chat template (Qwen3.5: tools rendered into the
system turn, XML tool calls ``<tool_call><function=...><parameter=...>``).
Training must use the same rendering, otherwise RL optimises a format the
evaluated model never sees. Before this module the training prompts carried a
hand-written Hermes/JSON tool block and assistant turns were rewritten as JSON
tool calls -- a different protocol from the one being evaluated.

Two rules keep training tokens aligned with evaluation:

* the initial prompt is ``chat_template(system + history, tools,
  add_generation_prompt=True)`` built from ``tau2.utils.llm_utils.
  to_litellm_messages`` -- the function tau2 itself calls before every request;
* environment turns (tool results, user replies) contribute
  ``render(history + new, gen=True) - render(history, gen=False)``, i.e. the
  exact text the template inserts between the last assistant turn and the
  next generation prompt.

Assistant tokens are the model's own sampled tokens. The only remaining
difference from evaluation is how the template re-renders *older* assistant
turns (``check_native_alignment`` measures it).
"""
from __future__ import annotations

import json
from typing import Any


def litellm_messages(tau2_messages: list[Any], system_prompt: str | None = None) -> list[dict]:
    """tau2 Message objects/dicts -> the OpenAI messages tau2 sends to the model."""
    from tau2_evolve import benchmark  # noqa: F401  (puts the vendored tau2 on sys.path)
    from pydantic import TypeAdapter
    from tau2.data_model.message import Message, SystemMessage
    from tau2.utils.llm_utils import to_litellm_messages

    msgs = [m for m in tau2_messages]
    if msgs and isinstance(msgs[0], dict):
        msgs = TypeAdapter(list[Message]).validate_python(msgs)
    if system_prompt is not None:
        msgs = [SystemMessage(role="system", content=system_prompt), *msgs]
    return to_litellm_messages(msgs)


def template_messages(messages: list[dict]) -> list[dict]:
    """OpenAI messages -> the dicts HF chat templates expect.

    Tool-call arguments are JSON strings on the wire; Qwen templates iterate
    over ``arguments.items()``, which vLLM's server parses back into a dict
    before rendering. Mirror that, and drop None-valued keys.
    """
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                fn["arguments"] = args
                calls.append({"type": "function", "id": c.get("id"), "function": fn})
            m["tool_calls"] = calls
        if m.get("role") == "assistant" and m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def render_text(tokenizer, messages: list[dict], tools: list[dict] | None, *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        template_messages(messages),
        tools=tools or None,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        enable_thinking=False,
    )


def render_ids(tokenizer, messages: list[dict], tools: list[dict] | None, *, add_generation_prompt: bool) -> list[int]:
    text = render_text(tokenizer, messages, tools, add_generation_prompt=add_generation_prompt)
    return tokenizer.encode(text, add_special_tokens=False)


class PrefixMismatch(ValueError):
    pass


_END = "<|im_end|>"


def env_suffix_text(tokenizer, history: list[dict], new_messages: list[dict], tools: list[dict] | None) -> str:
    """Template text inserted after ``history`` for ``new_messages`` + next gen prompt.

    Every rendered message ends with ``<|im_end|>``. The template may re-render
    the *content* of older turns once a new user message arrives (Qwen3 drops
    empty ``<think></think>`` from assistant turns before the latest query),
    so ``render(history)`` is not always a string prefix of
    ``render(history + new)``; the message count is stable, though. The
    inserted text is therefore everything after the N-th ``<|im_end|>`` of
    the full render, N = the number of end markers in the history render.
    """
    base = render_text(tokenizer, history, tools, add_generation_prompt=False)
    full = render_text(tokenizer, history + new_messages, tools, add_generation_prompt=True)
    n = base.count(_END)
    pos = -1
    for _ in range(n):
        pos = full.find(_END, pos + 1)
        if pos < 0:
            raise PrefixMismatch("full render has fewer messages than the history render")
    return full[pos + len(_END):] if n else full


def env_suffix_ids(tokenizer, history: list[dict], new_messages: list[dict], tools: list[dict] | None) -> list[int]:
    return tokenizer.encode(env_suffix_text(tokenizer, history, new_messages, tools), add_special_tokens=False)


def assistant_message(content: str, tool_calls: list[tuple[str, dict]], call_index: int) -> dict:
    """Parsed model output -> OpenAI assistant message (as tau2 would store it)."""
    calls = [
        {
            "id": f"call_{call_index}_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }
        for i, (name, arguments) in enumerate(tool_calls)
    ]
    return {"role": "assistant", "content": content or None, "tool_calls": calls or None}
