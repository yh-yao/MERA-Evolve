"""Build SFT parquet from successful tau2 trajectories: one row per passed
task, full conversation as `messages` (system + all turns). Trains on every
assistant decision point in a successful episode (verl's MultiTurnSFTDataset
masks loss by role=="assistant"), not one row per turn.

Tool definitions are rendered directly into the system message's `content`
(replicating Qwen's own `{%- if tools %}` chat-template block by hand)
rather than passed via a separate `tools` column. Passing `tools=` makes
verl's per-turn-concatenation masking diverge from whole-conversation
tokenization (Qwen's template only emits the "# Tools" block once, at the
system turn, conditioned on `tools` being set for the WHOLE render -- the
per-turn concatenation verl uses to build the loss mask doesn't reproduce
that). Silencing the mismatch via `ignore_input_ids_mismatch=True` doesn't
just tolerate a cosmetic difference -- it lets a corrupted loss mask through
and trains to `nan` loss on every step (confirmed empirically: first attempt
with tools-as-column + ignore_input_ids_mismatch=True produced train/loss:nan
and train/grad_norm:0.0 for all 20 steps). Baking tools into `content` and
never passing `tools=` keeps every turn's template rendering independent of
global conversation state, so per-turn concatenation matches whole-message
tokenization exactly, like any ordinary multi-turn conversation.

Usage (either venv works -- no tau2 imports needed here):
  python3 traces_to_sft.py --traces results/tau2_skills_only/train_traces.jsonl \
    --output results/tau2_skills_only/sft_pairs.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import lib_tau2

TOOLS_BLOCK_TEMPLATE = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within <tools></tools> "
    "XML tags:\n<tools>{tool_lines}\n</tools>\n\nFor each function call, return a "
    'json object with function name and arguments within <tool_call></tool_call> '
    'XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": '
    "<args-json-object>}}\n</tool_call>"
)


def _render_tools_block(tools: list[dict]) -> str:
    if not tools:
        return ""
    tool_lines = "".join(f"\n{json.dumps(t, separators=(',', ':'))}" for t in tools)
    return TOOLS_BLOCK_TEMPLATE.format(tool_lines=tool_lines)


def _merge_consecutive_tool_messages(messages: list[dict]) -> list[dict]:
    """Qwen's chat template merges consecutive tool-role messages into ONE
    <tool_response> block when rendering the whole conversation, but verl's
    MultiTurnSFTDataset tokenizes each message in isolation and concatenates
    -- which keeps them as separate <|im_start|>...<|im_end|> turns. That
    structural mismatch (extra im_start/im_end tokens) is real, not cosmetic
    (confirmed by diffing whole-vs-concatenated token ids directly), and
    corrupts the loss if silenced via ignore_input_ids_mismatch=True. Merge
    them ourselves so there's only ever one "tool" turn per run of tool
    results, matching what the whole-conversation template produces.
    """
    merged: list[dict] = []
    for m in messages:
        if m["role"] == "tool" and merged and merged[-1]["role"] == "tool":
            merged[-1] = {**merged[-1], "content": merged[-1]["content"] + "\n" + m["content"]}
        else:
            merged.append(dict(m))
    return merged


def _clean_message(m: dict) -> dict:
    tool_calls = m.get("tool_calls")
    out = {"role": m["role"], "content": m.get("content") or ""}
    if tool_calls:
        rendered_calls = []
        for tc in tool_calls:
            function = tc.get("function") or {}
            name = tc.get("name") or function.get("name")
            arguments = tc.get("arguments") or function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            rendered_calls.append(
                "<tool_call>\n"
                + json.dumps(
                    {"name": name, "arguments": arguments},
                    separators=(",", ":"),
                )
                + "\n</tool_call>"
            )
        # Keep tool calls as rendered assistant text. PyArrow otherwise infers
        # one union struct for every heterogeneous `arguments` dict in the
        # parquet column and fills absent keys with null. Those null keys then
        # become literal training targets and teach the model to pass every
        # domain's arguments to every tool.
        suffix = "\n".join(rendered_calls)
        out["content"] = f"{out['content']}\n{suffix}" if out["content"] else suffix
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows_out = []
    n_total = 0
    n_incomplete = 0
    for line in args.traces.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n_total += 1
        if not row.get("passed"):
            continue
        if not lib_tau2.trace_action_complete(row):
            n_incomplete += 1
            continue
        system_content = row["system_prompt"] + _render_tools_block(row.get("tools", []))
        messages = [{"role": "system", "content": system_content}]
        messages.extend(_clean_message(m) for m in row.get("messages", []))
        messages = _merge_consecutive_tool_messages(messages)
        # Loss is masked by role=="assistant" regardless of which model actually
        # generated the turn -- so in a fallback-rescued row (agent+user both
        # played by the fallback model), only the assistant-side turns become
        # imitation targets, exactly like a genuine main-attempt success. Track
        # fallback_used so downstream reporting can tell "reinforcing what the
        # small model already does" (main success) apart from "teaching it
        # something it couldn't do itself" (fallback rescue) -- same
        # teacher-pair vs self-repair-pair distinction as verl_code_rl's
        # traces_to_sft.py, just for a multi-turn agent instead of one-shot code.
        rows_out.append({
            "domain": row["domain"],
            "task_id": row["task_id"],
            "messages": messages,
            "fallback_used": bool(row.get("fallback_used", False)),
        })

    n_fallback = sum(r["fallback_used"] for r in rows_out)
    print(
        f"[traces_to_sft] {len(rows_out)}/{n_total} trajectories passed -> SFT rows "
        f"({len(rows_out) - n_fallback} main-success, {n_fallback} fallback-rescued/teacher; "
        f"excluded {n_incomplete} passed but action-incomplete)"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_parquet(args.output)
    print(f"[traces_to_sft] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
