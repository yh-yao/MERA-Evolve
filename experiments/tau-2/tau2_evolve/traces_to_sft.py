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
  PYTHONPATH=experiments/tau-2 python3 -m tau2_evolve.traces_to_sft \
    --traces results/tau2_skills_only/train_traces.jsonl \
    --output results/tau2_skills_only/sft_pairs.parquet
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from tau2_evolve import benchmark

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


def _drop_pre_user_assistant_messages(messages: list[dict]) -> list[dict]:
    """Drop TAU's generic agent greeting before the simulator's first request.

    Qwen3.5 requires a user query before an assistant turn. The greeting is
    fixed orchestration boilerplate rather than a task decision, so it should
    not be an imitation target.
    """
    first_user = next(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        len(messages),
    )
    return [
        message for index, message in enumerate(messages)
        if index >= first_user or message["role"] != "assistant"
    ]


def _assistant_is_trainable(message: dict) -> bool:
    """Treat Parquet nulls as an omitted marker, not an explicit mask."""
    marker = message.get("_trainable", True)
    return True if marker is None else bool(marker)


def _has_assistant_target(messages: list[dict]) -> bool:
    return any(
        message["role"] == "assistant"
        and _assistant_is_trainable(message)
        and str(message.get("content") or "").strip()
        for message in messages
    )


def _clean_message(m: dict) -> dict:
    tool_calls = m.get("tool_calls")
    out = {"role": m["role"], "content": m.get("content") or ""}
    if "_trainable" in m:
        out["_trainable"] = bool(m["_trainable"])
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
        # TAU requires either text or a tool call in one assistant turn. Keep
        # only the canonical tool payload even if the serving model emitted a
        # prose preamble; imitating mixed turns caused malformed tool calls in
        # later cycles.
        out["content"] = "\n".join(rendered_calls)
    return out


def _balance_domains(rows: list[dict], seed: int = 42) -> list[dict]:
    """Deterministically oversample each domain to the largest domain size."""
    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    if not by_domain:
        return []

    target = max(len(domain_rows) for domain_rows in by_domain.values())
    balanced = []
    for domain in sorted(by_domain):
        domain_rows = by_domain[domain]
        balanced.extend(domain_rows[i % len(domain_rows)] for i in range(target))
    random.Random(seed).shuffle(balanced)
    return balanced


def _deduplicate_tasks(rows: list[dict]) -> list[dict]:
    """Keep the newest preferred trajectory for each benchmark task.

    Callers pass current-cycle student successes and OPD repairs before older
    cycle traces. Preserving the first row therefore prevents repeated tasks
    from receiving progressively more weight across cycles without discarding
    a current repair in favor of stale history.
    """
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in rows:
        key = (row["domain"], str(row["task_id"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--balance-domains", action="store_true")
    ap.add_argument(
        "--deduplicate-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only the first trajectory for each domain/task (default: true)",
    )
    args = ap.parse_args()

    rows_out = []
    n_total = 0
    n_incomplete = 0
    n_without_target = 0
    for traces_path in args.traces:
        for line in traces_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            n_total += 1
            if not row.get("passed"):
                continue
            if not benchmark.trace_action_complete(row):
                n_incomplete += 1
                continue
            system_content = row["system_prompt"] + _render_tools_block(row.get("tools", []))
            messages = [{"role": "system", "content": system_content}]
            branch_index = int(row.get("opd_branch_message_index", 0) or 0)
            for message_index, message in enumerate(row.get("messages", [])):
                cleaned = _clean_message(message)
                if row.get("opd_used") and cleaned["role"] == "assistant":
                    cleaned["_trainable"] = message_index >= branch_index
                messages.append(cleaned)
            messages = _drop_pre_user_assistant_messages(messages)
            messages = _merge_consecutive_tool_messages(messages)
            if not _has_assistant_target(messages):
                n_without_target += 1
                continue
            # OPD rows preserve the replayed student prefix for context while
            # `_trainable` limits loss to the verified teacher continuation.
            rows_out.append({
                "domain": row["domain"],
                "task_id": row["task_id"],
                "messages": messages,
                "fallback_used": bool(row.get("fallback_used", False)),
                "opd_used": bool(row.get("opd_used", False)),
            })

    n_fallback = sum(r["fallback_used"] for r in rows_out)
    n_opd = sum(r["opd_used"] for r in rows_out)
    n_unbalanced = len(rows_out)
    if args.deduplicate_tasks:
        rows_out = _deduplicate_tasks(rows_out)
    n_unique = len(rows_out)
    if args.balance_domains:
        rows_out = _balance_domains(rows_out)
    print(
        f"[traces_to_sft] {n_unbalanced}/{n_total} trajectories passed -> SFT rows "
        f"({n_unbalanced - n_fallback - n_opd} student-success, {n_opd} OPD, "
        f"{n_fallback} legacy fallback; "
        f"excluded {n_incomplete} passed but action-incomplete and "
        f"{n_without_target} without assistant targets)"
    )
    if args.deduplicate_tasks:
        print(f"[traces_to_sft] task-deduplicated {n_unbalanced} -> {n_unique} rows")
    if args.balance_domains:
        print(f"[traces_to_sft] domain-balanced {n_unique} -> {len(rows_out)} rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_parquet(args.output)
    print(f"[traces_to_sft] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
