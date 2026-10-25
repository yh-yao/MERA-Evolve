"""Verify that training-time token construction matches evaluation prompts.

For each assistant turn t of real evaluation traces (collect_traces output,
which stores the agent system prompt, OpenAI tool schemas and tau2 messages)
this compares

  eval:   chat_template(system + messages[:t], tools, add_generation_prompt)
          -- what vLLM rendered for that request
  train:  initial prompt + sum over earlier turns of
          (assistant tokens + native_chat.env_suffix_ids(...))
          -- what Tau2AgentLoop accumulates

Assistant tokens are approximated by the template's own rendering of the
assistant turn as the latest message (a model sampling in-format produces
exactly this). Reports the exact-match rate and classifies mismatches.

Usage (training venv):
  python -m tau2_evolve.check_native_alignment --traces eval.jsonl [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tau2_evolve import native_chat


def assistant_turn_text(tokenizer, history, message, tools) -> str:
    """Text a model emits for ``message`` after the generation prompt."""
    with_gen = native_chat.render_text(tokenizer, history, tools, add_generation_prompt=True)
    full = native_chat.render_text(tokenizer, history + [message], tools, add_generation_prompt=False)
    if not full.startswith(with_gen):
        raise native_chat.PrefixMismatch("assistant render does not extend the generation prompt")
    text = full[len(with_gen):]
    end = text.find("<|im_end|>")
    return text[: end + len("<|im_end|>")] if end >= 0 else text


def check_trace(tokenizer, row) -> tuple[int, int, Counter]:
    tools = row["tools"]
    msgs = native_chat.litellm_messages(row["messages"], row["system_prompt"])
    agent_turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 1]
    # Training starts generating at the first assistant turn after the
    # greeting + first user message (the parquet prompt).
    first = next(i for i, m in enumerate(msgs) if m["role"] == "user")
    turns = [i for i in agent_turns if i > first]
    ok = total = 0
    kinds: Counter = Counter()
    if not turns:
        return ok, total, kinds
    history = msgs[: turns[0]]
    train = native_chat.render_ids(tokenizer, history, tools, add_generation_prompt=True)
    for k, t in enumerate(turns):
        if k > 0:
            prev = turns[k - 1]
            asst_text = assistant_turn_text(tokenizer, msgs[:prev], msgs[prev], tools)
            train += tokenizer.encode(asst_text, add_special_tokens=False)
            train += native_chat.env_suffix_ids(tokenizer, msgs[: prev + 1], msgs[prev + 1 : t], tools)
        eval_ids = native_chat.render_ids(tokenizer, msgs[:t], tools, add_generation_prompt=True)
        total += 1
        if train == eval_ids:
            ok += 1
            continue
        a = tokenizer.decode(train)
        b = tokenizer.decode(eval_ids)
        removed = a.replace("<think>\n\n</think>\n\n", "")
        kinds["only_empty_think_in_history" if removed == b.replace("<think>\n\n</think>\n\n", "") else "other"] += 1
        if kinds["other"] == 1 and removed != b.replace("<think>\n\n</think>\n\n", ""):
            i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print(f"[other mismatch] {row['domain']}/{row['task_id']} turn {t}:\n  train: {a[max(0,i-120):i+120]!r}\n  eval : {b[max(0,i-120):i+120]!r}", file=sys.stderr)
    return ok, total, kinds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--model", default="/mnt/data/users/luzheng/agent_rl/models/Qwen3.5-2B")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ok = total = 0
    kinds: Counter = Counter()
    rows = [json.loads(l) for l in args.traces.read_text().splitlines() if l.strip()][: args.limit]
    for row in rows:
        if row.get("error") or not row.get("messages"):
            continue
        o, t, k = check_trace(tok, row)
        ok += o; total += t; kinds.update(k)
    print(json.dumps({"traces": len(rows), "assistant_turns": total, "exact_match": ok,
                      "exact_match_rate": ok / max(total, 1), "mismatch_kinds": dict(kinds)}))
    return 0 if total and kinds.get("other", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
