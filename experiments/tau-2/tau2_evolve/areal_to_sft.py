"""Convert inclusionAI/AReaL-tau2-data SFT turns into whole-dialog native SFT rows.

The release stores one row per assistant turn (history + answer). All turns
of a dialog share one teacher conversation, so the longest row per dialog plus
its answer is the whole dialog; every assistant turn in it becomes trainable
(except tau2's fixed greeting).

Changes made so the rows match what our evaluated student sees:

* the release's system prompt (which adds a "think in <think> tags" block) is
  replaced with tau2's own agent system prompt for the domain, and tools come
  from tau2's schemas (both taken from one of our own traces per domain);
* thinking is dropped (the student is evaluated with thinking disabled);
* only dialogs labelled ``correct == 1`` are kept;
* telecom dialogs carry an official tau2 ``task_id``: any dialog whose task
  is in our VAL or EVAL split is dropped (TRAIN tasks are allowed);
* dialogs longer than ``--max-tokens`` (Qwen3.5 tokenizer) are dropped.

Usage (training venv):
  python -m tau2_evolve.areal_to_sft --areal-dir .../areal_tau2 --reference-traces train_greedy.jsonl \\
      --partition data/partition_v2.json --output areal_sft.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tau2_evolve import native_chat

GREETING = "Hi! How can I help you today?"
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def domain_of(dialog_id: str) -> str:
    head = dialog_id.split("_")[0]
    return head if head in ("airline", "retail") else "telecom"


def clean_assistant(m: dict) -> dict:
    out = {"role": "assistant", "content": _THINK.sub("", m.get("content") or "").strip() or None}
    if m.get("tool_calls"):
        calls = []
        for i, c in enumerate(m["tool_calls"]):
            fn = c.get("function") or {"name": c.get("name"), "arguments": c.get("arguments")}
            args = fn.get("arguments")
            calls.append({"id": c.get("id") or f"call_{i}", "type": "function",
                          "function": {"name": fn["name"],
                                       "arguments": args if isinstance(args, str) else json.dumps(args)}})
        out["tool_calls"] = calls
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--areal-dir", type=Path, required=True)
    ap.add_argument("--reference-traces", type=Path, action="append", required=True,
                    help="our own traces; supply tau2's system prompt and tools per domain")
    ap.add_argument("--partition", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokenizer", default="/mnt/data/users/luzheng/agent_rl/models/Qwen3.5-2B")
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--domains", nargs="+", default=["airline", "retail", "telecom"])
    args = ap.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer

    ref: dict[str, dict] = {}
    for path in args.reference_traces:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            ref.setdefault(r["domain"], {"system_prompt": r["system_prompt"], "tools": r["tools"]})
    tool_names = {d: {t["function"]["name"] for t in v["tools"]} for d, v in ref.items()}

    part = json.loads(args.partition.read_text())
    held = {str(t["task_id"]) for b in ("VAL", "EVAL") for t in part[b] if t["domain"] == "telecom"}

    last: dict[str, dict] = {}
    with (args.areal_dir / "tau2_sft_train.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            d = r["metadata"]["source_dialog_id"]
            if d not in last or r["metadata"].get("turn_index", 0) > last[d]["metadata"].get("turn_index", 0):
                last[d] = r

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows, skipped = [], Counter()
    for dialog_id, r in last.items():
        dom = domain_of(dialog_id)
        meta = r["metadata"]
        if dom not in args.domains:
            continue
        if meta.get("correct") != 1:
            skipped[f"{dom}_incorrect"] += 1
            continue
        if dom == "telecom" and str(meta.get("task_id")) in held:
            skipped["telecom_heldout_task"] += 1
            continue
        msgs = []
        for m in r["messages"][1:] + [r["answer"]]:
            role = m["role"]
            if role == "assistant":
                msgs.append(clean_assistant(m))
            elif role == "tool":
                msgs.append({"role": "tool", "content": m.get("content") or "",
                             "tool_call_id": m.get("tool_call_id")})
            elif role == "user":
                if m.get("tool_calls"):  # telecom user device actions: never shown to the agent
                    continue
                msgs.append({"role": "user", "content": m.get("content") or ""})
        used = {c["function"]["name"] for m in msgs for c in m.get("tool_calls") or []}
        if used - tool_names[dom]:
            skipped[f"{dom}_unknown_tool"] += 1
            continue
        messages = [{"role": "system", "content": ref[dom]["system_prompt"]}]
        for m in msgs:
            if m["role"] == "assistant":
                m["_trainable"] = (m.get("content") or "").strip() != GREETING or bool(m.get("tool_calls"))
            messages.append(m)
        messages = native_chat.template_messages(messages)
        text = tok.apply_chat_template([{k: v for k, v in m.items() if k != "_trainable"} for m in messages],
                                       tools=ref[dom]["tools"], tokenize=False, enable_thinking=False)
        if len(tok.encode(text, add_special_tokens=False)) > args.max_tokens:
            skipped[f"{dom}_too_long"] += 1
            continue
        rows.append({"messages": json.dumps(messages, ensure_ascii=False),
                     "tools": json.dumps(ref[dom]["tools"], ensure_ascii=False),
                     "domain": dom, "task_id": str(meta.get("task_id") or dialog_id),
                     "error_kind": "areal", "verified_by": "areal"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output)
    print(json.dumps({"dialogs": len(last), "rows": len(rows), "by_domain": dict(Counter(r["domain"] for r in rows)),
                      "skipped": dict(skipped)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
