"""AReaL-tau2-data SFT turns -> whole-dialog token windows for train_lora.py.

Same rules as tau2_evolve.areal_to_sft (jdc): keep correct==1 dialogs, drop
telecom dialogs whose official task is in our VAL/EVAL split, replace the
release's think-first system prompt with tau2's own (captured by
capture_context.py), drop thinking, drop user device actions, render with the
Qwen3.5 chat template (thinking disabled) and train only on assistant turns
(tau2's fixed greeting excluded).

Output: one JSON line per dialog {"input_ids": [...], "labels": [...], "domain", "dialog"}.
"""
import argparse, json, re
from collections import Counter

from transformers import AutoTokenizer

GREETING = "Hi! How can I help you today?"
THINK = re.compile(r"<think>.*?</think>\s*", re.S)

ap = argparse.ArgumentParser()
ap.add_argument("--sft", required=True)
ap.add_argument("--context", required=True)
ap.add_argument("--heldout-telecom", required=True, help="JSON list of telecom task ids in VAL/EVAL")
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=32768)
args = ap.parse_args()

ctx = json.load(open(args.context))
held = set(json.load(open(args.heldout_telecom)))
names = {d: {t["function"]["name"] for t in v["tools"]} for d, v in ctx.items()}
tok = AutoTokenizer.from_pretrained(args.tokenizer)
a_start = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
im_end = tok.convert_tokens_to_ids("<|im_end|>")

last = {}
for line in open(args.sft):
    r = json.loads(line)
    d = r["metadata"]["source_dialog_id"]
    if d not in last or r["metadata"].get("turn_index", 0) > last[d]["metadata"].get("turn_index", 0):
        last[d] = r


def call(c, i):
    fn = c.get("function") or {"name": c.get("name"), "arguments": c.get("arguments")}
    a = fn.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            pass
    return {"type": "function", "id": c.get("id") or f"call_{i}", "function": {"name": fn["name"], "arguments": a}}


skipped, n_out, by_dom, tok_total = Counter(), 0, Counter(), 0
with open(args.out, "w") as out:
    for dialog, r in last.items():
        head = dialog.split("_")[0]
        dom = head if head in ("airline", "retail") else "telecom"
        meta = r["metadata"]
        if meta.get("correct") != 1:
            skipped[f"{dom}_incorrect"] += 1; continue
        if dom == "telecom" and str(meta.get("task_id")) in held:
            skipped["telecom_heldout"] += 1; continue
        msgs, trainable = [{"role": "system", "content": ctx[dom]["system_prompt"]}], []
        bad = False
        for m in r["messages"][1:] + [r["answer"]]:
            role = m["role"]
            if role == "assistant":
                calls = [call(c, i) for i, c in enumerate(m.get("tool_calls") or [])]
                if any(c["function"]["name"] not in names[dom] for c in calls):
                    bad = True
                content = THINK.sub("", m.get("content") or "").strip()
                am = {"role": "assistant", "content": content}
                if calls:
                    am["tool_calls"] = calls
                msgs.append(am)
                trainable.append(not (content == GREETING and not calls))
            elif role == "tool":
                msgs.append({"role": "tool", "content": m.get("content") or ""})
            elif role == "user" and not m.get("tool_calls"):
                msgs.append({"role": "user", "content": m.get("content") or ""})
        if bad:
            skipped[f"{dom}_unknown_tool"] += 1; continue
        ids = tok.apply_chat_template(msgs, tools=ctx[dom]["tools"], tokenize=True, add_generation_prompt=False,
                                      enable_thinking=False, return_dict=False)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if len(ids) > args.max_tokens:
            skipped[f"{dom}_too_long"] += 1; continue
        labels, k, active, i = [-100] * len(ids), 0, False, 0
        while i < len(ids):
            if ids[i:i + len(a_start)] == a_start:
                active = trainable[k] if k < len(trainable) else False
                k += 1; i += len(a_start); continue
            if active:
                labels[i] = ids[i]
                if ids[i] == im_end:
                    active = False
            i += 1
        if k != len(trainable):
            skipped[f"{dom}_turn_mismatch"] += 1; continue
        out.write(json.dumps({"input_ids": ids, "labels": labels, "domain": dom, "dialog": dialog}) + "\n")
        n_out += 1; by_dom[dom] += 1; tok_total += len(ids)
print(json.dumps({"dialogs": len(last), "written": n_out, "by_domain": dict(by_dom), "tokens": tok_total,
                  "skipped": dict(skipped)}))
