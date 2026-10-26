"""Turn verified error points into single-target SFT rows (native protocol).

Each row is the student's own history up to its first mistake (context,
``_trainable: false`` on its assistant turns) followed by the teacher's
corrected turn (the only trainable turn). Messages are tau2's OpenAI format
(``native_chat``), and tools are passed to the chat template, so the
rendered prompt is the one the evaluated model sees.

Messages and tools are stored as JSON strings: Parquet would otherwise
union the per-domain tool schemas into one struct and pad missing keys with
nulls, which the chat template would render into the tool block.
Load with ``tau2_evolve.sft_dataset.Qwen35NativeSFTDataset``.

Usage (tau2 venv):
  python -m tau2_evolve.error_points_to_sft --error-points ep.jsonl --output corrections.parquet \\
      [--verified-by student teacher]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tau2_evolve import native_chat


def sft_row(rec: dict, teacher_message: dict | None = None) -> dict:
    context = native_chat.litellm_messages(rec["context_messages"], rec["system_prompt"])
    target = native_chat.litellm_messages([teacher_message or rec["teacher_message"]])[0]
    messages = [{**m, "_trainable": False} if m["role"] == "assistant" else m for m in context]
    messages.append({**target, "_trainable": True})
    return {
        "messages": json.dumps(native_chat.template_messages(messages), ensure_ascii=False),
        "tools": json.dumps(rec["tools"], ensure_ascii=False),
        "domain": rec["domain"],
        "task_id": str(rec["task_id"]),
        "error_kind": rec["kind"],
        "verified_by": rec["verified_by"],
    }


def rehearsal_row(trace: dict) -> dict:
    """A passed student trajectory with every assistant turn trainable.

    Mixed in with the corrections so the update keeps the student's own
    successful behaviour (single-turn corrections alone collapsed it into
    repeated tool calls at lr 1e-5).
    """
    messages = [{**m, "_trainable": True} if m["role"] == "assistant" else m
                for m in native_chat.litellm_messages(trace["messages"], trace["system_prompt"])]
    return {
        "messages": json.dumps(native_chat.template_messages(messages), ensure_ascii=False),
        "tools": json.dumps(trace["tools"], ensure_ascii=False),
        "domain": trace["domain"],
        "task_id": str(trace["task_id"]),
        "error_kind": "rehearsal",
        "verified_by": "rehearsal",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-points", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verified-by", nargs="+", default=["student", "teacher"],
                    help="which verification outcomes to keep")
    ap.add_argument("--no-variants", action="store_true", help="ignore sampled teacher_variants")
    ap.add_argument("--rehearsal-traces", type=Path, action="append", default=[],
                    help="student traces; passed ones become whole-trajectory rows")
    ap.add_argument("--rehearsal-per-task", type=int, default=1)
    args = ap.parse_args()
    import pandas as pd

    rows, skipped = [], Counter()
    seen = set()
    for path in args.error_points:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("turn_index") is None:
                skipped["no_divergence"] += 1
                continue
            if rec.get("verified_by") not in args.verified_by:
                skipped[f"unverified_{rec.get('verified_by')}"] += 1
                continue
            key = (rec["domain"], str(rec["task_id"]), rec["turn_index"], json.dumps(rec["teacher_message"].get("tool_calls"), sort_keys=True))
            if key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(key)
            rows.append(sft_row(rec))
            if not args.no_variants:
                rows.extend(sft_row(rec, v) for v in rec.get("teacher_variants") or [])
    per_task = Counter()
    for path in args.rehearsal_traces:
        for line in path.read_text().splitlines():
            trace = json.loads(line)
            if trace.get("error") or not trace.get("passed") or not trace.get("messages"):
                continue
            key = (trace["domain"], str(trace["task_id"]))
            if per_task[key] >= args.rehearsal_per_task:
                continue
            per_task[key] += 1
            rows.append(rehearsal_row(trace))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output)
    print(json.dumps({"rows": len(rows), "skipped": dict(skipped),
                      "by_kind": dict(Counter(r["error_kind"] for r in rows)),
                      "by_domain": dict(Counter(r["domain"] for r in rows)),
                      "by_verifier": dict(Counter(r["verified_by"] for r in rows))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
