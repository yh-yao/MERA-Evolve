"""Extract teacher and self-repair SFT pairs from MERA oracle traces.

This module deliberately only prepares an interoperable JSONL artifact.  The
choice of SFT launcher is environment-specific (FSDP/LoRA/verl version), while
the subsequent RL phase remains the existing verl GRPO invocation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verl_code_rl.prepare_data import _make_prompt
from verl_code_rl.skills import SkillBook


def _load_skillbook(path: Path | None) -> SkillBook | None:
    if path is None:
        return None
    book = SkillBook()
    book.load(path)
    return book


def _task(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "prompt": row["prompt"],
        "entry_point": row["entry_point"],
        "test": row["test"],
    }


def extract_pairs(rows: list[dict[str, Any]], skillbook: SkillBook | None = None) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("prompt"):
            continue
        task = _task(row)
        procedure = skillbook.get_procedure(task["prompt"]) if skillbook else ""
        prompt = _make_prompt(task, procedure=procedure)
        # Teacher distillation: the student failed, teacher passed.
        if row.get("small_success") is False and row.get("large_success") is True:
            completion = str(row.get("large_code") or row.get("large_completion") or "").strip()
            if completion:
                pairs.append({
                    "messages": prompt,
                    "completion": completion,
                    "source": "teacher",
                    "task_id": row.get("task_id"),
                    "has_procedure": bool(procedure),
                })
        # A later successful small repair is a high-value on-policy target.
        turns = row.get("small_turns") or []
        if len(turns) >= 2 and not turns[0].get("success") and turns[-1].get("success"):
            completion = str(turns[-1].get("code") or turns[-1].get("completion") or "").strip()
            if completion:
                pairs.append({
                    "messages": prompt + [
                        {"role": "assistant", "content": str(turns[0].get("completion") or "")},
                        {"role": "user", "content": "Repair the failing solution and return only complete Python code."},
                    ],
                    "completion": completion,
                    "source": "self_repair",
                    "task_id": row.get("task_id"),
                    "has_procedure": bool(procedure),
                })
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skillbook", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    with args.traces.open() as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    pairs = extract_pairs(rows, _load_skillbook(args.skillbook))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    counts = {kind: sum(p["source"] == kind for p in pairs) for kind in ("teacher", "self_repair")}
    print(f"wrote {len(pairs)} SFT pairs -> {args.output}; {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
