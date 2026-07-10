"""Build verl-compatible parquet files for HumanEval and MBPP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verl_code_rl.skills import SkillBook


SYSTEM = (
    "You are a Python coding assistant. Return only valid Python code. "
    "Do not include Markdown fences, explanations, examples, or tests."
)


def _dataset_name(task_id: str, raw: dict[str, Any]) -> str:
    src = (raw.get("_src") or "").lower()
    if task_id.lower().startswith("humaneval") or src == "humaneval":
        return "humaneval"
    if task_id.lower().startswith("mbpp") or src == "mbpp":
        return "mbpp"
    return src or "code"


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if any("split" in row for row in rows):
        return [row for row in rows if row.get("split") == split]
    if split == "train":
        return rows[0::2]
    if split in {"val", "eval", "test"}:
        return rows[1::2]
    raise ValueError(f"unknown split: {split}")


def _make_prompt(task: dict[str, Any], procedure: str = "") -> list[dict[str, str]]:
    problem = f"{procedure}\n\n---\n\n{task['prompt']}" if procedure else task["prompt"]
    content = (
        "Complete the following Python function. Return only the complete code "
        "needed to solve the task.\n\n"
        f"{problem}"
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_skillbook(path: Path | None) -> SkillBook | None:
    if not path:
        return None
    skillbook = SkillBook()
    skillbook.load(path)
    return skillbook


def _procedure_for(skillbook: SkillBook | None, prompt: str, dataset: str = "") -> str:
    return skillbook.get_procedure(prompt, dataset) if skillbook else ""


def _trace_to_raw_task(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "prompt": row["prompt"],
        "entry_point": row["entry_point"],
        "test": row["test"],
        "split": row.get("split"),
        "_src": row.get("dataset"),
    }


def convert_rows(
    rows: list[dict[str, Any]],
    split: str,
    max_rows: int | None = None,
    skillbook: SkillBook | None = None,
) -> list[dict[str, Any]]:
    selected = _split_rows(rows, split)
    if max_rows is not None and max_rows >= 0:
        selected = selected[:max_rows]

    out = []
    for idx, row in enumerate(selected):
        task_id = row["task_id"]
        dataset = _dataset_name(task_id, row)
        task = {
            "task_id": task_id,
            "dataset": dataset,
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
            "timeout": 10,
        }
        procedure = _procedure_for(skillbook, task["prompt"], dataset)
        out.append({
            "data_source": f"code/{dataset}",
            "prompt": _make_prompt(task, procedure=procedure),
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(task, ensure_ascii=False),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "task_id": task_id,
                "dataset": dataset,
                "has_procedure": bool(procedure),
            },
        })
    return out


def write_parquet(rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required: pip install pandas pyarrow") from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/he_mbpp.jsonl"))
    parser.add_argument("--traces", type=Path, default=None,
                        help="Optional trace JSONL source. Rows must include prompt, entry_point, and test.")
    parser.add_argument("--skillbook", type=Path, default=None,
                        help="Optional skillbook whose procedure is prepended to prompts.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="eval")
    parser.add_argument("--max-train", type=int, default=-1)
    parser.add_argument("--max-val", type=int, default=-1)
    args = parser.parse_args()

    rows = load_jsonl(args.traces) if args.traces else load_jsonl(args.input)
    if args.traces:
        rows = [_trace_to_raw_task(row) for row in rows]
        # Trace files normally represent a concrete split; tag them as train so
        # _split_rows keeps the whole file for train parquet generation.
        for row in rows:
            row["split"] = row.get("split") or "train"
    skillbook = _load_skillbook(args.skillbook)
    train = convert_rows(rows, args.train_split, args.max_train, skillbook=skillbook)
    val_source = load_jsonl(args.input) if args.traces else rows
    val = convert_rows(val_source, args.val_split, args.max_val, skillbook=skillbook)
    train_path = args.out_dir / "train.parquet"
    val_path = args.out_dir / "val.parquet"
    write_parquet(train, train_path)
    write_parquet(val, val_path)
    print(f"wrote {len(train)} rows -> {train_path}")
    print(f"wrote {len(val)} rows -> {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
