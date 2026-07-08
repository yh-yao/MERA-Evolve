"""Build verl-compatible parquet files for HumanEval and MBPP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def _make_prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    content = (
        "Complete the following Python function. Return only the complete code "
        "needed to solve the task.\n\n"
        f"{task['prompt']}"
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


def convert_rows(rows: list[dict[str, Any]], split: str, max_rows: int | None = None) -> list[dict[str, Any]]:
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
        out.append({
            "data_source": f"code/{dataset}",
            "prompt": _make_prompt(task),
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
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="eval")
    parser.add_argument("--max-train", type=int, default=-1)
    parser.add_argument("--max-val", type=int, default=-1)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    train = convert_rows(rows, args.train_split, args.max_train)
    val = convert_rows(rows, args.val_split, args.max_val)
    train_path = args.out_dir / "train.parquet"
    val_path = args.out_dir / "val.parquet"
    write_parquet(train, train_path)
    write_parquet(val, val_path)
    print(f"wrote {len(train)} rows -> {train_path}")
    print(f"wrote {len(val)} rows -> {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
