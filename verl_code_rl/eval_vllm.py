"""Evaluate an OpenAI-compatible vLLM server on HumanEval/MBPP JSONL."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from verl_code_rl.code_eval import extract_code, run_code_tests
from verl_code_rl.prepare_data import SYSTEM, _dataset_name, load_jsonl
from verl_code_rl.skills import SkillBook


def _call_one(
    client: OpenAI,
    model: str,
    task: dict,
    temperature: float,
    max_tokens: int,
    skillbook: SkillBook | None,
) -> dict:
    started = time.time()
    procedure = skillbook.get_procedure(task["prompt"]) if skillbook else ""
    problem = f"{procedure}\n\n---\n\n{task['prompt']}" if procedure else task["prompt"]
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Complete the following Python function. Return only the complete code "
                "needed to solve the task.\n\n"
                f"{problem}"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        code = extract_code(text, task["entry_point"], task["prompt"])
        ok, error = run_code_tests(task, code)
        return {
            "task_id": task["task_id"],
            "dataset": _dataset_name(task["task_id"], task),
            "passed": ok,
            "error": error,
            "latency": time.time() - started,
            "completion": text,
            "code": code,
            "has_procedure": bool(procedure),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "task_id": task["task_id"],
            "dataset": _dataset_name(task["task_id"], task),
            "passed": False,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "latency": time.time() - started,
            "completion": "",
            "code": "",
            "has_procedure": bool(procedure),
        }


def _filter_tasks(rows: list[dict], split: str, dataset: str, limit: int) -> list[dict]:
    if any("split" in row for row in rows):
        rows = [row for row in rows if row.get("split") == split]
    elif split == "train":
        rows = rows[0::2]
    else:
        rows = rows[1::2]
    if dataset != "all":
        rows = [row for row in rows if _dataset_name(row["task_id"], row) == dataset]
    if limit >= 0:
        rows = rows[:limit]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/he_mbpp.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--dataset", default="all", choices=["all", "humaneval", "mbpp"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--skillbook", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/eval.jsonl"))
    args = parser.parse_args()

    tasks = _filter_tasks(load_jsonl(args.data), args.split, args.dataset, args.limit)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    skillbook = None
    if args.skillbook:
        skillbook = SkillBook()
        skillbook.load(args.skillbook)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    passed = 0
    by_dataset: dict[str, list[bool]] = {}
    with args.out.open("w") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_call_one, client, args.model, task, args.temperature, args.max_tokens, skillbook)
                for task in tasks
            ]
            for fut in as_completed(futures):
                row = fut.result()
                passed += int(row["passed"])
                by_dataset.setdefault(row["dataset"], []).append(bool(row["passed"]))
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

    total = len(tasks)
    print(f"pass@1 {passed}/{total} = {passed / total if total else 0:.4f}")
    for name, vals in sorted(by_dataset.items()):
        print(f"{name}: {sum(vals)}/{len(vals)} = {sum(vals) / len(vals):.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
