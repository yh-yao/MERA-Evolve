"""Build a resumable deterministic cache for a fixed code teacher."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from verl_code_rl.collect_traces import (
    _call_with_repair,
    _check_model_registered,
    large_cache_key,
    load_tasks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/he_mbpp.jsonl"))
    parser.add_argument("--dataset", default="all", choices=["all", "humaneval", "mbpp"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument(
        "--empty-retry-max-tokens",
        type=int,
        default=8192,
        help="Retry once with this budget when the teacher returns no visible completion.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-repair-turns", type=int, default=0)
    args = parser.parse_args()

    tasks = load_tasks(args.data, "train", -1, args.dataset)
    tasks += load_tasks(args.data, "eval", -1, args.dataset)
    tasks = list({str(task["task_id"]): task for task in tasks}.values())
    existing: dict[str, dict] = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[row["cache_key"]] = row

    pending = [
        task for task in tasks
        if large_cache_key(
            task, args.model, args.temperature, args.max_tokens, args.max_repair_turns,
        ) not in existing
    ]
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=float(os.environ.get("LARGE_CACHE_REQUEST_TIMEOUT", "180")),
        max_retries=0,
    )
    _check_model_registered(client, args.model, "large", args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[large-cache] total={len(tasks)} existing={len(existing)} pending={len(pending)}",
        flush=True,
    )

    def collect(task: dict) -> dict:
        result = _call_with_repair(
            client=client,
            model=args.model,
            task=task,
            procedure="",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            repair_turns=args.max_repair_turns,
        )
        effective_max_tokens = args.max_tokens
        if not str(result.get("completion", "")).strip():
            effective_max_tokens = max(args.max_tokens, args.empty_retry_max_tokens)
            result = _call_with_repair(
                client=client,
                model=args.model,
                task=task,
                procedure="",
                temperature=args.temperature,
                max_tokens=effective_max_tokens,
                retries=args.retries,
                repair_turns=args.max_repair_turns,
            )
        if not str(result.get("completion", "")).strip():
            raise RuntimeError(
                f"teacher returned no completion for {task['task_id']}: {result.get('error', '')}"
            )
        result["cache_generation_max_tokens"] = effective_max_tokens
        return {
            "cache_key": large_cache_key(
                task, args.model, args.temperature, args.max_tokens, args.max_repair_turns,
            ),
            "task_id": task["task_id"],
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "repair_turns": args.max_repair_turns,
            "result": result,
        }

    mode = "a" if args.output.exists() else "w"
    failures_path = args.output.with_suffix(args.output.suffix + ".failures.jsonl")
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, args.output.open(mode) as out:
        futures = {pool.submit(collect, task): task for task in pending}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"task_id": str(task["task_id"]), "error": str(exc)})
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                print(f"[large-cache] completed={completed}/{len(pending)}", flush=True)
    if failures:
        failures_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"teacher cache has {len(failures)} failed tasks; details: {failures_path}"
        )
    failures_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
