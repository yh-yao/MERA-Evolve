"""Collect small/large execution traces for the MERA evolve loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from verl_code_rl.code_eval import extract_code, run_code_tests
from verl_code_rl.prepare_data import SYSTEM, _dataset_name, load_jsonl
from verl_code_rl.skills import SkillBook, extract_signature


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if any("split" in row for row in rows):
        return [row for row in rows if row.get("split") == split]
    if split == "train":
        return rows[0::2]
    if split in {"val", "eval", "test"}:
        return rows[1::2]
    raise ValueError(f"unknown split: {split}")


def load_tasks(path: Path, split: str, limit: int, dataset: str) -> list[dict[str, Any]]:
    rows = _split_rows(load_jsonl(path), split)
    if dataset != "all":
        rows = [row for row in rows if _dataset_name(row["task_id"], row) == dataset]
    if limit >= 0:
        rows = rows[:limit]
    return rows


def _messages(prompt: str, procedure: str = "") -> list[dict[str, str]]:
    content = (
        "Complete the following Python function. Return only the complete code "
        "needed to solve the task.\n\n"
    )
    problem = f"{procedure}\n\n---\n\n{prompt}" if procedure else prompt
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content + problem},
    ]


def _call_model(
    *,
    client: OpenAI,
    model: str,
    task: dict[str, Any],
    procedure: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    started = time.time()
    last_error = ""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=_messages(task["prompt"], procedure),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            completion = resp.choices[0].message.content or ""
            code = extract_code(completion, task["entry_point"], task["prompt"])
            ok, error = run_code_tests(task, code, timeout=int(task.get("timeout", 10)))
            return {
                "success": ok,
                "error": error,
                "completion": completion,
                "code": code,
                "latency": time.time() - started,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    return {
        "success": False,
        "error": last_error,
        "completion": "",
        "code": "",
        "latency": time.time() - started,
    }


def _mock_result(task: dict[str, Any], model_role: str, cycle: int) -> dict[str, Any]:
    digest = int(hashlib.md5(f"{task['task_id']}|{model_role}|{cycle}".encode()).hexdigest(), 16)
    threshold = 7 if model_role == "small" else 9
    ok = digest % 10 < threshold
    code = (
        f"def {task['entry_point']}(*args, **kwargs):\n"
        f"    # mock {model_role} solution for {task['task_id']}\n"
        "    return None\n"
    )
    return {
        "success": ok,
        "error": "" if ok else "mock failure",
        "completion": code,
        "code": code,
        "latency": 0.0,
    }


def _load_router(path: str | None):
    if not path:
        return None
    router_path = Path(path)
    if router_path.is_dir():
        router_path = router_path / "router.joblib"
    if not router_path.exists():
        print(f"[collect] router not found: {router_path}", file=sys.stderr)
        return None
    import joblib

    return joblib.load(router_path)


def _router_route(router, prompt: str, threshold: float) -> tuple[str, float | None]:
    if router is None:
        return "small", None
    try:
        proba = router.predict_proba([prompt])[0]
        prob_large = float(proba[1])
        return ("large" if prob_large >= threshold else "small"), prob_large
    except Exception:  # noqa: BLE001
        pred = int(router.predict([prompt])[0])
        return ("large" if pred else "small"), None


def _collect_one(
    task: dict[str, Any],
    *,
    cycle: int,
    small_model: str,
    large_model: str,
    small_client: OpenAI,
    large_client: OpenAI,
    skillbook: SkillBook | None,
    router,
    router_threshold: float,
    force_both: bool,
    mock: bool,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    prompt = task["prompt"]
    procedure = skillbook.get_procedure(prompt) if skillbook else ""
    route, route_prob = _router_route(router, prompt, router_threshold)

    if mock:
        small = _mock_result(task, "small", cycle)
    else:
        small = _call_model(
            client=small_client,
            model=small_model,
            task=task,
            procedure=procedure,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )

    should_run_large = force_both or (not small["success"]) or route == "large"
    large_skipped = not should_run_large
    if should_run_large:
        if mock:
            large = _mock_result(task, "large", cycle)
        else:
            large = _call_model(
                client=large_client,
                model=large_model,
                task=task,
                procedure=procedure if os.environ.get("LARGE_USE_SKILLS", "1") != "0" else "",
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
            )
    else:
        large = {"success": False, "error": "", "completion": "", "code": "", "latency": 0.0}

    if small["success"]:
        final_model = small_model
        final_success = True
        decision = "oracle:small_OK+large_run" if should_run_large else "probe:small->small_OK"
    else:
        final_model = large_model
        final_success = bool(large["success"])
        decision = f"probe:small_fail->large_{'OK' if large['success'] else 'fail'}"

    return {
        "task_id": task["task_id"],
        "dataset": _dataset_name(task["task_id"], task),
        "signature": extract_signature(prompt),
        "prompt": prompt,
        "entry_point": task["entry_point"],
        "test": task["test"],
        "round": cycle,
        "decision": decision,
        "small_model": small_model,
        "large_model": large_model,
        "small_success": bool(small["success"]),
        "large_success": bool(large["success"]),
        "large_skipped": large_skipped,
        "final_model": final_model,
        "final_success": final_success,
        "small_completion": small["completion"],
        "large_completion": large["completion"],
        "small_code": small["code"],
        "large_code": large["code"],
        "small_error": small["error"],
        "large_error": large["error"],
        "small_latency": small["latency"],
        "large_latency": large["latency"],
        "policy_route": route,
        "policy_router_prob": route_prob,
        "has_procedure": bool(procedure),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/he_mbpp.jsonl"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset", default="all", choices=["all", "humaneval", "mbpp"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument("--small-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--large-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--router", default="")
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--skillbook", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip the large model when small already passes and router chooses small.")
    args = parser.parse_args()

    tasks = load_tasks(args.data, args.split, args.limit, args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mock = os.environ.get("SCALING_MOCK", "0") == "1"

    skillbook = None
    if args.skillbook:
        skillbook = SkillBook()
        skillbook.load(args.skillbook)

    router = _load_router(args.router)
    small_client = OpenAI(api_key=args.api_key, base_url=args.small_base_url)
    large_client = OpenAI(api_key=args.api_key, base_url=args.large_base_url)
    force_both = not args.probe_only

    print(
        f"[collect] tasks={len(tasks)} split={args.split} force_both={force_both} "
        f"mock={mock} out={args.out}",
        file=sys.stderr,
    )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                _collect_one,
                task,
                cycle=args.cycle,
                small_model=args.small_model,
                large_model=args.large_model,
                small_client=small_client,
                large_client=large_client,
                skillbook=skillbook,
                router=router,
                router_threshold=args.router_threshold,
                force_both=force_both,
                mock=mock,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
            )
            for task in tasks
        ]
        with args.out.open("w") as out:
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

    small_ok = sum(1 for row in rows if row["small_success"])
    large_ok = sum(1 for row in rows if row["large_success"])
    final_ok = sum(1 for row in rows if row["final_success"])
    print(
        f"[collect] done small={small_ok}/{len(rows)} large={large_ok}/{len(rows)} "
        f"final={final_ok}/{len(rows)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
