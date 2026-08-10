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


def shard_tasks(tasks: list[dict[str, Any]], modulo: int, remainder: int) -> list[dict[str, Any]]:
    """Select a deterministic task-id shard without relying on input ordering."""
    if modulo <= 1:
        return tasks
    if not 0 <= remainder < modulo:
        raise ValueError("task-shard remainder must be in [0, modulo)")
    return [
        task for task in tasks
        if int(hashlib.sha256(str(task["task_id"]).encode()).hexdigest(), 16) % modulo == remainder
    ]


def shard_tasks_many(
    tasks: list[dict[str, Any]], modulo: int, remainders: list[int],
) -> list[dict[str, Any]]:
    """Select multiple deterministic shards while preserving input order."""
    if modulo <= 1:
        return tasks
    selected = set(remainders)
    if not selected or any(not 0 <= remainder < modulo for remainder in selected):
        raise ValueError("task-shard remainders must be non-empty and in [0, modulo)")
    return [
        task for task in tasks
        if int(hashlib.sha256(str(task["task_id"]).encode()).hexdigest(), 16) % modulo in selected
    ]


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
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    started = time.time()
    last_error = ""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages or _messages(task["prompt"], procedure),
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
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
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
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def _mock_result(task: dict[str, Any], model_role: str, cycle: int, sample: int = 0) -> dict[str, Any]:
    digest = int(hashlib.md5(f"{task['task_id']}|{model_role}|{cycle}|{sample}".encode()).hexdigest(), 16)
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
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def _call_with_repair(
    *,
    client: OpenAI,
    model: str,
    task: dict[str, Any],
    procedure: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    repair_turns: int,
) -> dict[str, Any]:
    """Generate code and, optionally, let the same model repair failing code.

    The final result remains the serving outcome.  ``turns`` preserves the
    compact trajectory required for self-repair SFT extraction.
    """
    messages = _messages(task["prompt"], procedure)
    turns: list[dict[str, Any]] = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    result: dict[str, Any] | None = None
    for turn in range(max(0, repair_turns) + 1):
        result = _call_model(
            client=client, model=model, task=task, procedure=procedure,
            temperature=temperature, max_tokens=max_tokens, retries=retries,
            messages=messages,
        )
        total_latency += float(result["latency"])
        total_prompt_tokens += int(result["prompt_tokens"])
        total_completion_tokens += int(result["completion_tokens"])
        turns.append({
            "turn": turn,
            "completion": result["completion"],
            "code": result["code"],
            "success": result["success"],
            "error": result["error"],
        })
        if result["success"] or turn >= repair_turns:
            break
        messages = messages + [
            {"role": "assistant", "content": result["completion"]},
            {
                "role": "user",
                "content": (
                    "The submitted code failed the tests with:\n"
                    f"{result['error']}\n\nRepair it. Return only the complete Python code."
                ),
            },
        ]
    assert result is not None
    result = dict(result)
    result.update({
        "latency": total_latency,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "turns": turns,
        "initial_success": bool(turns[0]["success"]),
    })
    return result


def _priced_cost(result: dict[str, Any], input_per_million: float, output_per_million: float) -> float:
    return (
        float(result.get("prompt_tokens", 0)) * input_per_million
        + float(result.get("completion_tokens", 0)) * output_per_million
    ) / 1_000_000


def _load_router(path: str | None):
    """Load the classifier plus the embedding model id it was trained against."""
    if not path:
        return None
    router_dir = Path(path)
    router_path = router_dir / "router.joblib" if router_dir.is_dir() else router_dir
    meta_path = (router_dir if router_dir.is_dir() else router_dir.parent) / "router_meta.json"
    if not router_path.exists():
        print(f"[collect] router not found: {router_path}", file=sys.stderr)
        return None
    import joblib

    from verl_code_rl.embedding import DEFAULT_EMBED_MODEL

    embed_model = DEFAULT_EMBED_MODEL
    if meta_path.exists():
        try:
            embed_model = json.loads(meta_path.read_text()).get("embedding_model", DEFAULT_EMBED_MODEL)
        except (json.JSONDecodeError, OSError):
            pass
    return joblib.load(router_path), embed_model


def _router_route(router, prompt: str, threshold: float) -> tuple[str, float | None]:
    if router is None:
        return "small", None
    clf, embed_model = router
    from verl_code_rl.embedding import embed

    features = embed([prompt], embed_model)
    try:
        proba = clf.predict_proba(features)[0]
        prob_large = float(proba[1])
        return ("large" if prob_large >= threshold else "small"), prob_large
    except Exception:  # noqa: BLE001
        pred = int(clf.predict(features)[0])
        return ("large" if pred else "small"), None


def _router_routes(router, prompts: list[str], threshold: float) -> list[tuple[str, float | None]]:
    """Route a collection with one batched embedding pass."""
    if router is None:
        return [("small", None)] * len(prompts)
    clf, embed_model = router
    from verl_code_rl.embedding import embed

    features = embed(prompts, embed_model)
    try:
        probabilities = clf.predict_proba(features)
        return [
            ("large" if float(row[1]) >= threshold else "small", float(row[1]))
            for row in probabilities
        ]
    except Exception:  # noqa: BLE001
        return [("large" if int(pred) else "small", None) for pred in clf.predict(features)]


def large_cache_key(
    task: dict[str, Any], model: str, temperature: float, max_tokens: int, repair_turns: int,
) -> str:
    payload = {
        "model": model,
        "task_id": str(task["task_id"]),
        "prompt": task["prompt"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repair_turns": repair_turns,
        "large_use_skills": False,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_large_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["cache_key"]] = row["result"]
    return cache


def _apply_policy(
    *,
    route: str,
    route_prob: float | None,
    small: dict[str, Any],
    large: dict[str, Any],
    large_skipped: bool,
    small_model: str,
    large_model: str,
    small_cost: float,
    large_cost: float,
) -> dict[str, Any]:
    """Compute deployment semantics separately from oracle collection.

    A route to ``small`` is a cascade: use small and fall back to large only on
    a failed execution.  A route to ``large`` bypasses the small result.  During
    oracle collection both results are normally available; probe-only rows keep
    unknown fields rather than fabricating a large outcome.
    """
    if route == "large":
        if large_skipped:
            return {
                "policy_final_model": large_model,
                "policy_final_success": None,
                "policy_total_cost": None,
                "policy_total_latency": None,
                "policy_decision": "policy:route_large(unknown:large_skipped)",
                "policy_used_fallback": False,
            }
        return {
            "policy_final_model": large_model,
            "policy_final_success": bool(large["success"]),
            "policy_total_cost": large_cost,
            "policy_total_latency": float(large["latency"]),
            "policy_decision": "policy:route_large",
            "policy_used_fallback": False,
        }
    if small["success"]:
        return {
            "policy_final_model": small_model,
            "policy_final_success": True,
            "policy_total_cost": small_cost,
            "policy_total_latency": float(small["latency"]),
            "policy_decision": "policy:small_ok",
            "policy_used_fallback": False,
        }
    if large_skipped:
        return {
            "policy_final_model": large_model,
            "policy_final_success": None,
            "policy_total_cost": None,
            "policy_total_latency": None,
            "policy_decision": "policy:small_fail->large(unknown:large_skipped)",
            "policy_used_fallback": True,
        }
    return {
        "policy_final_model": large_model,
        "policy_final_success": bool(large["success"]),
        "policy_total_cost": small_cost + large_cost,
        "policy_total_latency": float(small["latency"]) + float(large["latency"]),
        "policy_decision": "policy:small_fail->large",
        "policy_used_fallback": True,
    }


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
    route_override: tuple[str, float | None] | None,
    large_cache: dict[str, dict[str, Any]],
    large_temperature: float,
    large_max_tokens: int,
    force_both: bool,
    mock: bool,
    temperature: float,
    max_tokens: int,
    retries: int,
    small_samples: int,
    repair_turns: int,
    small_input_cost_per_million: float,
    small_output_cost_per_million: float,
    large_input_cost_per_million: float,
    large_output_cost_per_million: float,
) -> dict[str, Any]:
    prompt = task["prompt"]
    dataset = _dataset_name(task["task_id"], task)
    procedure = skillbook.get_procedure(prompt, dataset) if skillbook else ""
    route, route_prob = route_override or _router_route(router, prompt, router_threshold)

    small_rollouts: list[dict[str, Any]] = []
    for sample in range(max(1, small_samples)):
        if mock:
            sample_result = _mock_result(task, "small", cycle, sample=sample)
            sample_result["turns"] = [{"turn": 0, **sample_result}]
            sample_result["initial_success"] = sample_result["success"]
        else:
            sample_result = _call_with_repair(
                client=small_client, model=small_model, task=task, procedure=procedure,
                temperature=temperature, max_tokens=max_tokens, retries=retries,
                repair_turns=repair_turns,
            )
        small_rollouts.append(sample_result)
    # The first rollout is the actual small-first serving attempt.  The complete
    # sample set estimates P(small fails) for the router target.
    small = small_rollouts[0]

    cache_key = large_cache_key(
        task, large_model, large_temperature, large_max_tokens, repair_turns,
    )
    cached_large = large_cache.get(cache_key)
    should_run_large = force_both or (not small["success"]) or route == "large"
    large_skipped = not should_run_large and cached_large is None
    large_cache_hit = cached_large is not None
    if cached_large is not None:
        large = dict(cached_large)
    elif should_run_large:
        if mock:
            large = _mock_result(task, "large", cycle)
            large["turns"] = [{"turn": 0, **large}]
            large["initial_success"] = large["success"]
        else:
            large = _call_with_repair(
                client=large_client,
                model=large_model,
                task=task,
                procedure=procedure if os.environ.get("LARGE_USE_SKILLS", "1") != "0" else "",
                temperature=large_temperature,
                max_tokens=large_max_tokens,
                retries=retries,
                repair_turns=repair_turns,
            )
    else:
        large = {
            "success": False, "error": "", "completion": "", "code": "", "latency": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0, "turns": [], "initial_success": False,
        }

    small_cost = _priced_cost(small, small_input_cost_per_million, small_output_cost_per_million)
    large_cost = _priced_cost(large, large_input_cost_per_million, large_output_cost_per_million)
    policy = _apply_policy(
        route=route, route_prob=route_prob, small=small, large=large, large_skipped=large_skipped,
        small_model=small_model, large_model=large_model, small_cost=small_cost, large_cost=large_cost,
    )

    if small["success"]:
        final_model = small_model
        final_success = True
        decision = "oracle:small_OK+large_run" if should_run_large else "probe:small->small_OK"
    else:
        final_model = large_model
        final_success = bool(large["success"])
        decision = f"probe:small_fail->large_{'OK' if large['success'] else 'fail'}"

    small_pass_count = sum(bool(item["success"]) for item in small_rollouts)
    return {
        "task_id": task["task_id"],
        "dataset": dataset,
        "signature": extract_signature(prompt, dataset),
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
        "large_cache_hit": large_cache_hit,
        "oracle_final_model": final_model,
        "oracle_final_success": final_success,
        "final_model": policy["policy_final_model"],
        "final_success": policy["policy_final_success"],
        "small_completion": small["completion"],
        "large_completion": large["completion"],
        "small_code": small["code"],
        "large_code": large["code"],
        "small_error": small["error"],
        "large_error": large["error"],
        "small_latency": small["latency"],
        "large_latency": large["latency"],
        "small_prompt_tokens": small["prompt_tokens"],
        "small_completion_tokens": small["completion_tokens"],
        "large_prompt_tokens": large["prompt_tokens"],
        "large_completion_tokens": large["completion_tokens"],
        "small_cost": small_cost,
        "large_cost": large_cost,
        "small_samples": len(small_rollouts),
        "small_pass_count": small_pass_count,
        "small_pass_rate": small_pass_count / len(small_rollouts),
        "small_initial_success": bool(small.get("initial_success", small["success"])),
        "small_turns": small.get("turns", []),
        "large_turns": large.get("turns", []),
        "policy_route": route,
        "policy_router_prob": route_prob,
        "has_procedure": bool(procedure),
        **policy,
    }


def _check_model_registered(client: OpenAI, model: str, role: str, base_url: str) -> None:
    """Fail fast and loud instead of quietly recording hundreds of identical
    404s as 'model failures' -- this exact failure mode (a reload leaving the
    wrong model/adapter registered, or a stale/untracked server still bound
    to the port) has silently corrupted full collection runs before."""
    try:
        registered = {m.id for m in client.models.list().data}
    except Exception as exc:  # noqa: BLE001
        print(
            f"[collect] FATAL: could not reach {role} server at {base_url}: "
            f"{type(exc).__name__}: {str(exc)[:200]}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    if model not in registered:
        print(
            f"[collect] FATAL: {role} model '{model}' is not registered at {base_url} "
            f"(it currently serves: {sorted(registered)}). Every request would fail "
            "identically instead of reflecting real model behavior -- refusing to "
            "proceed. Check that the reload/serve step actually completed for this model.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/he_mbpp.jsonl"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset", default="all", choices=["all", "humaneval", "mbpp"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--task-offset", type=int, default=0,
        help="Skip this many tasks after deterministic sharding and before --limit.",
    )
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument("--small-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--large-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument(
        "--api-key", default=os.environ.get("API_KEY", "EMPTY"),
        help="OpenAI-compatible API key; defaults to API_KEY from the environment.",
    )
    parser.add_argument("--router", default="")
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--skillbook", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--large-temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--large-max-tokens", type=int, default=768)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--small-samples", type=int, default=1,
                        help="Independent small rollouts per task for P(small fail) estimation.")
    parser.add_argument("--max-repair-turns", type=int, default=0,
                        help="Extra test-feedback repair attempts per model rollout.")
    parser.add_argument("--task-modulo", type=int, default=1)
    parser.add_argument("--task-remainder", type=int, default=0)
    parser.add_argument(
        "--task-remainders", default="",
        help="Comma-separated shard remainders; overrides --task-remainder.",
    )
    parser.add_argument("--small-input-cost-per-million", type=float,
                        default=float(os.environ.get("SMALL_INPUT_COST_PER_MILLION", "0")))
    parser.add_argument("--small-output-cost-per-million", type=float,
                        default=float(os.environ.get("SMALL_OUTPUT_COST_PER_MILLION", "0")))
    parser.add_argument("--large-input-cost-per-million", type=float,
                        default=float(os.environ.get("LARGE_INPUT_COST_PER_MILLION", "0")))
    parser.add_argument("--large-output-cost-per-million", type=float,
                        default=float(os.environ.get("LARGE_OUTPUT_COST_PER_MILLION", "0")))
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip the large model when small already passes and router chooses small.")
    parser.add_argument("--large-cache", type=Path,
                        help="Deterministic large-model result cache built by build_large_cache.")
    parser.add_argument("--require-large-cache", action="store_true",
                        help="Fail before collection if any selected task is absent from --large-cache.")
    args = parser.parse_args()

    # Apply a deterministic shard before the optional limit.  Otherwise a small
    # smoke limit can accidentally contain no examples for a router shard.
    all_tasks = load_tasks(args.data, args.split, -1, args.dataset)
    if args.task_remainders:
        remainders = [int(value) for value in args.task_remainders.split(",") if value.strip()]
        tasks = shard_tasks_many(all_tasks, args.task_modulo, remainders)
    else:
        tasks = shard_tasks(all_tasks, args.task_modulo, args.task_remainder)
    tasks = tasks[max(0, args.task_offset):]
    if args.limit >= 0:
        tasks = tasks[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mock = os.environ.get("SCALING_MOCK", "0") == "1"

    skillbook = None
    if args.skillbook:
        skillbook = SkillBook()
        skillbook.load(args.skillbook)

    router = _load_router(args.router)
    routes = _router_routes(router, [task["prompt"] for task in tasks], args.router_threshold)
    large_cache = load_large_cache(args.large_cache)
    if args.require_large_cache:
        missing = [
            task["task_id"] for task in tasks
            if large_cache_key(
                task, args.large_model, args.large_temperature, args.large_max_tokens,
                args.max_repair_turns,
            ) not in large_cache
        ]
        if missing:
            parser.error(
                f"large cache is missing {len(missing)} selected tasks; first={missing[0]}"
            )
    small_client = OpenAI(api_key=args.api_key, base_url=args.small_base_url)
    large_client = OpenAI(api_key=args.api_key, base_url=args.large_base_url)
    force_both = not args.probe_only

    if not mock:
        _check_model_registered(small_client, args.small_model, "small", args.small_base_url)
        if not args.require_large_cache:
            _check_model_registered(large_client, args.large_model, "large", args.large_base_url)

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
                route_override=route_override,
                large_cache=large_cache,
                large_temperature=args.large_temperature,
                large_max_tokens=args.large_max_tokens,
                force_both=force_both,
                mock=mock,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
                small_samples=args.small_samples,
                repair_turns=args.max_repair_turns,
                small_input_cost_per_million=args.small_input_cost_per_million,
                small_output_cost_per_million=args.small_output_cost_per_million,
                large_input_cost_per_million=args.large_input_cost_per_million,
                large_output_cost_per_million=args.large_output_cost_per_million,
            )
            for task, route_override in zip(tasks, routes)
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
