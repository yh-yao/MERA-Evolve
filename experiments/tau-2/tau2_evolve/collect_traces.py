"""Run a batch of tau2 tasks and write trace records to JSONL. Analogous to
verl_code_rl/collect_traces.py's probe-only fallback design: the main attempt
(agent=small model under test, user=3B local simulator) is what measures
actual performance. If it fails or omits required golden actions, a SEPARATE,
INDEPENDENT fallback attempt using the larger local model for both roles
re-runs the same task from
scratch to rescue a working trajectory for distillation -- not a
mid-conversation model handoff (which needs fragile message-history surgery,
see adapter.py's run_task_with_substitution), just two full, independent
rollouts, exactly like the code domain's small-then-large fallback. Both the
main user role and the fallback (agent+user) default to the same local user
endpoint -- no hosted user-simulator API calls.

Usage (tau2_stage2 venv):
  PYTHONPATH=experiments/tau-2 .../.venv_tau2/bin/python3 \
    -m tau2_evolve.collect_traces \
    --bucket TRAIN \
    --agent-base-url http://127.0.0.1:8200/v1 \
    --user-base-url http://127.0.0.1:8201/v1 \
    --out results/tau2_train_traces.jsonl [--skillbook path/to/skillbook.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau2_evolve import benchmark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="TRAIN", choices=["TRAIN", "EVAL", "DIAG", "VAL"])
    ap.add_argument("--domain", default=None, choices=benchmark.DOMAINS)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--agent-model", default="openai/evol-llm-agent")
    ap.add_argument("--agent-base-url", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--agent-api-key", default="EMPTY")
    ap.add_argument(
        "--agent-max-tokens", type=int, default=None,
        help="Maximum tokens per agent turn; useful for bounding verbose local models.",
    )
    ap.add_argument(
        "--agent-thinking", action=argparse.BooleanOptionalAction, default=None,
        help="Enable or disable Qwen3 thinking through the server chat template.",
    )
    ap.add_argument("--user-model", default="openai/evol-llm-user")
    ap.add_argument("--user-base-url", default="http://127.0.0.1:8201/v1")
    ap.add_argument("--user-api-key", default="EMPTY")
    ap.add_argument("--user-max-tokens", type=int, default=None)
    ap.add_argument(
        "--user-thinking", action=argparse.BooleanOptionalAction, default=None,
        help="Enable or disable Qwen3 thinking for the user simulator.",
    )
    ap.add_argument("--probe-only", action="store_true",
                     help="If the main attempt fails or omits required golden actions, retry "
                          "with the fallback agent+user -- a separate full rollout, not a "
                          "mid-conversation handoff.")
    ap.add_argument("--fallback-agent-model", default=None,
                     help="Defaults to --user-model/--user-base-url/--user-api-key (e.g. the "
                          "3B model already serving the user role).")
    ap.add_argument("--fallback-agent-base-url", default=None)
    ap.add_argument("--fallback-agent-api-key", default=None)
    ap.add_argument("--fallback-user-model", default=None)
    ap.add_argument("--fallback-user-base-url", default=None)
    ap.add_argument("--fallback-user-api-key", default=None)
    ap.add_argument(
        "--fallback-attempts", type=int, default=2,
        help="Independent fallback rollouts; stop early on an action-complete pass.",
    )
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-errors", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--resume", action="store_true",
        help="Append to an existing output and skip task IDs already present.",
    )
    ap.add_argument(
        "--tau2-log-level", default="CRITICAL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Loguru verbosity for tau2 internals; task summaries are always printed separately.",
    )
    ap.add_argument("--skillbook", type=Path, default=None,
                     help="JSON {domain: skill_text} to prepend to the agent's domain policy.")
    ap.add_argument("--router-model", type=Path)
    ap.add_argument("--router-teacher-model")
    ap.add_argument("--router-teacher-base-url")
    ap.add_argument("--router-teacher-api-key", default="EMPTY")
    ap.add_argument("--router-teacher-max-tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.fallback_attempts < 1:
        ap.error("--fallback-attempts must be at least 1")

    # Tau2 logs every message and tool payload at DEBUG and reports unknown
    # local-model pricing at ERROR. Under a large thread pool, serializing
    # those records becomes measurable work and obscures task-level progress.
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=args.tau2_log_level)

    tasks = benchmark.task_ids_for(args.bucket, args.domain)
    if args.limit >= 0:
        tasks = tasks[: args.limit]
    total_tasks = len(tasks)

    existing_rows: list[dict] = []
    if args.resume and args.out.exists():
        with args.out.open() as existing:
            existing_rows = [json.loads(line) for line in existing if line.strip()]
        failed_rows = [row for row in existing_rows if row.get("error")]
        if failed_rows:
            # Infrastructure/parser failures are not completed benchmark
            # outcomes. Remove them before append mode so resume replaces the
            # rows instead of creating duplicate task IDs.
            existing_rows = [row for row in existing_rows if not row.get("error")]
            with args.out.open("w") as cleaned:
                for row in existing_rows:
                    cleaned.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[collect] retrying {len(failed_rows)} errored rows", file=sys.stderr)
        completed = {(row["domain"], str(row["task_id"])) for row in existing_rows}
        tasks = [(domain, task_id) for domain, task_id in tasks if (domain, str(task_id)) not in completed]

    skillbook: dict[str, str] = {}
    if args.skillbook:
        skillbook = json.loads(args.skillbook.read_text())

    agent_spec = benchmark.make_llm_spec(
        args.agent_model,
        args.agent_base_url,
        args.agent_api_key,
        enable_thinking=args.agent_thinking,
        max_tokens=args.agent_max_tokens,
    )
    user_spec = benchmark.make_llm_spec(
        args.user_model,
        args.user_base_url,
        args.user_api_key,
        enable_thinking=args.user_thinking,
        max_tokens=args.user_max_tokens,
    )
    benchmark.prime_nl_judge_routing(user_spec)  # single-threaded, before the pool below

    fallback_agent_spec = fallback_user_spec = None
    if args.probe_only:
        fallback_agent_spec = benchmark.make_llm_spec(
            args.fallback_agent_model or args.user_model,
            args.fallback_agent_base_url or args.user_base_url,
            args.fallback_agent_api_key or args.user_api_key,
            enable_thinking=args.agent_thinking,
            max_tokens=args.agent_max_tokens,
        )
        fallback_user_spec = benchmark.make_llm_spec(
            args.fallback_user_model or args.user_model,
            args.fallback_user_base_url or args.user_base_url,
            args.fallback_user_api_key or args.user_api_key,
            enable_thinking=args.user_thinking,
            max_tokens=args.user_max_tokens,
        )
    router_teacher_spec = None
    if args.router_model:
        if not args.router_teacher_model or not args.router_teacher_base_url:
            ap.error("--router-model requires --router-teacher-model and --router-teacher-base-url")
        router_teacher_spec = benchmark.make_llm_spec(
            args.router_teacher_model,
            args.router_teacher_base_url,
            args.router_teacher_api_key,
            max_tokens=args.router_teacher_max_tokens,
        )

    print(
        f"[collect] bucket={args.bucket} n_tasks={total_tasks} remaining={len(tasks)} "
        f"skillbook={'yes' if skillbook else 'no'} probe_only={args.probe_only}",
        file=sys.stderr,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _collect_one(domain: str, task_id: str) -> dict:
        skill_text = skillbook.get(domain, "")
        main_row = benchmark.run_task(
            domain=domain, task_id=task_id, agent_spec=agent_spec, user_spec=user_spec,
            seed=args.seed, max_steps=args.max_steps, max_errors=args.max_errors,
            skill_text=skill_text,
            router_model_path=args.router_model,
            router_teacher_spec=router_teacher_spec,
        )
        main_row["fallback_used"] = False
        if (main_row["passed"] and benchmark.trace_action_complete(main_row)) or not args.probe_only:
            return main_row

        fallback_rows = []
        for attempt in range(1, args.fallback_attempts + 1):
            fallback_row = benchmark.run_task(
                domain=domain, task_id=task_id, agent_spec=fallback_agent_spec,
                user_spec=fallback_user_spec, seed=args.seed, max_steps=args.max_steps,
                max_errors=args.max_errors, skill_text=skill_text,
            )
            fallback_row["fallback_attempt"] = attempt
            fallback_rows.append(fallback_row)
            if fallback_row.get("passed") and benchmark.trace_action_complete(fallback_row):
                break
        fallback_row = max(
            fallback_rows,
            key=lambda row: (
                bool(row.get("passed") and benchmark.trace_action_complete(row)),
                bool(row.get("passed")),
                benchmark.trace_action_complete(row),
                float(row.get("reward", 0.0)),
                float(row.get("action_recall", 0.0)),
            ),
        )
        fallback_row["fallback_used"] = True
        fallback_row["fallback_attempts_run"] = len(fallback_rows)
        fallback_row["main_passed"] = bool(main_row["passed"])
        fallback_row["main_action_complete"] = bool(main_row.get("action_complete", False))
        fallback_row["main_termination_reason"] = main_row["termination_reason"]
        return fallback_row

    passed = sum(bool(row.get("passed")) for row in existing_rows)
    fallback_rescued = sum(
        bool(row.get("fallback_used"))
        and bool(row.get("passed"))
        and benchmark.trace_action_complete(row)
        for row in existing_rows
    )
    with args.out.open("a" if args.resume else "w") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_collect_one, domain, task_id): (domain, task_id) for domain, task_id in tasks}
            for fut in as_completed(futures):
                domain, task_id = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[collect] FAILED {domain}/{task_id}: {e}", file=sys.stderr)
                    row = {"domain": domain, "task_id": task_id, "passed": False, "reward": 0.0, "error": str(e), "fallback_used": False}
                passed += int(row.get("passed", False))
                fallback_rescued += int(
                    row.get("fallback_used", False)
                    and row.get("passed", False)
                    and benchmark.trace_action_complete(row)
                )
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                tag = (
                    f" [fallback {row.get('fallback_attempt')}/{row.get('fallback_attempts_run')}]"
                    if row.get("fallback_used") else ""
                )
                print(f"[collect] {domain}/{task_id}{tag} passed={row.get('passed')} reward={row.get('reward', 0.0):.2f}", file=sys.stderr)

    print(f"[collect] done passed={passed}/{total_tasks} (fallback_rescued={fallback_rescued})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
