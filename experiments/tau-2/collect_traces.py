"""Run a batch of tau2 tasks and write trace records to JSONL. Analogous to
verl_code_rl/collect_traces.py's probe-only fallback design: the main attempt
(agent=small model under test, user=3B local simulator) is what measures
actual performance. If it fails or omits required golden actions, a SEPARATE,
INDEPENDENT fallback attempt
(agent=3B, user=3B self-playing both sides) re-runs the same task from
scratch to rescue a working trajectory for distillation -- not a
mid-conversation model handoff (which needs fragile message-history surgery,
see adapter.py's run_task_with_substitution), just two full, independent
rollouts, exactly like the code domain's small-then-large fallback. Both the
main user role and the fallback (agent+user) default to the same local 3B
endpoint for now -- no hosted user-simulator API calls.

Usage (tau2_stage2 venv):
  .../.venv_tau2/bin/python3 experiments/tau-2/collect_traces.py \
    --bucket TRAIN \
    --agent-base-url http://127.0.0.1:8200/v1 \
    --user-base-url http://127.0.0.1:8201/v1 \
    --out results/tau2_train_traces.jsonl [--skillbook path/to/skillbook.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lib_tau2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="TRAIN", choices=["TRAIN", "EVAL", "DIAG", "VAL"])
    ap.add_argument("--domain", default=None, choices=lib_tau2.DOMAINS)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--agent-model", default="openai/evol-llm-agent")
    ap.add_argument("--agent-base-url", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--agent-api-key", default="EMPTY")
    ap.add_argument("--user-model", default="openai/evol-llm-user")
    ap.add_argument("--user-base-url", default="http://127.0.0.1:8201/v1")
    ap.add_argument("--user-api-key", default="EMPTY")
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
    ap.add_argument("--skillbook", type=Path, default=None,
                     help="JSON {domain: skill_text} to prepend to the agent's domain policy.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.fallback_attempts < 1:
        ap.error("--fallback-attempts must be at least 1")

    tasks = lib_tau2.task_ids_for(args.bucket, args.domain)
    if args.limit >= 0:
        tasks = tasks[: args.limit]

    skillbook: dict[str, str] = {}
    if args.skillbook:
        skillbook = json.loads(args.skillbook.read_text())

    agent_spec = lib_tau2.make_llm_spec(args.agent_model, args.agent_base_url, args.agent_api_key)
    user_spec = lib_tau2.make_llm_spec(args.user_model, args.user_base_url, args.user_api_key)
    lib_tau2.prime_nl_judge_routing(user_spec)  # single-threaded, before the pool below

    fallback_agent_spec = fallback_user_spec = None
    if args.probe_only:
        fallback_agent_spec = lib_tau2.make_llm_spec(
            args.fallback_agent_model or args.user_model,
            args.fallback_agent_base_url or args.user_base_url,
            args.fallback_agent_api_key
            or os.environ.get("COMMONSTACK_API_KEY")
            or args.user_api_key,
        )
        fallback_user_spec = lib_tau2.make_llm_spec(
            args.fallback_user_model or args.user_model,
            args.fallback_user_base_url or args.user_base_url,
            args.fallback_user_api_key or args.user_api_key,
        )

    print(
        f"[collect] bucket={args.bucket} n_tasks={len(tasks)} skillbook={'yes' if skillbook else 'no'} "
        f"probe_only={args.probe_only}",
        file=sys.stderr,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _collect_one(domain: str, task_id: str) -> dict:
        skill_text = skillbook.get(domain, "")
        main_row = lib_tau2.run_task(
            domain=domain, task_id=task_id, agent_spec=agent_spec, user_spec=user_spec,
            seed=args.seed, max_steps=args.max_steps, max_errors=args.max_errors,
            skill_text=skill_text,
        )
        main_row["fallback_used"] = False
        if (main_row["passed"] and main_row.get("action_complete", False)) or not args.probe_only:
            return main_row

        fallback_rows = []
        for attempt in range(1, args.fallback_attempts + 1):
            fallback_row = lib_tau2.run_task(
                domain=domain, task_id=task_id, agent_spec=fallback_agent_spec,
                user_spec=fallback_user_spec, seed=args.seed, max_steps=args.max_steps,
                max_errors=args.max_errors, skill_text=skill_text,
            )
            fallback_row["fallback_attempt"] = attempt
            fallback_rows.append(fallback_row)
            if fallback_row.get("passed") and fallback_row.get("action_complete", False):
                break
        fallback_row = max(
            fallback_rows,
            key=lambda row: (
                bool(row.get("passed") and row.get("action_complete", False)),
                bool(row.get("passed")),
                bool(row.get("action_complete", False)),
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

    passed = 0
    fallback_rescued = 0
    with args.out.open("w") as out:
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
                    and row.get("action_complete", False)
                )
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                tag = (
                    f" [fallback {row.get('fallback_attempt')}/{row.get('fallback_attempts_run')}]"
                    if row.get("fallback_used") else ""
                )
                print(f"[collect] {domain}/{task_id}{tag} passed={row.get('passed')} reward={row.get('reward', 0.0):.2f}", file=sys.stderr)

    print(f"[collect] done passed={passed}/{len(tasks)} (fallback_rescued={fallback_rescued})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
