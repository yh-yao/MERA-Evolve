"""Generate verified teacher suffixes from failed student tau2 trajectories."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau2_evolve import benchmark
from tau2_evolve.router import router_features


def decision_prefixes(messages: list[dict], max_candidates: int = 3) -> list[tuple[int, list[dict]]]:
    candidates: list[tuple[int, list[dict]]] = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user" and "###STOP###" not in content:
            candidates.append((index, messages[:index]))
        elif role == "tool" and (message.get("requestor") in (None, "assistant")):
            candidates.append((index, messages[:index]))
    if not candidates:
        return []
    if max_candidates <= 1:
        return [candidates[-1]]
    if len(candidates) <= max_candidates:
        return list(reversed(candidates))
    # A failure may follow an irreversible environment mutation. Trying only
    # the final few turns leaves the teacher no way to repair it, so spread
    # branches across the trajectory while attempting the cheapest late
    # continuation first.
    positions = {
        round(index * (len(candidates) - 1) / (max_candidates - 1))
        for index in range(max_candidates)
    }
    return [candidates[index] for index in sorted(positions, reverse=True)]


def router_example(row: dict, prefix: list[dict], label: int, source: str) -> dict:
    return {
        "domain": row["domain"],
        "task_id": str(row["task_id"]),
        "label": int(label),
        "source": source,
        "features": router_features(prefix),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--router-data", type=Path, required=True)
    parser.add_argument("--skillbook", type=Path)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-base-url", required=True)
    parser.add_argument("--teacher-api-key", default=os.environ.get("OPD_TEACHER_API_KEY", "EMPTY"))
    parser.add_argument("--teacher-max-tokens", type=int, default=1024)
    parser.add_argument("--user-model", default="openai/evol-llm-user")
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--user-api-key", default="EMPTY")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--branch-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--no-teacher-thinking", action="store_true")
    args = parser.parse_args()
    if args.branch_attempts < 1:
        parser.error("--branch-attempts must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    rows = [json.loads(line) for line in args.traces.read_text().splitlines() if line.strip()]
    skills = json.loads(args.skillbook.read_text()) if args.skillbook else {}
    teacher = benchmark.make_llm_spec(
        args.teacher_model,
        args.teacher_base_url,
        args.teacher_api_key,
        enable_thinking=False if args.no_teacher_thinking else None,
        max_tokens=args.teacher_max_tokens,
    )
    user = benchmark.make_llm_spec(args.user_model, args.user_base_url, args.user_api_key, enable_thinking=False)
    benchmark.prime_nl_judge_routing(user)

    corrections: list[dict] = []
    router_rows: list[dict] = []
    failed_rows: list[dict] = []
    for row in rows:
        prefixes = decision_prefixes(row.get("messages", []), args.branch_attempts)
        if row.get("passed") and benchmark.trace_action_complete(row):
            for _, prefix in prefixes:
                router_rows.append(router_example(row, prefix, 0, "student_success"))
        else:
            failed_rows.append(row)

    def correct(row: dict) -> tuple[dict | None, dict | None]:
        for branch_index, prefix in decision_prefixes(row.get("messages", []), args.branch_attempts):
            try:
                candidate = benchmark.run_task(
                    domain=row["domain"],
                    task_id=str(row["task_id"]),
                    agent_spec=teacher,
                    user_spec=user,
                    seed=args.seed,
                    max_steps=args.max_steps,
                    skill_text=skills.get(row["domain"], ""),
                    message_history=prefix,
                )
            except Exception as exc:  # replay can reject an inconsistent mutated prefix
                print(f"[opd] replay failed {row['domain']}/{row['task_id']}@{branch_index}: {exc}", file=sys.stderr)
                continue
            if candidate.get("passed") and benchmark.trace_action_complete(candidate):
                candidate.update({
                    "opd_used": True,
                    "opd_branch_message_index": branch_index,
                    "student_passed": bool(row.get("passed")),
                    "student_action_complete": benchmark.trace_action_complete(row),
                    "student_termination_reason": row.get("termination_reason"),
                })
                return candidate, router_example(row, prefix, 1, "teacher_rescue")
        return None, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(correct, row): row for row in failed_rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                correction, route = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[opd] failed {row['domain']}/{row['task_id']}: {exc}", file=sys.stderr)
                continue
            if correction:
                corrections.append(correction)
                router_rows.append(route)
                print(f"[opd] rescued {row['domain']}/{row['task_id']}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corrections))
    args.router_data.parent.mkdir(parents=True, exist_ok=True)
    args.router_data.write_text("".join(json.dumps(row) + "\n" for row in router_rows))
    print(
        f"[opd] corrected={len(corrections)}/{len(failed_rows)} "
        f"router_continue={sum(row['label'] == 0 for row in router_rows)} "
        f"router_escalate={sum(row['label'] == 1 for row in router_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
