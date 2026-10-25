"""Find where a failed student trajectory first goes wrong; get the teacher's fix.

DAgger-style correction (improvement plan v2, Phase 1). For each failed
student trajectory:

1. Walk its assistant turns in order. At each one, rebuild the agent state
   from the student's own history and ask the teacher agent for its next
   message through tau2's normal agent path (same system prompt, tools and
   rendering the evaluation endpoint sees).
2. The first turn whose action differs is the error point: a tool call vs.
   text, a different tool, or different arguments. Two text replies are not
   compared (wording differences are not errors).
3. Verify the fix: replay the history up to the error point, answer it
   with the teacher's message, then let the *student* finish the task; if
   that fails, let the teacher finish. Early divergences are often a
   different but valid ordering, so up to ``--max-points`` divergences are
   tried in order and the first one that verifies is kept.

Only the corrected turn becomes a training target (see
``error_points_to_sft``); everything before it is context the student
reached by itself.

Usage (tau2 venv):
  python -m tau2_evolve.collect_error_points --traces train_k8_*.jsonl \\
      --teacher-base-url http://HOST:8270/v1 --student-base-url http://127.0.0.1:8280/v1 \\
      --user-base-url http://127.0.0.1:8281/v1 --out error_points.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tau2_evolve import benchmark


def action_signature(message: dict) -> tuple:
    """Comparable form of an assistant turn: ("text",) or sorted tool calls."""
    calls = message.get("tool_calls") or []
    if not calls:
        return ("text",)
    sig = []
    for call in calls:
        args = call.get("arguments")
        if args is None and call.get("function"):
            args = call["function"].get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        sig.append((call.get("name") or (call.get("function") or {}).get("name"), json.dumps(args, sort_keys=True)))
    return ("tools", tuple(sorted(sig)))


def divergence_kind(student: dict, teacher: dict) -> str | None:
    s, t = action_signature(student), action_signature(teacher)
    if s == t or (s[0] == "text" and t[0] == "text"):
        return None
    if s[0] != t[0]:
        return "tool_vs_text"
    s_names = sorted(n for n, _ in s[1])
    t_names = sorted(n for n, _ in t[1])
    return "different_tool" if s_names != t_names else "different_arguments"


def _agent_turn_inputs(history: list[Any]):
    """Split history into (prior state messages, the message the agent answers)."""
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    tail = []
    i = len(history)
    while i > 0 and isinstance(history[i - 1], ToolMessage):
        tail.insert(0, history[i - 1])
        i -= 1
    if len(tail) > 1:
        return history[:i], MultiToolMessage(role="tool", tool_messages=tail)
    return history[:-1], history[-1]


def build_agent(domain: str, task_id: str, agent_spec, user_spec, seed: int):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.data_model.tasks import Task
    from tau2.runner import build_text_orchestrator

    cfg = TextRunConfig(
        domain=domain, agent="llm_agent", llm_agent=agent_spec.model, llm_args_agent=dict(agent_spec.args),
        user="user_simulator", llm_user=user_spec.model, llm_args_user=dict(user_spec.args),
        seed=seed, max_steps=60, max_errors=10, num_trials=1,
    )
    task = Task.model_validate(benchmark.load_task(domain, task_id))
    return build_text_orchestrator(cfg, task, seed=seed).agent


def divergence_points(row: dict, teacher_spec, user_spec, seed: int, max_turns: int, max_points: int):
    """Assistant turns where the teacher would act differently, in order."""
    from pydantic import TypeAdapter
    from tau2.data_model.message import AssistantMessage, Message

    msgs = TypeAdapter(list[Message]).validate_python(row["messages"])
    first_user = next(i for i, m in enumerate(msgs) if getattr(m, "role", None) == "user")
    teacher = build_agent(row["domain"], str(row["task_id"]), teacher_spec, user_spec, seed)
    checked = 0
    points: list[dict] = []
    for t, msg in enumerate(msgs):
        if t <= first_user or not isinstance(msg, AssistantMessage):
            continue
        checked += 1
        if checked > max_turns:
            break
        prior, incoming = _agent_turn_inputs(msgs[:t])
        state = teacher.get_init_state(message_history=prior)
        reply, _ = teacher.generate_next_message(incoming, state)
        student_d = msg.model_dump(mode="json")
        teacher_d = reply.model_dump(mode="json")
        kind = divergence_kind(student_d, teacher_d)
        if kind:
            points.append({"turn_index": t, "kind": kind, "turns_checked": checked,
                           "student_message": student_d, "teacher_message": teacher_d})
            if len(points) >= max_points:
                break
    return points, checked


def verify(row: dict, point: dict, agent_spec, user_spec, seed: int, max_steps: int) -> dict:
    """Replay up to the error point, answer it with the teacher's message, finish with ``agent_spec``."""
    result = benchmark.run_task(
        domain=row["domain"], task_id=str(row["task_id"]), agent_spec=agent_spec, user_spec=user_spec,
        seed=seed, max_steps=max_steps, message_history=row["messages"][: point["turn_index"]],
        first_agent_message=point["teacher_message"],
    )
    return {"passed": bool(result.get("passed")), "reward": result.get("reward", 0.0),
            "termination_reason": result.get("termination_reason")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--teacher-model", default="openai/evol-llm-teacher")
    ap.add_argument("--teacher-base-url", required=True)
    ap.add_argument("--student-model", default="openai/evol-llm-agent")
    ap.add_argument("--student-base-url", required=True)
    ap.add_argument("--user-model", default="openai/evol-llm-user")
    ap.add_argument("--user-base-url", required=True)
    ap.add_argument("--per-task", type=int, default=2, help="failed trajectories sampled per task")
    ap.add_argument("--max-turns", type=int, default=20, help="assistant turns compared per trajectory")
    ap.add_argument(
        "--max-points", type=int, default=3,
        help="divergences verified per trajectory, in order; the first that verifies is kept "
             "(early divergences are often a different but valid ordering, not the error)",
    )
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")

    teacher = benchmark.make_llm_spec(args.teacher_model, args.teacher_base_url, temperature=0.0,
                                      enable_thinking=False, max_tokens=1024, timeout=args.request_timeout)
    student = benchmark.make_llm_spec(args.student_model, args.student_base_url, temperature=0.0,
                                      enable_thinking=False, max_tokens=1024, timeout=args.request_timeout)
    user = benchmark.make_llm_spec(args.user_model, args.user_base_url, temperature=0.0,
                                   enable_thinking=False, timeout=args.request_timeout)
    benchmark.prime_nl_judge_routing(user)

    by_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in args.traces:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") or row.get("passed") or not row.get("messages"):
                continue
            by_task[(row["domain"], str(row["task_id"]))].append(row)
    jobs = [r for rows in by_task.values() for r in sorted(rows, key=lambda r: r.get("trial", 0))[: args.per_task]]
    done = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["domain"], r["task_id"], r.get("trial", 0)))
    jobs = [r for r in jobs if (r["domain"], str(r["task_id"]), r.get("trial", 0)) not in done]
    print(f"[error-points] failed trajectories to inspect: {len(jobs)} over {len(by_task)} tasks", file=sys.stderr)

    def process(row: dict) -> dict:
        base = {"domain": row["domain"], "task_id": str(row["task_id"]), "trial": row.get("trial", 0),
                "system_prompt": row["system_prompt"], "tools": row["tools"]}
        points, checked = divergence_points(row, teacher, user, args.seed, args.max_turns, args.max_points)
        base.update({"turns_checked": checked, "divergences_found": len(points),
                     "turn_index": None, "kind": None, "verified_by": None, "attempts": []})
        for point in points:
            attempt = {"turn_index": point["turn_index"], "kind": point["kind"]}
            attempt["student_continuation"] = verify(row, point, student, user, args.seed, args.max_steps)
            verified = "student" if attempt["student_continuation"]["passed"] else None
            if not verified:
                attempt["teacher_continuation"] = verify(row, point, teacher, user, args.seed, args.max_steps)
                verified = "teacher" if attempt["teacher_continuation"]["passed"] else None
            base["attempts"].append(attempt)
            if verified:
                base.update({k: point[k] for k in ("turn_index", "kind", "student_message", "teacher_message")})
                base["context_messages"] = row["messages"][: point["turn_index"]]
                base["verified_by"] = verified
                break
        return base

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with args.out.open("a" if args.resume else "w") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process, r): r for r in jobs}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad replay must not stop the run
                print(f"[error-points] FAILED {r['domain']}/{r['task_id']} t{r.get('trial', 0)}: {exc}", file=sys.stderr)
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            counts[rec.get("kind")] += 1
            counts[f"verified_by_{rec.get('verified_by')}"] += 1
            print(f"[error-points] {rec['domain']}/{rec['task_id']} t{rec['trial']} turn={rec.get('turn_index')} "
                  f"kind={rec.get('kind')} verified={rec.get('verified_by')}", file=sys.stderr)
    print(f"[error-points] done {dict(counts)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
