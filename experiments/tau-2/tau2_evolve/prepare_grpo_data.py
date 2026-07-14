"""Convert tau2 trace/task metadata into verl multi-turn interaction parquet."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import datasets

from tau2_evolve import benchmark
from tau2_evolve.traces_to_sft import _render_tools_block


def first_user_message(row: dict) -> str:
    for message in row["messages"]:
        if message.get("role") == "user" and message.get("content"):
            return message["content"]
    raise ValueError(f"trace {row.get('domain')}/{row.get('task_id')} has no user message")


def balance_and_interleave_domains(rows: list[dict]) -> list[dict]:
    """Repeat minority domains and emit a deterministic round-robin order."""
    counts = Counter(row["ability"] for row in rows)
    if not counts:
        return []
    target = max(counts.values())
    by_domain = {
        domain: [row for row in rows if row["ability"] == domain]
        for domain in sorted(counts)
    }
    return [
        by_domain[domain][index % len(by_domain[domain])]
        for index in range(target)
        for domain in by_domain
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument(
        "--skillbook", type=Path,
        help="Optional SkillBook JSON. Omit it for a strict GRPO-only run.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--user-base-url", default="http://127.0.0.1:8211/v1")
    ap.add_argument(
        "--user-thinking", action=argparse.BooleanOptionalAction, default=None,
        help="Enable or disable thinking for the user simulator chat template.",
    )
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument(
        "--balance-domains", action=argparse.BooleanOptionalAction, default=True,
        help="Repeat minority domains and interleave them in deterministic round-robin order.",
    )
    args = ap.parse_args()
    skills = json.loads(args.skillbook.read_text()) if args.skillbook else {}
    user_spec = benchmark.make_llm_spec(
        "openai/evol-llm-user", args.user_base_url, enable_thinking=args.user_thinking
    )
    contexts: dict[str, tuple[str, list[dict]]] = {}
    rows = []
    skipped = 0
    for index, line in enumerate(args.traces.read_text().splitlines()):
        trace = json.loads(line)
        if not trace.get("messages"):
            skipped += 1
            continue
        domain = trace["domain"]
        if domain not in contexts:
            contexts[domain] = benchmark.build_agent_context(
                domain=domain,
                task_id=str(trace["task_id"]),
                user_spec=user_spec,
                seed=args.seed,
                skill_text=skills.get(domain, ""),
            )
        system_prompt, tools = contexts[domain]
        initial_user = first_user_message(trace)
        rows.append({
            "data_source": "tau2",
            "agent_name": "tau2_agent",
            "prompt": [
                {"role": "system", "content": system_prompt + _render_tools_block(tools)},
                {"role": "user", "content": initial_user},
            ],
            "ability": trace["domain"],
            "reward_model": {"style": "rule", "ground_truth": "tau2_environment"},
            "extra_info": {
                "split": "train", "index": index,
                "interaction_kwargs": {
                    "name": "tau2", "domain": trace["domain"], "task_id": str(trace["task_id"]),
                    "seed": args.seed, "max_steps": args.max_steps,
                    "user_model": "openai/evol-llm-user", "user_base_url": args.user_base_url,
                    "user_thinking": args.user_thinking,
                    "skill_text": skills.get(domain, ""),
                    "expected_initial_user": initial_user,
                },
            },
        })
    raw_counts = Counter(row["ability"] for row in rows)
    if args.balance_domains and raw_counts:
        rows = balance_and_interleave_domains(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(args.output))
    print(
        f"[prepare_grpo_data] wrote {len(rows)} rows to {args.output} "
        f"(source domains={dict(sorted(raw_counts.items()))}, balanced={args.balance_domains}; "
        f"skipped {skipped} traces without messages)"
    )


if __name__ == "__main__":
    main()
