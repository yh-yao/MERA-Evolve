"""Fold tau2 TRAIN trace outcomes into a domain-bucketed SkillBook and
distill each domain's pitfalls/patterns from its successful exemplars.
Mirrors verl_code_rl/build_skillbook.py's role.

Usage (tau2_stage2 venv):
  PYTHONPATH=experiments/tau-2 .../.venv_tau2/bin/python3 \
    -m tau2_evolve.build_skillbook \
    --traces results/tau2_skills_only/train_traces.jsonl \
    --output results/tau2_skills_only/skillbook.json \
    [--distiller-model openai/gpt-5.5 --distiller-base-url https://api.commonstack.ai/v1 --api-key ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tau2_evolve.skills import SkillBook, make_llm_distiller


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--distiller-model", default=None)
    ap.add_argument("--distiller-base-url", default="https://api.commonstack.ai/v1")
    ap.add_argument("--api-key", default=os.environ.get("DISTILLER_API_KEY"))
    args = ap.parse_args()

    book = SkillBook()
    n_by_domain: dict[str, list[int]] = {}
    for line in args.traces.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row["domain"]
        book.add_exemplar(domain, row)
        n_by_domain.setdefault(domain, [0, 0])
        n_by_domain[domain][1] += 1
        n_by_domain[domain][0] += int(row.get("passed", False))

    for domain, (passed, total) in sorted(n_by_domain.items()):
        print(f"[build_skillbook] {domain}: {passed}/{total} exemplars available")

    distiller = None
    if args.distiller_model:
        api_key = args.api_key or os.environ.get("COMMONSTACK_API_KEY", "")
        distiller = make_llm_distiller(args.distiller_model, args.distiller_base_url, api_key)
        print(f"[build_skillbook] distilling pitfalls/patterns via {args.distiller_model}")

    book.distill_all(distiller=distiller)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    book.save(args.output)
    print(f"[build_skillbook] wrote {args.output}")
    for domain, skill in book.skills.items():
        print(f"--- {domain} ---")
        print(skill.render())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
