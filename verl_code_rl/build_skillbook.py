"""Build or update a MERA skillbook from trace rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from verl_code_rl.skills import SkillBook, make_llm_distiller


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-model", default="small")
    parser.add_argument("--large-model", default="large")
    parser.add_argument("--distiller-model", default="",
                        help="Optional OpenAI-compatible model for compact Skill distillation.")
    parser.add_argument("--distiller-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument(
        "--api-key", default=os.environ.get("DISTILLER_API_KEY", "EMPTY"),
        help="Distiller API key; defaults to DISTILLER_API_KEY from the environment.",
    )
    parser.add_argument("--distiller-max-examples", type=int, default=40)
    args = parser.parse_args()

    skillbook = SkillBook()
    if args.previous and args.previous.exists():
        skillbook.load(args.previous)

    count = 0
    with args.traces.open() as fh:
        for line in fh:
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            skillbook.update_from_trace(trace, small_model=args.small_model, large_model=args.large_model)
            count += 1

    distiller = None
    if args.distiller_model:
        distiller = make_llm_distiller(
            args.distiller_model, args.distiller_base_url, args.api_key,
            max_examples=args.distiller_max_examples,
        )
    distilled = skillbook.distill_all(distiller=distiller)
    skillbook.save(args.output)
    print(
        f"[skillbook] read={count} skills={skillbook.summary()['total_skills']} "
        f"distilled={distilled} out={args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
