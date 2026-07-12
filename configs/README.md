# Configs, grouped by task domain

Each subdirectory holds the `.env` files for one task domain's training runs
(`source` before the corresponding `scripts/train_*.sh`, or override
variables inline). Keep new domains in their own subdirectory rather than
adding more top-level files, so domain-specific tuning (learning rate, batch
sizes, eager-mode requirements, etc.) never gets silently mixed up across
domains.

- `humaneval_mbpp/` — HumanEval + MBPP code generation (`data/raw/he_mbpp.jsonl`),
  the domain documented in `CLAUDE.md` and `docs/experiments/`.
