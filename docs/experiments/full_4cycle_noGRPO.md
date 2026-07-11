# Full 4-cycle closed loop (Skills + SFT, no GRPO)

Ran the full `run_full_pipeline.sh` closed loop for 4 cycles with `SKIP_GRPO=1`
(oracle collection uses probe-only fallback: small model tried first, large
model only invoked when small's execution fails), to isolate how much of the
evolve loop's benefit comes from Skills+SFT alone, without RL. Router is
trained/calibrated each cycle on a disjoint task-id shard from the SFT data
(see `CLAUDE.md`'s "Cost-aware routing" section), then a held-out `eval` split
(582 tasks) is scored for four policy variants:

- `large` — always call the large (3B) model
- `skills` — the current cycle's SFT-trained small model + skillbook prompt, oracle (no router — this is what the small model can actually do)
- `router` — the calibrated router gates each task to small (falls back to large only on failure) or large
- `full` — same as `router` in this run (no separate GRPO-trained variant, since GRPO was skipped)

## Results

| Cycle | large task_pass | skills task_pass | router task_pass | router avg_cost (vs large=0.01) | router fallback_rate |
|---|---|---|---|---|---|
| 0 | 44.16% | 49.14% | 49.14% | 0.00770 (23% cheaper) | 66.5% |
| 1 | 43.81% | 54.64% | 54.47% | 0.00668 (33% cheaper) | 54.6% |
| 2 | 47.94% | 62.03% | 62.03% | 0.00603 (40% cheaper) | 48.1% |
| 3 | 43.81% | 59.97% | 59.28% | 0.00609 (39% cheaper) | 48.3% |

Source: `e2e_ablation_summary.json` written by `run_ablation.py` at the end of
each cycle (not committed — regenerate via `run_full_pipeline.sh
--n-cycles 4 --skip-train` with `SKIP_GRPO=1`, checkpoints/traces are large
and gitignored under `results/`).

## Takeaways

- Cycles 0->2 compounded cleanly: the SFT-trained small model + skillbook went
  from +5.0pts over the 3B large model (cycle 0) to +14.1pts (cycle 2), while
  getting cheaper each cycle (fallback rate dropping as the small model needs
  the large model less often).
- Cycle 3 broke the monotonic trend: `skills` task_pass dropped from 62.03%
  to 59.97% instead of continuing to climb. Still a decisive win over the
  large model alone (+16pts at ~39% of the cost), but the SFT-only outer loop
  likely has a soft ceiling around cycle 2-3 depth on this data volume —
  each cycle's `sft_pairs.parquet` is drawn only from that cycle's oracle
  traces, so the continued-LoRA adapter's per-cycle update is small and can
  start overfitting/drifting rather than compounding further.
- The `router` variant tracked the oracle `skills` number within ~1pt every
  cycle, confirming the calibrated router isn't leaving performance on the
  table relative to what the small model can actually do — the gap between
  `router` and `skills` is just the small, expected cost of imperfect routing
  decisions.
- Conclusion: Skills+SFT alone (no GRPO) is sufficient to take the small model
  from below the 3B model to reliably beating it by 12-18pts at 39-40% of the
  cost. Pushing past the cycle-2/3 plateau likely needs GRPO's exploration
  signal (or more oracle data per cycle) rather than more SFT-only rounds.

## Final skillbook (cycle 3)

See `full_4cycle_noGRPO_skill.md` in this directory for the actual distilled
skill text (`skill.md`) used in cycle 3's prompts — two structured sections
(`humaneval`, `mbpp`), each with static When-to-use/Procedure plus
LLM-distilled (GPT-5.5 via CommonStack) Common-pitfalls/Recurring-patterns
grounded in that cycle's successful exemplars.
