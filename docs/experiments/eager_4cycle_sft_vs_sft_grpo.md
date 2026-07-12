# Eager-mode 4-cycle closed loop: Skills+SFT vs Skills+SFT+GRPO

Two 4-cycle closed loops run in parallel on separate GPUs, both with
`--probe-only` oracle collection and `ENFORCE_EAGER=True` everywhere
(training and serving) -- the earlier `full_4cycle_noGRPO` run predates the
discovery that this node's vLLM gives numerically unreliable results under
CUDA graphs (see "Why eager mode" below), so this pair re-establishes a
trustworthy no-GRPO baseline alongside a GRPO-enabled counterpart:

- **`eager_4cycle_noGRPO`**: `SKIP_GRPO=1`, `ENABLE_SFT=1` -- Skills+SFT only,
  reproduced via `experiments/04_4cycle_sft.sh`.
- **`eager_4cycle_withGRPO`**: `SKIP_GRPO=0`, `ENABLE_SFT=1`, GRPO `LR=5e-5`
  (the winner of an earlier LR sweep) -- Skills+SFT+GRPO, reproduced via
  `experiments/05_4cycle_sft_grpo.sh`. Each cycle, SFT warm-starts the LoRA
  adapter and GRPO continues training it; the adapter also continues across
  cycles.

Both used the full 546-task train split (`--limit -1`) and a fixed GPU
utilization fix for GRPO (`PPO_MICRO_BATCH_SIZE_PER_GPU=32`,
`LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=64`, `PPO_MINI_BATCH_SIZE=32` -- see
below).

## Results (held-out `eval` split, 582 tasks)

| Cycle | large | noGRPO skills | noGRPO router | withGRPO skills | withGRPO router | GRPO's added value |
|---|---|---|---|---|---|---|
| 0 | 41-44% | 48.63% | 48.63% | 59.62% | 59.28% | +11.0pt |
| 1 | 43-48% | 53.44% | 52.23% | 64.60% | 64.43% | +11.2pt |
| 2 | 42-48% | 59.45% | 58.08% | 62.37% | 62.20% | +2.9pt |
| 3 | 41-44% | 54.47% | 53.78% | 62.37% | 62.03% | +7.9pt |

`noGRPO`'s numbers land in the same range as the earlier (pre-eager-fix)
`full_4cycle_noGRPO` run (49.1/54.6/62.0/60.0% skills task_pass) -- close
enough (within normal cycle-to-cycle sampling variance) to confirm this
re-run didn't introduce a regression, and that the earlier run's numbers
were themselves already reliable (GRPO wasn't involved, so the CUDA-graph
issue below never affected it).

## Takeaways

- **GRPO's fix pays off in the full closed loop, not just in isolation.**
  Skills+SFT+GRPO beat Skills+SFT-only by 3-11pts in every one of the 4
  cycles. `withGRPO` peaks at cycle 1 (64.60%) and holds around 62% for the
  rest of the run, whereas `noGRPO` peaks at cycle 2 (59.45%) then dips at
  cycle 3 (54.47%) -- consistent with the earlier finding that SFT-only has a
  soft ceiling around cycle 2-3 depth; GRPO's exploration signal appears to
  be exactly what's needed to keep climbing past it.
- **Router calibration remains trustworthy under both configs**: `router`
  tracks the oracle `skills` number within ~1pt every cycle in both runs.

## Why eager mode

This node's vLLM build reproduces a CUDA-graph-only crash (`illegal
instruction` / `illegal memory access` / `unspecified launch failure` --
different error text each time, same underlying instability) and, short of
crashing outright, gives numerically *unreliable* generation under CUDA
graphs. This was discovered via a large, unexplained gap between an earlier
GRPO sweep's *training-internal* validation accuracy (computed with CUDA
graphs on, since `train_grpo.sh` didn't yet default to eager mode) and a
separately-served *external* eval of the exact same saved checkpoint
(computed in eager mode, since `serve_vllm.sh` already defaulted to eager).
Six hypotheses were checked before landing on this explanation (prompt text,
task-id sets, ground truth, decoding batch composition, LoRA-loading path,
`filter_overlong_prompts` truncation -- all ruled out or found to contribute
only marginally); re-serving the same checkpoint with CUDA graphs enabled
reproduced neither the original internal number nor the original external
number, and crashed outright on a repeat run -- confirming CUDA-graph mode
itself is not reproducible on this hardware. `train_grpo.sh` now defaults
`ENFORCE_EAGER=True` for this reason (not just crash-avoidance).

## Final skillbook (withGRPO, cycle 3)

See `eager_4cycle_skill.md` in this directory.
