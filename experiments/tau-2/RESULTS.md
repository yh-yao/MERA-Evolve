# TAU-2 results

All numbers below use the same fixed 35-task EVAL split, Qwen3.5-4B as the
user simulator, non-thinking chat templates, no fallback rollout, and a
1024-token cap per agent turn. The held-out metric is the official TAU-2 pass
rate; shaped rewards are used only during training.

## Qwen3.5-2B ablations

| Agent | Airline | Retail | Telecom | Overall |
| --- | ---: | ---: | ---: | ---: |
| Base | 5/9 | 7/18 | 2/8 | 14/35 (40.0%) |
| GRPO pilot, step 5 | 4/9 | 6/18 | 4/8 | 14/35 (40.0%) |
| GRPO balanced v3 | 5/9 | 10/18 | 3/8 | **18/35 (51.4%)** |

Balanced v3 improves the observed pass rate by 4/35, or 11.4 percentage
points. In the paired comparison it has 7 wins, 3 losses, and 25 ties versus
base. The one-sided exact McNemar p-value is 0.171875, so the gain is not yet
significant at alpha 0.05 on this small split.

## Balanced v3 configuration

- Base model: `Qwen/Qwen3.5-2B`; strict GRPO-only LoRA rank 16, alpha 32.
- Trainer GPU and rollout GPU are separated with VERL 0.8's one-step
  off-policy pipeline and the `nixl_tau2` checkpoint backend.
- Nine updates, learning rate `2e-6`, two prompts per update, eight generations
  per prompt, and deterministic domain-interleaved input with shuffle disabled.
- Maximum prompt/response lengths: 8192/4096; maximum 12 user and assistant
  turns; sampling temperature 1.0 and top-p 0.98.
- Stable post-warmup throughput: about 1,210-1,270 tokens/s. The final update
  reached 1,262 tokens/s, KL 0.0072, and 25% response clipping without OOM or
  NaN.

The reproducible entry point is `03_grpo_only.sh`. Raw JSONL traces, logs,
checkpoints, adapters, and merged model weights remain under `results/` and
are intentionally excluded from Git.
