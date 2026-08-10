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

## Qwen3.5-4B endpoint-only reference

The unadapted Qwen3.5-4B model, used as both agent and user simulator with no
SkillBook, router, or fallback, passes 17/35 tasks (48.6%): 5/9 Airline, 9/18
Retail, and 3/8 Telecom. Two malformed-JSON generations are counted as
failures. The raw artifact is
`results/tau2_qwen35_4b_baseline_20260728_184300/eval.jsonl`.

This reference uses a 512-token cap per agent and user turn, Triton GDN
prefill, one sequence, and disabled prefix caching. A 1024-token Qwen3.5-4B
run reproducibly stalled in vLLM's GDN decode on a long completion. Because
the strict 2B comparison above uses a 1024-token agent cap, the 4B result is a
descriptive capacity reference rather than a paired ablation.

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

## Evidence and limitations

The strict comparison is suitable for a narrowly qualified rebuttal claim:

- Base and GRPO evaluation contain the same 35 `(domain, task_id)` pairs.
- Neither evaluation uses SkillBook or fallback.
- The Qwen3.5-4B user simulator is recorded under its served name
  `evol-llm-user`; the corresponding server log records the actual model path.
- The 14/35 to 18/35 difference is one evaluation run per checkpoint, not a
  multi-seed estimate, and its paired test is not significant.
- The JSONL rows do not embed a complete config snapshot. Configuration
  provenance therefore depends on the associated server and evaluation logs.

See `rebuttal/TAU2_PROVENANCE.md` for artifact hashes and evidence paths.

## Exploratory four-cycle run

The older OPD+SkillBook+SFT+GRPO+router run under
`results/tau2_qwen35_opd_router_4cycle_20260716` is not a clean four-cycle
learning comparison. Its raw pass counts are 11/35, 13/35, 17/35, and 15/35,
but cycles 0 and 1 contain five and two collection errors, respectively.
Cycles 2 and 3 contain no recorded collection exception; the cycle-3 routed
evaluation is 16/35. This run may be reported only as exploratory integration
evidence with the early-cycle confound stated, not as evidence of monotonic
improvement or as a matched base-versus-method result.

| Cycle/evaluation | Airline | Retail | Telecom | Overall | Collection exceptions | Router escalations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 standard | 4/9 | 7/18 | 0/8 | 11/35 | 5 | - |
| 0 router enabled | 5/9 | 6/18 | 0/8 | 11/35 | 5 | 0 |
| 1 standard | 4/9 | 9/18 | 0/8 | 13/35 | 2 | - |
| 2 standard | 4/9 | 8/18 | 5/8 | 17/35 | 0 | - |
| 2 router enabled | 5/9 | 7/18 | 5/8 | 17/35 | 0 | 0 |
| 3 standard | 4/9 | 7/18 | 4/8 | 15/35 | 0 | - |
| 3 router enabled | 7/9 | 9/18 | 0/8 | 16/35 | 0 | 0 |

The router threshold was 0.5, while the maximum predicted escalation
probability was 0.361, 0.427, and 0.403 in cycles 0, 2, and 3. Thus the router
made zero escalations; differences between standard and router-enabled files
are repeated-sampling variation, not measured router benefit. Cycle 1 has no
router-enabled evaluation artifact.
