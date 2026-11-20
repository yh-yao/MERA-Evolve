# MERA-Evolve TAU-2 reproduction artifacts

Snapshot of the 4B-student / 9B-teacher TAU-2 reproduction run, 2026-11-16 to 2026-11-20,
on a single node with 4 × H20 95 GiB. Source of truth for numbers: the JSON/JSONL files
here; prose summaries are in `reports/` and each stage's `SUMMARY.md`.

## What is included

| Path | Content |
|---|---|
| `reports/REPRODUCTION_REPORT.md` | Full reproduction report (questions, findings, deviations, pitfalls) |
| `results/e0_base_val/` | 4B base VAL (seeds 300/1300) and frontier (58 tasks × 8) |
| `results/e1_r1/` | R1 sampling over valid AReaL + TRAIN tasks (4 trials/task) |
| `results/e2_arm1_lr1e-5/` | GRPO arm 1: training log, config, GRPO parquet, per-checkpoint frontier evals, summary |
| `results/e6_teacher_val/` | Teacher Step 0: raw, amended and matched-budget verdicts, pooled analysis |
| `results/section_i_pipeline/` | Infrastructure acceptance run (timings, truncation, TIS metrics) |
| `results/sft_only/` | SFT-only rehearsal run: pairs, filter record, summary, VAL comparison |
| `results/final_eval/` | Final EVAL (Q4): base vs SFT, 536 conversations per policy, stats |
| `results/e7_sft_selector_baseline_v2/` | Rule-based TRAIN/AReaL-only selector outputs and manifest |
| `results/e8_data_utility_router_v3/` | Per-cohort proxy contracts (CPU pre-registration) |
| `results/e9_retail_utility_proxy_v3/` | Retail data-utility pilot: plan, 6 arms' eval traces, labels, summary |
| `logs/` | Stage logs (gzipped) |

Text artifacts (`*.jsonl`, `*.json`, `*.md`, `*.txt`, `*.log`) are gzip-compressed
(`.gz`). Small `.parquet` files are kept uncompressed.

## What is deliberately excluded

- `verl_checkpoints/` — full-model FSDP dumps (≈2.85 TB; each 20.8 GB). Not reproducible
  evidence beyond the extracted LoRA adapters, which are excluded as well.
- LoRA adapter weights — this snapshot is reports + raw traces only. They can be added
  separately (≈1.4 GB) if a binary checkpoint is required.
- `*.pt` optimizer/scheduler states and `served_adapters/` key-converted duplicates.
- Model weights (`models/Qwen3.5-4B`, `Qwen3.5-9B`).

## One split file

`results/e2_arm1_lr1e-5/train/group_O/train.log.gz` exceeded GitHub's 100 MB per-file
limit, so it was split (git-lfs is not available on the reproduction node). Reassemble with:

```bash
cat train.log.gz.part-00 train.log.gz.part-01 > train.log.gz
gunzip train.log.gz
```

## Verification

`MANIFEST.sha256` lists the SHA-256 of every file in this directory (excluding itself).

```bash
sha256sum -c MANIFEST.sha256
```

## Method notes

- Split `experiments/tau-2/data/partition_v2.json`: TRAIN 139 / VAL 67 / EVAL 67.
- VAL and EVAL were used only as declared; EVAL was reported once and never used for
  selection. The selector (`e7`) and the utility pilot (`e9`) touch only TRAIN and valid
  AReaL tasks.
- `.env`, API keys and model weights are not part of this snapshot.
