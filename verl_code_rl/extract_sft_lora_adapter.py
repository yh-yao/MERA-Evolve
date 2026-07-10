"""Extract a clean PEFT LoRA adapter from a verl native SFT checkpoint.

verl's generic FSDP checkpoint saver (used by both the SFT trainer and PPO's
non-actor workers) only dumps a raw `model.state_dict()` shard -- it does not
write a `lora_adapter/adapter_config.json` the way PPO's actor worker does
(that logic is specific to `verl/workers/fsdp_workers.py`, not the shared
checkpoint manager). This reconstructs the same LoRA-wrapped model used during
training and re-saves just the adapter via PEFT's own `save_pretrained`, which
serve_vllm.sh (and everything downstream) already auto-detects.

Single-GPU (world_size=1, FSDP NO_SHARD) checkpoints only: `model.state_dict()`
under NO_SHARD is the full, unsharded state dict, so no shard-gathering logic
is needed here -- it's a plain state dict load.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_target_modules(raw: str) -> list[str]:
    return [m.strip() for m in raw.strip("[]").split(",") if m.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. .../verl_checkpoints/global_step_3")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--target-modules", required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    ckpt_dir = Path(args.checkpoint_dir)
    shard = ckpt_dir / "model_world_size_1_rank_0.pt"
    if not shard.exists():
        raise SystemExit(
            f"{shard} not found -- this extractor only supports single-GPU "
            "(world_size=1) checkpoints. For multi-GPU runs, gather a full "
            "state dict first."
        )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    peft_model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=_parse_target_modules(args.target_modules),
        bias="none",
    ))

    state_dict = torch.load(shard, map_location="cpu", weights_only=False)
    missing, unexpected = peft_model.load_state_dict(state_dict, strict=False)
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    missing_trainable = [m for m in missing if m in trainable]
    if missing_trainable:
        raise SystemExit(
            f"{len(missing_trainable)} trainable LoRA params were not found in "
            f"the checkpoint (e.g. {missing_trainable[:5]}) -- key naming "
            "assumption doesn't match this checkpoint's state dict."
        )
    print(f"[extract_sft_lora_adapter] loaded checkpoint ({len(unexpected)} unexpected keys ignored)")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(args.output)
    print(f"[extract_sft_lora_adapter] wrote adapter to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
