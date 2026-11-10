"""Extract a clean PEFT LoRA adapter from a verl native SFT checkpoint.

verl's generic FSDP checkpoint saver (used by both the SFT trainer and PPO's
non-actor workers) only dumps a raw `model.state_dict()` shard -- it does not
write a `lora_adapter/adapter_config.json` the way PPO's actor worker does
(that logic is specific to `verl/workers/fsdp_workers.py`, not the shared
checkpoint manager). This reconstructs the same LoRA-wrapped model used during
training and re-saves just the adapter via PEFT's own `save_pretrained`, which
serve_vllm.sh (and everything downstream) already auto-detects.

Single-GPU checkpoints (world_size=1, FSDP NO_SHARD) hold the full state dict
in `model_world_size_1_rank_0.pt`. Multi-GPU actor checkpoints are sharded
across `model_world_size_<N>_rank_<r>.pt`; they are merged with verl's own
FSDP shard merger (`verl.model_merger`, the same code behind
`python -m verl.model_merger merge`) without writing a full HF model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _parse_target_modules(raw: str) -> list[str]:
    return [m.strip() for m in raw.strip("[]").split(",") if m.strip()]


def _normalize_checkpoint_keys(
    state_dict: dict[str, object], model_keys: set[str]
) -> tuple[dict[str, object], int]:
    """Map multimodal Qwen checkpoints onto their text-only PEFT model."""
    normalized: dict[str, object] = {}
    remapped = 0
    for key, value in state_dict.items():
        target_key = key
        if key not in model_keys and ".language_model." in key:
            candidate = key.replace(".language_model.", ".", 1)
            if candidate in model_keys:
                target_key = candidate
                remapped += 1
        normalized[target_key] = value
    return normalized, remapped


def _checkpoint_world_size(ckpt_dir: Path) -> int:
    shards = sorted(ckpt_dir.glob("model_world_size_*_rank_0.pt"))
    if len(shards) != 1:
        raise SystemExit(f"expected one model_world_size_<N>_rank_0.pt in {ckpt_dir}, found {len(shards)}")
    return int(re.search(r"model_world_size_(\d+)_rank_0\.pt$", shards[0].name).group(1))


def _load_full_state_dict(ckpt_dir: Path) -> dict[str, object]:
    """Full (unsharded) actor state dict from a verl FSDP checkpoint of any world size, in its stored dtype."""
    import torch
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # older torch
        from torch.distributed._tensor import DTensor

    world_size = _checkpoint_world_size(ckpt_dir)
    paths = [ckpt_dir / f"model_world_size_{world_size}_rank_{r}.pt" for r in range(world_size)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{ckpt_dir}: missing shards {missing}")
    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    if world_size == 1:
        return shards[0]
    print(f"[extract_sft_lora_adapter] merging {world_size} FSDP shards")
    # Same rules as verl.model_merger's FSDP merger, without its bf16 cast (the actor stores fp32 LoRA weights)
    # and without its non-DTensor mesh inference, which counts one shard and would merge rank 0 only.
    merged = {}
    for key in shards[0]:
        parts = [sd[key] for sd in shards]
        if isinstance(parts[0], DTensor):
            placements = [p for p in parts[0].placements]
            if len(placements) != 1:
                raise SystemExit(f"{key}: {len(placements)}-D device mesh (HSDP/TP) is not supported")
            placement, local = placements[0], [p._local_tensor for p in parts]
            if placement.is_replicate():
                merged[key] = local[0]
            elif placement.is_shard():
                merged[key] = torch.cat(local, dim=placement.dim).contiguous()
            else:
                raise SystemExit(f"{key}: unsupported placement {placement}")
            if tuple(merged[key].shape) != tuple(parts[0].shape):  # a DTensor knows its global shape
                raise SystemExit(f"{key}: merged shape {tuple(merged[key].shape)} != global {tuple(parts[0].shape)}")
        elif isinstance(parts[0], torch.Tensor):
            merged[key] = torch.cat(parts, dim=0).contiguous()
        else:
            merged[key] = parts[0]
    return merged


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
    _checkpoint_world_size(ckpt_dir)  # fail fast before loading the base model

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

    state_dict = _load_full_state_dict(ckpt_dir)
    state_dict, remapped = _normalize_checkpoint_keys(
        state_dict, set(peft_model.state_dict())
    )
    missing, unexpected = peft_model.load_state_dict(state_dict, strict=False)
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    missing_trainable = [m for m in missing if m in trainable]
    if missing_trainable:
        raise SystemExit(
            f"{len(missing_trainable)} trainable LoRA params were not found in "
            f"the checkpoint (e.g. {missing_trainable[:5]}) -- key naming "
            "assumption doesn't match this checkpoint's state dict."
        )
    print(
        "[extract_sft_lora_adapter] loaded checkpoint "
        f"({remapped} keys remapped, {len(unexpected)} unexpected keys ignored)"
    )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(args.output)
    print(f"[extract_sft_lora_adapter] wrote adapter to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
