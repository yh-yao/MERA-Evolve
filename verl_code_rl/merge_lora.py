"""Merge a PEFT LoRA adapter into a standalone Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def _merge_qwen35(adapter: Path, base_model: str):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    with (adapter / "adapter_config.json").open() as handle:
        adapter_config = json.load(handle)
    scale = float(adapter_config["lora_alpha"]) / float(adapter_config["r"])
    weights_path = adapter / "adapter_model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        for a_key in sorted(key for key in keys if key.endswith(".lora_A.weight")):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            if b_key not in keys:
                raise KeyError(f"missing LoRA B tensor for {a_key}")
            module_path = a_key.removeprefix("base_model.model.model.").removesuffix(
                ".lora_A.weight"
            )
            module = model.get_submodule(f"model.language_model.{module_path}")
            delta = weights.get_tensor(b_key) @ weights.get_tensor(a_key)
            module.weight.data.add_(delta.to(module.weight.dtype), alpha=scale)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with (args.adapter / "adapter_config.json").open() as handle:
        base_model = json.load(handle)["base_model_name_or_path"]

    args.output.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    if config.model_type == "qwen3_5":
        model = _merge_qwen35(args.adapter, base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(args.adapter))
        model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(args.output)
    if config.model_type == "qwen3_5":
        AutoProcessor.from_pretrained(base_model, trust_remote_code=True).save_pretrained(
            args.output
        )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
