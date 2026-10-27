"""Merge a PEFT LoRA adapter trained on Qwen3_5ForCausalLM into the original Qwen3.5 checkpoint.

The A800 SFT adapters (train_lora_cc.py) also cover the GDN projections
(``linear_attn.in_proj_qkv/z/a/b``, ``out_proj``), which vLLM's LoRA path does
not serve, so evaluation uses a merged standalone checkpoint instead.

Adapter keys ``base_model.model.model.layers.N.<path>.lora_{A,B}.weight`` map to
``model.language_model.layers.N.<path>.weight`` in the original
(Qwen3_5ForConditionalGeneration) shards. Every LoRA pair must find its weight,
otherwise the merge aborts. delta = (B @ A) * alpha / r, computed in fp32.

Usage: python merge_lora_qwen35.py --base MODEL_DIR --adapter ADAPTER_DIR --out OUT_DIR
"""
import argparse, json, os, shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
lora = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
pairs = {}
for k, v in lora.items():
    if ".lora_A." in k or ".lora_B." in k:
        stem, which = k.replace(".lora_A.weight", ""), "A"
        if ".lora_B." in k:
            stem, which = k.replace(".lora_B.weight", ""), "B"
        pairs.setdefault(stem, {})[which] = v
    else:
        raise SystemExit(f"unexpected adapter tensor {k}")


def target_name(stem):
    assert stem.startswith("base_model.model.model."), stem
    return "model.language_model." + stem[len("base_model.model.model."):] + ".weight"


targets = {target_name(s): p for s, p in pairs.items()}
index = json.load(open(os.path.join(args.base, "model.safetensors.index.json")))
missing = [t for t in targets if t not in index["weight_map"]]
if missing:
    raise SystemExit(f"{len(missing)} LoRA targets not in base checkpoint, e.g. {missing[:3]}")

os.makedirs(args.out, exist_ok=True)
for f in os.listdir(args.base):
    if not f.endswith(".safetensors"):
        src = os.path.join(args.base, f)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, f))
merged = 0
for shard in sorted(set(index["weight_map"].values())):
    tensors = {}
    with safe_open(os.path.join(args.base, shard), "pt") as f:
        for name in f.keys():
            w = f.get_tensor(name)
            if name in targets:
                p = targets[name]
                delta = (p["B"].float() @ p["A"].float()) * scale
                assert delta.shape == w.shape, (name, delta.shape, w.shape)
                w = (w.float() + delta).to(w.dtype)
                merged += 1
            tensors[name] = w
    save_file(tensors, os.path.join(args.out, shard), metadata={"format": "pt"})
assert merged == len(targets), (merged, len(targets))
print(f"merged {merged} LoRA pairs (scale {scale}) into {args.out}")
