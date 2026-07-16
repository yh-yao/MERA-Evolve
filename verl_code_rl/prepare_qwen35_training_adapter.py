"""Convert a served Qwen3.5 LoRA adapter back to VERL's training key layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _training_adapter_key(key: str) -> str:
    marker = ".model.model.layers."
    if marker in key:
        return key.replace(marker, ".model.model.language_model.layers.", 1)
    return key


def prepare_training_adapter(source: Path, output: Path) -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file

    weights = source / "adapter_model.safetensors"
    config = source / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError(f"invalid LoRA adapter directory: {source}")

    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.name != weights.name:
            shutil.copy2(item, output / item.name)

    with safe_open(weights, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        metadata = handle.metadata()
        converted = {
            _training_adapter_key(key): handle.get_tensor(key) for key in source_keys
        }

    remapped = sum(key != _training_adapter_key(key) for key in source_keys)
    save_file(converted, output / weights.name, metadata=metadata)
    return remapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    remapped = prepare_training_adapter(args.input, args.output)
    if remapped == 0:
        raise SystemExit(
            f"no Qwen3.5 adapter keys were remapped from {args.input}; "
            "refusing to continue with an incompatible training adapter"
        )
    print(
        f"[prepare_qwen35_training_adapter] remapped {remapped} keys into {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
