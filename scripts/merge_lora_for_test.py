"""One-off: merge a verl-trained LoRA adapter into its base model and save a
full HF checkpoint, to test whether vLLM's runtime LoRA application (vs the
in-training hot-swap sync) explains the internal-vs-external eval gap."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    print(f"wrote merged model to {args.output}")


if __name__ == "__main__":
    main()
