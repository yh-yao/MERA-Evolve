"""Compatibility shims for serving dense Qwen3.5 LoRA adapters in SGLang."""

try:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    if not hasattr(Qwen3_5Config, "vocab_size"):
        Qwen3_5Config.vocab_size = property(
            lambda self: self.text_config.vocab_size,
            lambda self, value: setattr(self.text_config, "vocab_size", value),
        )
except ImportError:
    pass
