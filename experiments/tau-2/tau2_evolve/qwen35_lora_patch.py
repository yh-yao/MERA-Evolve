"""Reroute VERL separated-rollout weights into vLLM's LoRA ``base_layer``.

vLLM 0.25 wraps LoRA-targeted modules in ``*WithLoRA`` holders; the real weight
lives in a ``base_layer`` child. VERL's weight sync sends the un-prefixed name,
so the recursive loader fails with:

    ValueError: There is no module or parameter named
    'layers.0.linear_attn.in_proj_qkvz.weight' ... available:
    {'layers.0.linear_attn.in_proj_qkvz.base_layer.weight'}

VERL 0.8.0 and vLLM 0.25.1 were never released as a compatible pair (the repo
pins VERL 0.8 with vLLM 0.18 internally, but Qwen3.5 needs vLLM >= 0.25), so we
patch ``AutoWeightsLoader._load_module``: when the normal lookup raises and the
module is a LoRA holder, retry the same weights against its ``base_layer``
child. Modules without a ``base_layer`` re-raise unchanged.
"""

_PATCHED = False


def _make_patched(orig):
    def _load_module(self, base_prefix, module, weights):
        try:
            yield from orig(self, base_prefix, module, weights)
            return
        except ValueError:
            try:
                children = dict(module.named_children())
            except Exception:
                children = {}
            if "base_layer" not in children:
                raise
        new_prefix = f"{base_prefix}.base_layer" if base_prefix else "base_layer"
        yield from self._load_module(new_prefix, children["base_layer"], weights)

    _load_module._tau2_lora_base_layer_patch = True
    return _load_module


def install():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    try:
        from vllm.model_executor.models import utils as model_utils
    except Exception:
        return
    cls = getattr(model_utils, "AutoWeightsLoader", None)
    if cls is None or not hasattr(cls, "_load_module"):
        return
    if getattr(cls._load_module, "_tau2_lora_base_layer_patch", False):
        return
    cls._load_module = _make_patched(cls._load_module)
