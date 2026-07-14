"""Select Transformers' correct Qwen3.5 torch GDN fallback for GRPO.

FLA rejects Triton >=3.4 backward on Hopper because of known incorrect
gradients, while vLLM 0.18 requires the newer Triton bundled with PyTorch.
VERL colocates actor and rollout imports, so GRPO must keep new Triton for
vLLM and disable only Transformers' optional FLA fast path.
"""

import importlib.util


# NIXL 1.2 includes an old optional expert-parallel module. vLLM 0.18
# detects it by module presence, then fails because the APIs do not match.
# VERL's NIXL checkpoint engine uses nixl._api and does not need NIXL-EP.
_find_spec = importlib.util.find_spec


def _find_spec_without_nixl_ep(name, package=None):
    if name == "nixl_ep" or name.startswith("nixl_ep."):
        return None
    return _find_spec(name, package)


importlib.util.find_spec = _find_spec_without_nixl_ep

try:
    from transformers.utils import import_utils

    import_utils.is_flash_linear_attention_available = lambda: False
except ImportError:
    pass


def _index_first_axis(values, indices):
    return values[indices]


def _rearrange(values, pattern, **axes_lengths):
    from einops import rearrange

    return rearrange(values, pattern, **axes_lengths)


def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    import torch
    import torch.nn.functional as F

    active_mask = attention_mask if unused_mask is None else attention_mask + unused_mask
    sequence_lengths = active_mask.sum(dim=-1, dtype=torch.int32)
    used_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(active_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(sequence_lengths.max().item())
    flattened = hidden_states.flatten(0, 1)
    return flattened[indices], indices, cu_seqlens, max_seqlen, used_lengths


def _pad_input(hidden_states, indices, batch_size, sequence_length):
    import torch

    output = torch.zeros(
        batch_size * sequence_length,
        *hidden_states.shape[1:],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output[indices] = hidden_states
    return output.reshape(batch_size, sequence_length, *hidden_states.shape[1:])


try:
    from verl.utils import attention_utils

    attention_utils._get_attention_functions = lambda: (
        _index_first_axis,
        _pad_input,
        _rearrange,
        _unpad_input,
    )
except ImportError:
    pass
