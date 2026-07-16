"""Isolate Qwen3.5's training and rollout kernel runtimes.

Qwen3.5 training needs FLA with Triton 3.3 on Hopper, while vLLM 0.18 needs
the Triton 3.6 bundled with PyTorch.  Separated VERL runs use different Ray
processes for those roles, so inject the Triton overlay only into training
workers and leave rollout servers on the environment's default Triton.
"""

import importlib.util
import os


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
    from verl.experimental.separation import utils as separation_utils
    from verl.single_controller.ray import RayWorkerGroup

    _create_role_worker_mapping = separation_utils.create_role_worker_mapping

    class Qwen35TrainingRayWorkerGroup(RayWorkerGroup):
        """Give actor/reference workers the FLA-compatible Triton runtime."""

        def __init__(self, *args, **kwargs):
            overlay = os.environ.get("QWEN35_TRAIN_TRITON_OVERLAY")
            if not overlay or not os.path.isfile(os.path.join(overlay, "triton", "__init__.py")):
                raise RuntimeError(
                    "QWEN35_TRAIN_TRITON_OVERLAY must point to a Triton 3.3 overlay"
                )
            worker_env = dict(kwargs.pop("worker_env", {}))
            worker_env["PYTHONPATH"] = os.pathsep.join(
                part for part in (overlay, os.environ.get("PYTHONPATH", "")) if part
            )
            # TileLang 0.1.12 segfaults in cuModuleLoadData on this CUDA 12.9
            # driver; Triton 3.3 is the validated FLA backend instead.
            worker_env["FLA_TILELANG"] = "0"
            kwargs["worker_env"] = worker_env
            super().__init__(*args, **kwargs)

    def _create_qwen35_role_worker_mapping(config):
        mapping, _ = _create_role_worker_mapping(config)
        return mapping, Qwen35TrainingRayWorkerGroup

    separation_utils.create_role_worker_mapping = _create_qwen35_role_worker_mapping
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
