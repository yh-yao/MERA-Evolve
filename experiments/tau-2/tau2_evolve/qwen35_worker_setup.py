"""Deferred Qwen3.5 compatibility setup for VERL Ray training workers."""

import os


def install_shm_weight_transfer_patch():
    """Route rollout weight sync through host shared memory instead of CUDA IPC.

    VERL treats every CUDA device as IPC-capable, but the rollout sender and
    the vLLM worker can sit on different GPUs, and opening a CUDA IPC handle
    on another device needs peer access. Consumer cards such as the RTX 5090
    have none ("peer access is not supported between these two devices").
    VERL's shared-memory path (written for NPUs) works everywhere; enable it
    with MERA_WEIGHT_TRANSFER_SHM=1. Must run before vllm_rollout is imported,
    which binds is_support_ipc at import time.
    """
    if os.environ.get("MERA_WEIGHT_TRANSFER_SHM") != "1":
        return
    from verl.utils import device

    device.is_support_ipc = lambda: False
    import sys

    module = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_rollout")
    if module is not None:
        module.is_support_ipc = device.is_support_ipc


def install_qwen35_worker_patches():
    from sitecustomize import install_attention_patch

    install_shm_weight_transfer_patch()
    install_attention_patch()
