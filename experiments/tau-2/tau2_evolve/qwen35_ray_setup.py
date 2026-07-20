"""Serializable Ray worker-group integration for Qwen3.5 training."""

import os

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

    def _create_worker(self, *args, **kwargs):
        ray_cls_with_init = kwargs["ray_cls_with_init"]
        update_options = ray_cls_with_init.update_options

        def update_options_with_setup_hook(options):
            runtime_env = options.get("runtime_env")
            if runtime_env is not None:
                options = dict(options)
                runtime_env = dict(runtime_env)
                runtime_env["worker_process_setup_hook"] = (
                    "tau2_evolve.qwen35_worker_setup.install_qwen35_worker_patches"
                )
                options["runtime_env"] = runtime_env
            update_options(options)

        ray_cls_with_init.update_options = update_options_with_setup_hook
        try:
            return super()._create_worker(*args, **kwargs)
        finally:
            ray_cls_with_init.update_options = update_options


def create_qwen35_role_worker_mapping(config):
    mapping, _ = _create_role_worker_mapping(config)
    return mapping, Qwen35TrainingRayWorkerGroup
