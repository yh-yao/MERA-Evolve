"""Merging of multi-rank verl FSDP actor checkpoints in extract_sft_lora_adapter (CPU only)."""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from verl_code_rl.extract_sft_lora_adapter import _checkpoint_world_size, _load_full_state_dict  # noqa: E402

FULL = {
    "m.layers.0.q_proj.lora_A.default.weight": torch.randn(16, 40, dtype=torch.float32),
    "m.layers.0.q_proj.lora_B.default.weight": torch.randn(37, 16, dtype=torch.float32),  # uneven split
    "m.layers.0.mlp.down_proj.lora_A.default.weight": torch.randn(16, 24, dtype=torch.float32),
}


def _write(ckpt: Path, world_size: int, shards: list[dict]):
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 1, "world_size": world_size}))
    for rank, sd in enumerate(shards):
        torch.save(sd, ckpt / f"model_world_size_{world_size}_rank_{rank}.pt")


def _assert_same(merged: dict):
    assert merged.keys() == FULL.keys()
    for k, v in FULL.items():
        assert merged[k].dtype == torch.float32, "the stored dtype must be kept (no bf16 cast)"
        assert torch.equal(merged[k], v), k


def test_single_rank(tmp_path):
    _write(tmp_path, 1, [FULL])
    assert _checkpoint_world_size(tmp_path) == 1
    _assert_same(_load_full_state_dict(tmp_path))


def test_plain_shards_concatenate_along_dim0(tmp_path):
    _write(tmp_path, 3, [{k: torch.chunk(v, 3, dim=0)[r].clone() for k, v in FULL.items()} for r in range(3)])
    assert _checkpoint_world_size(tmp_path) == 3
    _assert_same(_load_full_state_dict(tmp_path))


def test_missing_rank_is_an_error(tmp_path):
    _write(tmp_path, 2, [{k: torch.chunk(v, 2, dim=0)[0].clone() for k, v in FULL.items()}])
    with pytest.raises(SystemExit, match="missing shards"):
        _load_full_state_dict(tmp_path)


def _dtensor_worker(rank: int, world_size: int, ckpt: str, port: str):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
    full = torch.load(Path(ckpt) / "full.pt")  # spawned workers re-import this module, so FULL differs there
    sd = {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in full.items()}
    torch.save(sd, Path(ckpt) / f"model_world_size_{world_size}_rank_{rank}.pt")
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs torch.distributed with gloo")
def test_dtensor_shards(tmp_path):
    import torch.multiprocessing as mp
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 1, "world_size": 2}))
    torch.save(FULL, tmp_path / "full.pt")
    mp.spawn(_dtensor_worker, args=(2, str(tmp_path), "29541"), nprocs=2, join=True)
    (tmp_path / "full.pt").unlink()
    _assert_same(_load_full_state_dict(tmp_path))
