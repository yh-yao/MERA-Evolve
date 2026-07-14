"""TAU-2 NIXL checkpoint backend fixes for separated VERL rollout."""

from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.nixl_checkpoint_engine import NIXLCheckpointEngine


@CheckpointEngineRegistry.register("nixl_tau2")
class Tau2NIXLCheckpointEngine(NIXLCheckpointEngine):
    """Fix VERL 0.8's NIXL topology when there is one rollout worker."""

    @classmethod
    def build_topology(cls, trainer_world_size, rollout_world_size, metadata):
        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "prev_agent_metadata": [None] * trainer_world_size,
            "next_agent_metadata": [metadata[-rollout_world_size]]
            + [None] * (trainer_world_size - 1),
        }
        rollout_metadata = metadata[-rollout_world_size:]
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "prev_agent_metadata": [metadata[0]] + rollout_metadata[:-1],
            "next_agent_metadata": rollout_metadata[1:] + [None],
        }
        return trainer_kwargs, rollout_kwargs
