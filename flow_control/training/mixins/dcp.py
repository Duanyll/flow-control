import os
import shutil

import torch.distributed.checkpoint as dcp
from pydantic import BaseModel

from flow_control.utils.logging import get_logger

from .hsdp import main_process_only

logger = get_logger(__name__)


class DcpMixin(BaseModel):
    """Mixin providing DCP checkpoint load/save.

    Leaf classes must implement ``state_dict()`` and ``load_state_dict()``
    to satisfy the ``Stateful`` protocol used by DCP internally.
    """

    def load_dcp_checkpoint(self, checkpoint_path: str):
        state_dict = {"app": self}
        # dcp.load will call self.state_dict() and self.load_state_dict() internally
        dcp.load(
            state_dict,
            checkpoint_id=checkpoint_path,
            planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
        )
        logger.info(f"Resumed DCP checkpoint from {checkpoint_path}.")

    def save_dcp_checkpoint(self, checkpoint_path: str):
        state_dict = {"app": self}
        dcp.save(state_dict, checkpoint_id=checkpoint_path)
        logger.info(f"Saved DCP checkpoint to {checkpoint_path}.")


class CheckpointingMixin(DcpMixin):
    checkpoint_root: str
    max_checkpoints: int = 5

    @main_process_only
    def rotate_checkpoints_maybe(self):
        if self.max_checkpoints <= 0:
            return
        checkpoint_dirs = []
        for name in os.listdir(self.checkpoint_root):
            if name.startswith("step_"):
                checkpoint_dirs.append(name)
        if len(checkpoint_dirs) <= self.max_checkpoints:
            return
        checkpoint_dirs.sort()
        num_to_remove = len(checkpoint_dirs) - self.max_checkpoints
        for i in range(num_to_remove):
            dir_to_remove = os.path.join(self.checkpoint_root, checkpoint_dirs[i])
            shutil.rmtree(dir_to_remove)
            logger.info(f"Removed old checkpoint: {dir_to_remove}")

    def get_checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.checkpoint_root, f"step_{step:07d}")

    def save(self, step: int):
        checkpoint_dir = self.get_checkpoint_dir(step)
        self.save_dcp_checkpoint(checkpoint_dir)
        self.rotate_checkpoints_maybe()
