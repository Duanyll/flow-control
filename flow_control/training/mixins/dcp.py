import os
import re
import shutil

import torch.distributed.checkpoint as dcp
from pydantic import BaseModel

from flow_control.utils.logging import get_logger

from .hsdp import main_process_only

logger = get_logger(__name__)

AUTO_RESUME_ENV = "FLOW_CONTROL_AUTO_RESUME"
_STEP_DIR_RE = re.compile(r"^step_(\d+)$")


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

    def find_latest_checkpoint(self) -> str | None:
        """Return the newest ``step_XXXXXXX`` directory or None if none found.

        Only exact ``step_<digits>`` directories are considered; suffixed
        variants like ``step_XXXXX_final`` are skipped so we do not resume
        from the post-training artifact and continue past ``train_steps``.
        """
        if not os.path.isdir(self.checkpoint_root):
            return None
        best_step = -1
        best_path: str | None = None
        for name in os.listdir(self.checkpoint_root):
            match = _STEP_DIR_RE.match(name)
            if match is None:
                continue
            path = os.path.join(self.checkpoint_root, name)
            if not os.path.isfile(os.path.join(path, ".metadata")):
                continue
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_path = path
        return best_path

    def maybe_auto_resume(self, explicit_resume_dir: str | None) -> str | None:
        """Resolve the checkpoint dir to resume from.

        Priority: explicit ``resume_from_dir`` config > ``FLOW_CONTROL_AUTO_RESUME=1``
        env var with latest checkpoint in ``checkpoint_root`` > None.
        Returns the path that was resumed from (useful for logging).
        """
        if explicit_resume_dir is not None:
            self.load_dcp_checkpoint(explicit_resume_dir)
            return explicit_resume_dir

        if os.environ.get(AUTO_RESUME_ENV) != "1":
            return None

        latest = self.find_latest_checkpoint()
        if latest is None:
            logger.info(
                "%s=1 but no checkpoints found under %s; starting fresh.",
                AUTO_RESUME_ENV,
                self.checkpoint_root,
            )
            return None

        logger.info("%s=1; auto-resuming from %s", AUTO_RESUME_ENV, latest)
        self.load_dcp_checkpoint(latest)
        return latest
