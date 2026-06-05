import os
import re
import shutil
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from pydantic import BaseModel

from flow_control.utils import device as devutil
from flow_control.utils.logging import get_logger

from .hsdp import main_process_only

logger = get_logger(__name__)

AUTO_CHECKPOINT_ROOT_SENTINEL = "$auto"
# Matches both archival ``step_XXXXXXX`` and rolling ``rolling_step_XXXXXXX`` dirs,
# but not suffixed variants like ``step_XXXXX_final``.
_STEP_DIR_RE = re.compile(r"^(?:rolling_)?step_(\d+)$")
_ARCHIVAL_PREFIX = "step_"
_ROLLING_PREFIX = "rolling_step_"


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
    """Two-tier checkpoint retention.

    All checkpoints are written at *clean boundaries* (where resume is
    consistent). There is no preempt-time saving; instead a time-gated
    "rolling" checkpoint bounds how much work a hard kill can lose. Trainers
    just call :meth:`save_maybe` at every clean boundary.

    - **Archival** (``step_XXXXXXX``): sparse cadence (:pyattr:`checkpoint_interval`),
      capped at the newest :pyattr:`max_checkpoints`. Permanent reference points.
    - **Rolling** (``rolling_step_XXXXXXX``): exactly one kept, refreshed at most
      every :pyattr:`rolling_checkpoint_interval_seconds`.
    """

    checkpoint_root: str
    auto_resume: bool = False
    """When ``resume_from_dir`` is unset, resume from the newest complete
    checkpoint under ``checkpoint_root``."""
    checkpoint_interval: int = 5
    """Archival checkpoint cadence. The unit is trainer-defined: epochs for the
    RL trainers (GRPO/NFT), optimizer steps for SFT/VAE. ``0`` disables archival
    cadence saves (the final checkpoint is still written)."""
    max_checkpoints: int = 5
    """How many archival ``step_*`` checkpoints to keep. The rolling checkpoint
    is tracked separately and is always exactly one."""
    rolling_checkpoint_interval_seconds: float = 1800.0
    """Minimum wall-clock seconds between rolling checkpoint saves. ``0`` disables
    rolling checkpoints entirely. The clock is per-process (resets on resume), so
    a hard kill loses at most this interval plus one boundary's worth of work."""

    _last_rolling_time: float = 0.0

    def ensure_checkpoint_root_resolved(self) -> None:
        if self.checkpoint_root != AUTO_CHECKPOINT_ROOT_SENTINEL:
            return
        resolver = getattr(self, "resolve_run_context", None)
        if callable(resolver):
            resolver()
        if self.checkpoint_root == AUTO_CHECKPOINT_ROOT_SENTINEL:
            raise ValueError(
                "checkpoint_root='$auto' requires a trainer with LoggingMixin "
                "so the run context can be resolved before checkpoint access."
            )

    # ------------------------------- Dir naming --------------------------------- #

    def get_checkpoint_dir(self, step: int) -> str:
        self.ensure_checkpoint_root_resolved()
        return os.path.join(self.checkpoint_root, f"{_ARCHIVAL_PREFIX}{step:07d}")

    def get_rolling_dir(self, step: int) -> str:
        self.ensure_checkpoint_root_resolved()
        return os.path.join(self.checkpoint_root, f"{_ROLLING_PREFIX}{step:07d}")

    # --------------------------------- Saving ----------------------------------- #

    def save_maybe(
        self,
        step: int,
        progress: int | None = None,
        force_archival: bool = False,
    ) -> None:
        """Call at every clean boundary; decides both checkpoint tiers.

        Args:
            step: optimizer step, used to name the checkpoint directory.
            progress: value compared against :pyattr:`checkpoint_interval` to
                decide archival saves (pass the epoch for RL trainers). Defaults
                to ``step`` (SFT/VAE, where the cadence is in steps).
            force_archival: force an archival save regardless of cadence (e.g.
                the final boundary).
        """
        prog = step if progress is None else progress
        if self.checkpoint_interval > 0 and (
            force_archival or prog % self.checkpoint_interval == 0
        ):
            self.save_archival(step)
        self.maybe_save_rolling(step)

    def save_archival(self, step: int) -> None:
        self.save_dcp_checkpoint(self.get_checkpoint_dir(step))
        self.rotate_archival_maybe()

    def save_rolling(self, step: int) -> None:
        # Write the new rolling dir first, then drop the previous one(s), so an
        # interrupted write never destroys the last good rolling checkpoint.
        self.save_dcp_checkpoint(self.get_rolling_dir(step))
        self.rotate_rolling_maybe(keep_step=step)

    def maybe_save_rolling(self, step: int) -> None:
        """Save a rolling checkpoint if the interval has elapsed.

        The decision is reached collectively (all ranks act on rank 0's choice)
        so the cross-rank ``dcp.save`` collective stays aligned.
        """
        if self.rolling_checkpoint_interval_seconds <= 0:
            return
        now = time.monotonic()
        if self._last_rolling_time == 0.0:
            # First boundary of this process: establish the baseline, don't save.
            self._last_rolling_time = now
            return
        should_local = (
            now - self._last_rolling_time
        ) >= self.rolling_checkpoint_interval_seconds
        if not self._agree_rolling(should_local):
            return
        self.save_rolling(step)
        self._last_rolling_time = time.monotonic()

    def _agree_rolling(self, should_local: bool) -> bool:
        """Broadcast rank 0's save decision to all ranks via all_reduce(MAX)."""
        is_main: bool = getattr(self, "is_main_process", True)
        local = 1 if (should_local and is_main) else 0
        if not dist.is_available() or not dist.is_initialized():
            return bool(should_local)
        device = getattr(self, "device", None)
        if not isinstance(device, torch.device):
            device = devutil.default_device()
        flag = torch.tensor([local], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    # -------------------------------- Rotation ---------------------------------- #

    def _checkpoint_dirs(self, prefix: str) -> list[str]:
        self.ensure_checkpoint_root_resolved()
        if not os.path.isdir(self.checkpoint_root):
            return []
        return [
            name
            for name in os.listdir(self.checkpoint_root)
            if _STEP_DIR_RE.match(name) and name.startswith(prefix)
        ]

    @main_process_only
    def rotate_archival_maybe(self):
        if self.max_checkpoints <= 0:
            return
        # ``rolling_step_*`` does not start with ``step_``, so it is excluded.
        checkpoint_dirs = self._checkpoint_dirs(_ARCHIVAL_PREFIX)
        if len(checkpoint_dirs) <= self.max_checkpoints:
            return
        checkpoint_dirs.sort()
        num_to_remove = len(checkpoint_dirs) - self.max_checkpoints
        for i in range(num_to_remove):
            dir_to_remove = os.path.join(self.checkpoint_root, checkpoint_dirs[i])
            shutil.rmtree(dir_to_remove)
            logger.info(f"Removed old checkpoint: {dir_to_remove}")

    @main_process_only
    def rotate_rolling_maybe(self, keep_step: int | None = None):
        """Keep only the newest rolling checkpoint, dropping the rest."""
        rolling_dirs = self._checkpoint_dirs(_ROLLING_PREFIX)
        if len(rolling_dirs) <= 1:
            return
        rolling_dirs.sort()
        keep_name = (
            os.path.basename(self.get_rolling_dir(keep_step))
            if keep_step is not None
            else rolling_dirs[-1]
        )
        for name in rolling_dirs:
            if name == keep_name:
                continue
            dir_to_remove = os.path.join(self.checkpoint_root, name)
            shutil.rmtree(dir_to_remove)
            logger.info(f"Removed old rolling checkpoint: {dir_to_remove}")

    # --------------------------------- Resume ----------------------------------- #

    def find_latest_checkpoint(self) -> str | None:
        """Return the newest valid checkpoint dir (archival or rolling) or None.

        Only dirs matching ``[rolling_]step_<digits>`` with a written
        ``.metadata`` are considered, so partially-written checkpoints and
        suffixed artifacts like ``step_XXXXX_final`` are skipped.
        """
        self.ensure_checkpoint_root_resolved()
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

        Priority: explicit ``resume_from_dir`` config > ``auto_resume=true`` with
        latest checkpoint in ``checkpoint_root`` > None.
        Returns the path that was resumed from (useful for logging).
        """
        self.ensure_checkpoint_root_resolved()
        if explicit_resume_dir is not None:
            self.load_dcp_checkpoint(explicit_resume_dir)
            return explicit_resume_dir

        if not self.auto_resume:
            return None

        latest = self.find_latest_checkpoint()
        if latest is None:
            logger.info(
                "auto_resume=true but no checkpoints found under %s; starting fresh.",
                self.checkpoint_root,
            )
            return None

        logger.info("auto_resume=true; resuming from %s", latest)
        self.load_dcp_checkpoint(latest)
        return latest


if __name__ == "__main__":
    import tempfile

    from rich import print as rprint

    def _make_ckpt(root: str, name: str, valid: bool = True) -> None:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        if valid:
            open(os.path.join(d, ".metadata"), "w").close()

    with tempfile.TemporaryDirectory() as root:
        mixin = CheckpointingMixin(
            checkpoint_root=root,
            checkpoint_interval=2,
            max_checkpoints=3,
            rolling_checkpoint_interval_seconds=100.0,
        )

        # Archival rotation keeps newest max_checkpoints and ignores rolling dirs.
        for s in (10, 20, 30, 40, 50):
            _make_ckpt(root, f"step_{s:07d}")
        _make_ckpt(root, "rolling_step_0000045")
        _make_ckpt(root, "step_0000099_final")  # must be ignored everywhere
        mixin.rotate_archival_maybe()
        remaining = sorted(
            n for n in os.listdir(root) if n.startswith("step_") and "final" not in n
        )
        assert remaining == [
            "step_0000030",
            "step_0000040",
            "step_0000050",
        ], remaining
        assert os.path.isdir(os.path.join(root, "rolling_step_0000045"))

        # Latest picks the global max step across tiers (50 > 45).
        latest = mixin.find_latest_checkpoint()
        assert latest is not None and latest.endswith("step_0000050"), latest

        # Rolling newer than any archival wins.
        _make_ckpt(root, "rolling_step_0000060")
        latest = mixin.find_latest_checkpoint()
        assert latest is not None and latest.endswith("rolling_step_0000060"), latest

        # Rolling rotation keeps only the newest one.
        mixin.rotate_rolling_maybe(keep_step=60)
        rolling = sorted(n for n in os.listdir(root) if n.startswith("rolling_step_"))
        assert rolling == ["rolling_step_0000060"], rolling

        # Partial (no .metadata) rolling dir is ignored by resume.
        _make_ckpt(root, "rolling_step_0000070", valid=False)
        latest = mixin.find_latest_checkpoint()
        assert latest is not None and latest.endswith("rolling_step_0000060"), latest

        # Rolling timer gating (single-process: dist not initialized).
        mixin._last_rolling_time = 0.0
        mixin.maybe_save_rolling(80)  # first call only sets baseline
        assert not os.path.isdir(os.path.join(root, "rolling_step_0000080"))
        mixin._last_rolling_time = time.monotonic() - 200.0  # force elapsed
        mixin.maybe_save_rolling(90)
        # save_rolling -> save_dcp_checkpoint writes real DCP files including .metadata
        assert os.path.isdir(os.path.join(root, "rolling_step_0000090"))
        assert sorted(n for n in os.listdir(root) if n.startswith("rolling_step_")) == [
            "rolling_step_0000090"
        ], "rolling rotation should keep only newest"

    rprint("[green]CheckpointingMixin retention smoke test passed.[/green]")
