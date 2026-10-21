"""Export DCP training checkpoints to HuggingFace save_pretrained format."""

import os
import re
from pathlib import Path

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


def _checkpoint_root(trainer) -> str:
    """The trainer's checkpoint root, with ``$auto`` resolved to its run's directory:
    the configured ``run_id``, else the experiment's newest run with a checkpoint."""
    from flow_control.training.mixins.logging import (
        AUTO_CHECKPOINT_ROOT_SENTINEL,
        _find_resumable_run_id,
    )

    checkpoint_root = getattr(trainer, "checkpoint_root", None)
    if not checkpoint_root:
        raise ValueError(
            "No --checkpoint-dir provided and no checkpoint_root in config."
        )
    if checkpoint_root != AUTO_CHECKPOINT_ROOT_SENTINEL:
        return checkpoint_root
    experiment_dir = Path(trainer.runs_root) / trainer.experiment_name
    run_id = trainer.run_id or _find_resumable_run_id(experiment_dir)
    if run_id is None:
        raise FileNotFoundError(
            f"checkpoint_root='$auto': no run with a checkpoint under {experiment_dir}. "
            "Set run_id or pass --checkpoint-dir."
        )
    return str(experiment_dir / run_id / "checkpoints")


def _find_latest_checkpoint(checkpoint_root: str) -> str:
    """The newest archival, rolling or final checkpoint directory in checkpoint_root."""
    pattern = re.compile(r"^(?:rolling_)?step_(\d+)(_final)?$")
    best_step = -1
    best_name = ""
    for name in os.listdir(checkpoint_root):
        m = pattern.match(name)
        if m and os.path.isfile(os.path.join(checkpoint_root, name, ".metadata")):
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_name = name
    if best_step < 0:
        raise FileNotFoundError(
            f"No valid DCP checkpoint found in {checkpoint_root}. "
            "Expected [rolling_]step_XXXXXXX directories with a .metadata file."
        )
    return os.path.join(checkpoint_root, best_name)


def run(
    config: dict,
    output_dir: str,
    checkpoint_dir: str | None = None,
) -> None:
    """Export a DCP training checkpoint to HuggingFace format.

    Resolves the checkpoint generically, then delegates the format-specific
    export to the trainer's ``export_checkpoint(checkpoint_dir, output_dir)``
    method (the base raises for trainers that don't support export). Plugin
    trainers are already registered (cli loads ``imports`` before dispatching).
    """
    from flow_control.training import import_builtin_trainers
    from flow_control.training.mixins import trainer_registry

    import_builtin_trainers()
    launch_type = config.get("launch", {}).get("type", "")
    trainer_cls = trainer_registry.get(launch_type)
    if trainer_cls is None:
        raise ValueError(
            f"Unknown trainer type {launch_type!r}. Registered: "
            f"{sorted(trainer_registry.members())}. If it is a plugin trainer, add "
            "its module to the config's `imports`."
        )
    trainer = trainer_cls(**config)

    if checkpoint_dir is None:
        checkpoint_dir = _find_latest_checkpoint(_checkpoint_root(trainer))
        logger.info(f"Auto-selected latest checkpoint: {checkpoint_dir}")

    if not os.path.isfile(os.path.join(checkpoint_dir, ".metadata")):
        raise FileNotFoundError(
            f"{checkpoint_dir} is not a valid DCP checkpoint (missing .metadata)."
        )

    trainer.export_checkpoint(checkpoint_dir, output_dir)
