"""Export DCP training checkpoints to HuggingFace save_pretrained format."""

import argparse
import os
import re
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

from flow_control.utils.config import load_config_file
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


def _find_latest_checkpoint(checkpoint_root: str) -> str:
    """Find the latest step_XXXXXXX directory in checkpoint_root."""
    pattern = re.compile(r"^step_(\d+)(_final)?$")
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
            "Expected directories named step_XXXXXXX with a .metadata file."
        )
    return os.path.join(checkpoint_root, best_name)


def _export_vae(config: dict, checkpoint_dir: str, output_dir: str) -> None:
    """Export a VAE DCP checkpoint to HuggingFace format."""
    from pydantic import TypeAdapter

    from flow_control.processors.components.vae import VAE
    from flow_control.training.vae.convert import convert_to_rgba

    vae_adapter: TypeAdapter[Any] = TypeAdapter(VAE)
    vae_loader = vae_adapter.validate_python(config["vae"])
    model = vae_loader.load_model(device=torch.device("cpu"), frozen=False)

    if config.get("do_convert_to_rgba", False):
        model = convert_to_rgba(model)

    logger.info(f"Loading VAE weights from {checkpoint_dir}...")
    model_sd, _ = get_state_dict(model, [], options=StateDictOptions(strict=False))
    state: dict[str, Any] = {"app": {"vae": model_sd}}
    dcp.load(
        state,
        checkpoint_id=checkpoint_dir,
        no_dist=True,
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
    )
    model.load_state_dict(state["app"]["vae"])
    logger.info(f"Loaded VAE weights from {checkpoint_dir}.")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"Exported VAE to {output_dir}.")


def run(config_path: str, output_dir: str, checkpoint_dir: str | None = None) -> None:
    """Export a DCP training checkpoint to HuggingFace format."""
    config = load_config_file(config_path)
    launch_type = config.get("launch", {}).get("type", "")

    if launch_type != "vae":
        raise ValueError(
            f"Export currently only supports launch.type='vae', got '{launch_type}'."
        )

    if checkpoint_dir is None:
        checkpoint_root = config.get("checkpoint_root")
        if not checkpoint_root:
            raise ValueError(
                "No --checkpoint-dir provided and no checkpoint_root in config."
            )
        checkpoint_dir = _find_latest_checkpoint(checkpoint_root)
        logger.info(f"Auto-selected latest checkpoint: {checkpoint_dir}")

    if not os.path.isfile(os.path.join(checkpoint_dir, ".metadata")):
        raise FileNotFoundError(
            f"{checkpoint_dir} is not a valid DCP checkpoint (missing .metadata)."
        )

    _export_vae(config, checkpoint_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Export DCP training checkpoints to HuggingFace format."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the training config file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="DCP checkpoint directory. Defaults to latest step in checkpoint_root.",
    )
    args = parser.parse_args()
    run(args.config_path, args.output_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()
