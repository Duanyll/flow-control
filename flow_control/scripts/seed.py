import argparse

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

from flow_control.utils.config import load_config_file
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


def _run_transformer_seed(config: dict) -> None:
    """Generate a seed checkpoint for transformer models."""
    from flow_control.adapters import parse_model_adapter
    from flow_control.training.mixins import HsdpMixin

    model = parse_model_adapter(config["model"])
    model.load_transformer(device=torch.device("cpu"))
    mixin = HsdpMixin.model_construct()
    mixin.save_transformer_to_seed(model, config["seed_checkpoint_dir"])


def _run_vae_seed(config: dict) -> None:
    """Generate a seed checkpoint for VAE models."""
    from typing import Any

    from pydantic import TypeAdapter

    from flow_control.processors.components.vae import VAE
    from flow_control.training.vae.convert import convert_to_rgba

    vae_adapter: TypeAdapter[Any] = TypeAdapter(VAE)
    vae_loader = vae_adapter.validate_python(config["vae"])
    model = vae_loader.load_model(device=torch.device("cpu"), frozen=False)

    if config.get("do_convert_to_rgba", False):
        model = convert_to_rgba(model)

    seed_dir: str = config["seed_checkpoint_dir"]
    logger.info(f"Saving VAE DCP seed checkpoint to {seed_dir}...")
    model_sd, _ = get_state_dict(model, [], options=StateDictOptions(strict=False))
    dcp.save(model_sd, checkpoint_id=seed_dir, no_dist=True)
    logger.info(f"Saved VAE DCP seed checkpoint to {seed_dir}.")


def run(config_path: str) -> None:
    """Generate a seed checkpoint."""
    config = load_config_file(config_path)
    launch_type = config.get("launch", {}).get("type", "")
    if launch_type == "vae":
        _run_vae_seed(config)
    else:
        _run_transformer_seed(config)


def main():
    parser = argparse.ArgumentParser(description="Generate a seed checkpoint.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    run(args.config_path)


if __name__ == "__main__":
    main()
