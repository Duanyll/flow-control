import argparse

import torch

from flow_control.adapters import parse_model_adapter
from flow_control.training.mixins import HsdpMixin
from flow_control.utils.common import load_config_file


def run(config_path: str) -> None:
    """Generate a seed checkpoint."""
    config = load_config_file(config_path)
    model = parse_model_adapter(config["model"])
    model.load_transformer(device=torch.device("cpu"))
    # Use HsdpMixin only for save_transformer_to_seed (no distributed init needed)
    mixin = HsdpMixin.model_construct()
    mixin.save_transformer_to_seed(model, config["seed_checkpoint_dir"])


def main():
    parser = argparse.ArgumentParser(description="Generate a seed checkpoint.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    run(args.config_path)


if __name__ == "__main__":
    main()
