import argparse

from flow_control.training.hsdp import HsdpTrainer
from flow_control.utils.hf_model import load_config_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate a seed checkpoint for HsdpTrainer."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the finetuning configuration file."
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = load_config_file(config_path)
    finetuner = HsdpTrainer(**config)
    finetuner.generate_seed_checkpoint()


if __name__ == "__main__":
    main()
