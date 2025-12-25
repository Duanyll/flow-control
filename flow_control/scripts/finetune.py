from flow_control.training.accelerate_ddp import AccelerateDdpFinetuner
from flow_control.training.hsdp import HsdpTrainer
from flow_control.utils.loaders import load_config_file


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune a model based on the provided configuration."
    )
    parser.add_argument(
        "method",
        type=str,
        choices=["accelerate_ddp", "hsdp"],
        help="Finetuning method.",
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the finetuning configuration file."
    )
    args = parser.parse_args()
    config_path = args.config_path

    config = load_config_file(config_path)
    if args.method == "accelerate_ddp":
        finetuner = AccelerateDdpFinetuner(**config)
    elif args.method == "hsdp":
        finetuner = HsdpTrainer(**config)
    else:
        raise ValueError(f"Unknown finetuning method: {args.method}")
    finetuner.train()


if __name__ == "__main__":
    main()
