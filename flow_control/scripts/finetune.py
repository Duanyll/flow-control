from flow_control.training.finetuner import Fintuner
from flow_control.utils.loaders import load_config_file

def main(config_path: str):
    config = load_config_file(config_path)
    finetuner = Fintuner(**config)
    finetuner.train()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finetune a model based on the provided configuration.")
    parser.add_argument("config_path", type=str, help="Path to the finetuning configuration file.")
    args = parser.parse_args()

    main(args.config_path)