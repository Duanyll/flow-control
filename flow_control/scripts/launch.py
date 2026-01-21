import argparse
import os
import tomllib
from typing import Literal

from pydantic import BaseModel


class LaunchConfig(BaseModel):
    type: Literal["sft", "inference", "accelerate"]
    accelerate_config: str | None = None
    num_processes: int
    omp_num_threads: int | None = None


def main():
    parser = argparse.ArgumentParser(
        description="Launch training with specified configuration."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the training configuration file."
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Do not manually set this argument.",
    )
    args = parser.parse_args()

    with open(args.config_path, "rb") as f:
        config_data = tomllib.load(f)
    if "launch" not in config_data:
        raise ValueError("Launch configuration section is missing in the config file.")
    launch_config = LaunchConfig(**config_data["launch"])

    if args.type:
        # This is child process
        if launch_config.type == "sft":
            from flow_control.training.sft import HsdpSftTrainer

            del config_data["launch"]
            trainer = HsdpSftTrainer(**config_data)
            trainer.train()
        elif launch_config.type == "inference":
            from flow_control.training.inference import HsdpInference

            del config_data["launch"]
            trainer = HsdpInference(**config_data)
            trainer.inference()
        elif launch_config.type == "accelerate":
            from flow_control.training.accelerate_ddp import AccelerateDdpFinetuner

            del config_data["launch"]
            trainer = AccelerateDdpFinetuner(**config_data)
            trainer.train()
        else:
            raise ValueError(f"Unknown launch type: {launch_config.type}")
    else:
        # This is parent process
        if launch_config.type in ["sft", "inference"]:
            cmd = [
                "torchrun",
                "--nproc_per_node",
                str(launch_config.num_processes),
                "-m",
                "flow_control.scripts.launch",
                args.config_path,
                "--type",
                launch_config.type,
            ]
        else:
            cmd = [
                "accelerate",
                "launch",
            ]
            if launch_config.accelerate_config:
                cmd += ["--config_file", launch_config.accelerate_config]
            cmd += [
                "--num_processes",
                str(launch_config.num_processes),
                "-m",
                "flow_control.scripts.launch",
                args.config_path,
                "--type",
                "accelerate",
            ]
        if launch_config.omp_num_threads is None:
            launch_config.omp_num_threads = (
                os.cpu_count() or 0
            ) // launch_config.num_processes
        if launch_config.omp_num_threads > 0:
            print(f"Setting OMP_NUM_THREADS to {launch_config.omp_num_threads}")
            os.environ["OMP_NUM_THREADS"] = str(launch_config.omp_num_threads)
        print(f"Executing command: {' '.join(cmd)}")
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
