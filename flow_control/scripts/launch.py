import argparse
import os
import tomllib
from typing import Literal

from pydantic import BaseModel
from rich import print


class LaunchConfig(BaseModel):
    type: Literal["sft", "inference"]
    num_processes: int
    omp_num_threads: int | None = None
    nccl_p2p_level: str | None = None


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

            trainer = HsdpSftTrainer(**config_data)
            trainer.run()
        elif launch_config.type == "inference":
            from flow_control.training.inference import HsdpInference

            trainer = HsdpInference(**config_data)
            trainer.run()
        else:
            raise ValueError(f"Unknown launch type: {launch_config.type}")
    else:
        # This is parent process
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

        if launch_config.omp_num_threads is None:
            launch_config.omp_num_threads = (
                os.cpu_count() or 0
            ) // launch_config.num_processes
        if launch_config.omp_num_threads > 0:
            print(
                f"[green]Setting OMP_NUM_THREADS: [/green]{launch_config.omp_num_threads}"
            )
            os.environ["OMP_NUM_THREADS"] = str(launch_config.omp_num_threads)

        if launch_config.nccl_p2p_level:
            print(
                f"[green]Setting NCCL_P2P_LEVEL: [/green]{launch_config.nccl_p2p_level}"
            )
            os.environ["NCCL_P2P_LEVEL"] = launch_config.nccl_p2p_level

        print(f"[blue]Executing command:[/blue] {' '.join(cmd)}")
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
