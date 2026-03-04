import argparse
import os
import tomllib
from typing import Literal

from pydantic import BaseModel
from rich import print


class LaunchConfig(BaseModel):
    type: Literal["sft", "inference"]
    devices: int | list[int]
    env: dict[str, str] = {}


def _load_launch_config(config_path: str) -> tuple[LaunchConfig, dict]:
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)
    if "launch" not in config_data:
        raise ValueError("Launch configuration section is missing in the config file.")
    return LaunchConfig(**config_data["launch"]), config_data


def _run_child(launch_config: LaunchConfig, config_data: dict) -> None:
    """Run the child process (invoked by torchrun)."""
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


def run(config_path: str) -> None:
    """Launch training as the parent process (sets up env, execs torchrun)."""
    launch_config, _ = _load_launch_config(config_path)

    num_processes = (
        launch_config.devices
        if isinstance(launch_config.devices, int)
        else len(launch_config.devices)
    )
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(num_processes),
        "-m",
        "flow_control.scripts.launch",
        config_path,
        "--type",
        launch_config.type,
    ]

    envs = launch_config.env
    if isinstance(launch_config.devices, list):
        envs["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in launch_config.devices)
    if "OMP_NUM_THREADS" not in envs:
        omp_threads = max(1, (os.cpu_count() or 0) // num_processes)
        envs["OMP_NUM_THREADS"] = str(omp_threads)
    for k, v in envs.items():
        print(f"[blue]Setting environment variable:[/blue] {k}={v}")
        os.environ[k] = v

    print(f"[blue]Executing command:[/blue] {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


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

    if args.type:
        # This is child process (invoked by torchrun)
        launch_config, config_data = _load_launch_config(args.config_path)
        _run_child(launch_config, config_data)
    else:
        # This is parent process
        run(args.config_path)


if __name__ == "__main__":
    main()
