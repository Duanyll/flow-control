import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rich import print


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sft", "grpo", "inference"]
    devices: int | list[int]
    generate_dcp_seed: bool = False
    preprocess_config: str | None = None
    env: dict[str, str] = {}


def _load_launch_config(config_path: str) -> tuple[LaunchConfig, dict]:
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)
    if "launch" not in config_data:
        raise ValueError("Launch configuration section is missing in the config file.")
    launch_config = LaunchConfig(**config_data["launch"])
    del config_data["launch"]
    return launch_config, config_data


def _run_child(launch_config: LaunchConfig, config_data: dict) -> None:
    """Run the child process (invoked by torchrun)."""
    if launch_config.type == "sft":
        from flow_control.training.sft import HsdpSftTrainer

        trainer = HsdpSftTrainer(**config_data)
        trainer.run()
    elif launch_config.type == "grpo":
        from flow_control.training.grpo import HsdpGrpoTrainer

        trainer = HsdpGrpoTrainer(**config_data)
        trainer.run()
    elif launch_config.type == "inference":
        from flow_control.training.inference import HsdpInference

        trainer = HsdpInference(**config_data)
        trainer.run()
    else:
        raise ValueError(f"Unknown launch type: {launch_config.type}")


def is_on_ram_disk(path):
    path = Path(path).resolve()

    for parent in [path] + list(path.parents):
        if parent.exists():
            check_path = str(parent)
            break
    else:
        return False

    try:
        target_dev = os.stat(check_path).st_dev

        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue

                mount_point = parts[1]
                fs_type = parts[2]

                if (
                    os.path.exists(mount_point)
                    and os.stat(mount_point).st_dev == target_dev
                ):
                    return fs_type.lower() in ("tmpfs", "ramfs")
    except Exception as e:
        print(f"Error checking filesystem: {e}")

    return False


def try_preprocess_data(preprocess_config: str) -> None:
    command = [
        sys.executable,
        "-m",
        "flow_control.scripts.preprocess",
        preprocess_config,
    ]
    print(f"[blue]Executing data preprocessing command:[/blue] {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print("[green]Data preprocessing completed successfully.[/green]")
    except subprocess.CalledProcessError as e:
        code = e.returncode
        print(f"[red]Error during data preprocessing (exit code {code})[/red]")
        sys.exit(code)


def try_generate_dcp_seed(
    config_path: str, launch_config: LaunchConfig, config_data: dict
) -> None | str:
    if launch_config.generate_dcp_seed:
        if "seed_checkpoint_dir" not in config_data:
            print(
                "[yellow]Warning:[/yellow] 'launch.generate_dcp_seed' is True but 'seed_checkpoint_dir' is not specified "
                "in the config. Skipping DCP seed generation."
            )
            return None
        dir: str = config_data["seed_checkpoint_dir"]
        if not is_on_ram_disk(dir):
            print(
                f"[yellow]Warning:[/yellow] 'seed_checkpoint_dir' ({dir}) is not on a RAM disk. DCP seed generation may "
                "be slow and could wear out your SSD. Consider using a RAM disk for better performance and longevity."
            )
        if os.path.exists(dir):
            print(f"[blue]Removing existing seed checkpoint directory:[/blue] {dir}")
            shutil.rmtree(dir)
        print(f"[blue]Generating DCP seed checkpoint in:[/blue] {dir}")
        command = [sys.executable, "-m", "flow_control.scripts.seed", config_path]
        print(f"[blue]Executing command:[/blue] {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            print(
                f"[green]DCP seed checkpoint generated successfully in {dir}.[/green]"
            )
            return dir
        except subprocess.CalledProcessError as e:
            code = e.returncode
            print(
                f"[red]Error generating DCP seed checkpoint (exit code {code}) [/red]"
            )
            sys.exit(code)
    return None


def run(config_path: str) -> None:
    """Launch training as the parent process (sets up env, execs torchrun)."""
    launch_config, config = _load_launch_config(config_path)
    seed_path = None
    code = 1
    try:
        seed_path = try_generate_dcp_seed(config_path, launch_config, config)
        if launch_config.preprocess_config:
            try_preprocess_data(launch_config.preprocess_config)

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
            envs["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(d) for d in launch_config.devices
            )
        if "OMP_NUM_THREADS" not in envs:
            omp_threads = max(1, (os.cpu_count() or 0) // num_processes)
            envs["OMP_NUM_THREADS"] = str(omp_threads)
        for k, v in envs.items():
            print(f"[blue]Setting environment variable:[/blue] {k}={v}")
            os.environ[k] = v

        print(f"[blue]Executing command:[/blue] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        code = 0
    except subprocess.CalledProcessError as e:
        code = e.returncode
        print(f"[red]Error during torchrun execution (exit code {code})[/red]")
    finally:
        if seed_path and os.path.exists(seed_path):
            print(f"[blue]Cleaning up seed checkpoint directory:[/blue] {seed_path}")
            shutil.rmtree(seed_path)
    sys.exit(code)


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
