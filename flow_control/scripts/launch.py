import argparse
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from rich import print

from flow_control.training.launch_config import LaunchConfig
from flow_control.utils.config import (
    add_config_patch_arguments,
    format_config_patch_args,
    load_config_file,
)


def _load_launch_config(
    config_path: str,
    updates: Sequence[str] = (),
    removes: Sequence[str] = (),
) -> tuple[LaunchConfig, dict]:
    config_data = load_config_file(config_path, updates, removes)
    if "launch" not in config_data:
        raise ValueError("Launch configuration section is missing in the config file.")
    launch_config = LaunchConfig(**config_data["launch"])
    return launch_config, config_data


def _run_child(launch_config: LaunchConfig, config_data: dict) -> None:
    """Run the child process (invoked by torchrun)."""
    if launch_config.type == "sft":
        from flow_control.training.sft import SftTrainer

        trainer = SftTrainer(**config_data)
        trainer.run()
    elif launch_config.type == "grpo":
        from flow_control.training.grpo import GrpoTrainer

        trainer = GrpoTrainer(**config_data)
        trainer.run()
    elif launch_config.type == "nft":
        from flow_control.training.nft import NftTrainer

        trainer = NftTrainer(**config_data)
        trainer.run()
    elif launch_config.type == "vae":
        from flow_control.training.vae import VaeTrainer

        trainer = VaeTrainer(**config_data)
        trainer.run()
    elif launch_config.type == "inference":
        from flow_control.training.inference import Inference

        trainer = Inference(**config_data)
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


def try_preprocess_data(preprocess_config: str | list[str]) -> None:
    if isinstance(preprocess_config, str):
        configs = [preprocess_config]
    else:
        configs = preprocess_config

    for config in configs:
        command = [
            sys.executable,
            "-m",
            "flow_control.scripts.cli",
            "preprocess",
            config,
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
    config_path: str,
    launch_config: LaunchConfig,
    config_data: dict,
    updates: Sequence[str] = (),
    removes: Sequence[str] = (),
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
        command = [
            sys.executable,
            "-m",
            "flow_control.scripts.cli",
            "seed",
            config_path,
            *format_config_patch_args(updates, removes),
        ]
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


def _detect_num_gpus() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as e:
        print(f"[red]Error detecting GPUs with nvidia-smi: {e}[/red]")
        sys.exit(1)
    num_gpus = len(result.stdout.strip().splitlines())
    print(f"[blue]Detected {num_gpus} GPUs with nvidia-smi.[/blue]")
    return num_gpus


def _detect_nproc() -> int:
    try:
        result = subprocess.run(
            ["nproc"],
            capture_output=True,
            text=True,
            check=True,
        )
        num_cpus = int(result.stdout.strip())
        print(f"[blue]Detected {num_cpus} CPUs with nproc.[/blue]")
        return num_cpus
    except Exception as e:
        print(f"[red]Error detecting CPUs with nproc: {e}[/red]")
        fallback = os.cpu_count() or 0
        print(
            f"[yellow]Falling back to os.cpu_count(): {fallback} CPUs detected.[/yellow]"
        )
        return fallback


def _resolve_num_processes(launch_config: LaunchConfig) -> int:
    num_gpus = _detect_num_gpus()
    devices = launch_config.devices
    if devices == "all":
        return num_gpus
    if isinstance(devices, int):
        if devices > num_gpus:
            print(
                f"[red]Error: 'devices' is set to {devices} but only {num_gpus} GPUs are available.[/red]"
            )
            sys.exit(1)
        return devices
    if isinstance(devices, list):
        return len(devices)
    print(f"[red]Invalid 'devices' configuration: {devices}[/red]")
    sys.exit(1)


def _default_log_dir() -> str:
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        array = os.getenv("SLURM_ARRAY_TASK_ID")
        suffix = f"{job_id}_{array}" if array else job_id
        return str(Path("logs") / f"slurm-{suffix}")
    return str(Path("logs") / f"local-{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}")


def run(
    config_path: str,
    updates: Sequence[str] = (),
    removes: Sequence[str] = (),
) -> None:
    """Launch training as the parent process (sets up env, execs torchrun)."""
    launch_config, config = _load_launch_config(config_path, updates, removes)

    try_generate_dcp_seed(config_path, launch_config, config, updates, removes)
    if launch_config.preprocess_config:
        try_preprocess_data(launch_config.preprocess_config)

    num_processes = _resolve_num_processes(launch_config)

    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(num_processes),
        "-m",
        "flow_control.scripts.launch",
        config_path,
        *format_config_patch_args(updates, removes),
        "--type",
        launch_config.type,
    ]

    envs = dict(launch_config.env)
    if launch_config.trackio_dir is not None and "TRACKIO_DIR" not in envs:
        envs["TRACKIO_DIR"] = launch_config.trackio_dir
    if isinstance(launch_config.devices, list):
        envs["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in launch_config.devices)
    if "OMP_NUM_THREADS" not in envs:
        # omp_threads = max(1, (os.cpu_count() or 0) // num_processes)
        # Call `nproc` to get the number of CPUs available to this process
        omp_threads = max(1, _detect_nproc() // num_processes)
        envs["OMP_NUM_THREADS"] = str(omp_threads)
    if "LOG_DIR" not in envs and not os.getenv("LOG_DIR"):
        envs["LOG_DIR"] = _default_log_dir()
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
    add_config_patch_arguments(parser)
    args = parser.parse_args()

    if args.type:
        # This is child process (invoked by torchrun)
        launch_config, config_data = _load_launch_config(
            args.config_path, args.config_updates, args.config_removes
        )
        _run_child(launch_config, config_data)
    else:
        # This is parent process
        run(args.config_path, args.config_updates, args.config_removes)


if __name__ == "__main__":
    main()
