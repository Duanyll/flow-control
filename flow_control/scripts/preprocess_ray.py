from typing import Any, Literal

import torch
from pydantic import BaseModel
from rich.progress import Progress

from flow_control.datasets import DATASINK_REGISTRY, parse_dataset
from flow_control.processors import parse_processor
from flow_control.utils.common import deep_move_to_device, load_config_file
from flow_control.utils.logging import console, dump_if_failed, get_logger

logger = get_logger(__name__)


class RayPreprocessConfig(BaseModel):
    dataset: dict
    processor: dict
    output: dict

    num_loader_workers: int = 1
    num_sink_workers: int = 1
    processor_devices: list[int] = [0]
    processor_concurrency: int = 1
    processing_limit: int | None = None

    processing_mode: Literal["inference", "training"]
    save_intermediate: bool = False

    ray_address: str | None = None


class DatasetLoader:
    """Ray map callable: loads a single item from the torch Dataset by index."""

    def __init__(self, dataset_args: dict | None = None):
        if dataset_args is None:
            dataset_args = {}
        self.dataset = parse_dataset(dataset_args)

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        idx: int = row["id"]
        return self.dataset[idx]


class ProcessorMapper:
    """Ray map callable: runs GPU encoding on a single data item."""

    def __init__(
        self,
        processor_args: dict | None = None,
        processing_mode: Literal["inference", "training"] = "training",
        save_intermediate: bool = False,
    ):
        if processor_args is None:
            processor_args = {}
        self.device = torch.device("cuda")
        self.processing_mode = processing_mode
        self.save_intermediate = save_intermediate

        processor_args = {**processor_args, "device": self.device}
        self.processor = parse_processor(processor_args)
        self.processor.load_models("encode", device=self.device)
        self.logger = get_logger("ProcessorMapper")
        self.logger.info(
            f"Initialized processor for {self.processing_mode}: "
            f"{self.processor.__class__.__name__} on {self.device}"
        )

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        with torch.no_grad(), dump_if_failed(self.logger, row):
            item = deep_move_to_device(row, self.device)
            if self.processing_mode == "inference":
                output = await self.processor.prepare_inference_batch(item)
            else:
                output = await self.processor.prepare_training_batch(item)
            if self.save_intermediate:
                item.update(output)
                output = item
            if "__key__" not in output:
                output["__key__"] = item.get("__key__", None)  # type: ignore[union-attr]
            output = deep_move_to_device(output, torch.device("cpu"))
        return output


class SinkWriter:
    """Ray map callable: writes a single item via a DataSink."""

    def __init__(self, sink_type: str, **sink_kwargs: Any):
        sink_class = DATASINK_REGISTRY.get(sink_type)
        if sink_class is None:
            msg = f"Unknown datasink type: {sink_type}"
            raise ValueError(msg)
        self.sink = sink_class(worker_id=0, **sink_kwargs)

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        self.sink.write(row)
        return row


def run(config_path: str) -> None:
    """Run preprocessing pipeline using Ray Data with the given config file."""
    import ray.data

    config = RayPreprocessConfig(**load_config_file(config_path))

    ray.init(
        address=config.ray_address,
        log_to_driver=False,
        runtime_env={"env_vars": {"FLOW_CONTROL_RAY_WORKER": "1"}},
    )

    dataset = parse_dataset(config.dataset)
    total = len(dataset)
    if config.processing_limit is not None:
        total = min(total, config.processing_limit)
    del dataset  # free memory in driver

    sink_kwargs = config.output.copy()
    sink_type: str = sink_kwargs.pop("type")

    console.print(f"Starting Ray preprocessing pipeline for {total} items")

    ds = ray.data.range(total)
    # Ray Data supports passing classes as stateful callables with fn_constructor_kwargs,
    # but its type stubs only declare fn as Callable[[dict], dict], so we suppress here.
    ds = ds.map(
        DatasetLoader,  # type: ignore[arg-type]
        concurrency=config.num_loader_workers,
        fn_constructor_kwargs={"dataset_args": config.dataset},
    )
    ds = ds.map(
        ProcessorMapper,  # type: ignore[arg-type]
        concurrency=len(config.processor_devices),
        num_gpus=1,
        ray_remote_args={"max_concurrency": config.processor_concurrency},
        fn_constructor_kwargs={
            "processor_args": config.processor,
            "processing_mode": config.processing_mode,
            "save_intermediate": config.save_intermediate,
        },
    )
    ds = ds.map(
        SinkWriter,  # type: ignore[arg-type]
        concurrency=config.num_sink_workers,
        fn_constructor_kwargs={"sink_type": sink_type, **sink_kwargs},
    )
    # Drain the pipeline in streaming fashion — each batch is freed after consumption,
    # avoiding holding all processed data (potentially TBs) in memory.
    with Progress(console=console) as progress:
        task = progress.add_task("Processing", total=total)
        for _ in ds.iter_rows():
            progress.update(task, advance=1)

    console.print(f"Ray preprocessing complete: {total} items processed")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess dataset using a Ray Data pipeline."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the preprocessing configuration file.",
    )
    args = parser.parse_args()
    run(args.config_path)


if __name__ == "__main__":
    main()
