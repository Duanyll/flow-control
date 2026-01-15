import inspect
from collections.abc import Iterator
from typing import Any

import torch
from pydantic import BaseModel

from flow_control.datasets import DATASINK_REGISTRY, parse_dataset
from flow_control.processors import parse_processor
from flow_control.utils.common import deep_move_to_device
from flow_control.utils.hf_model import load_config_file
from flow_control.utils.logging import get_logger
from flow_control.utils.pipeline import (
    DataSource,
    Pipeline,
    PipelineStage,
    SinkConfig,
    SourceConfig,
    StageConfig,
)


class TorchDatasetSource(DataSource):
    def __init__(
        self, dataset_args: dict | None = None, processing_limit: int | None = None
    ):
        if dataset_args is None:
            dataset_args = {}
        self.dataset = parse_dataset(dataset_args)
        self.total = len(self.dataset)  # type: ignore
        self.processing_limit = processing_limit

    def scan(self) -> Iterator[tuple[Any, int | None]]:
        limit = len(self.dataset)  # type: ignore
        if self.processing_limit is not None:
            limit = min(limit, self.processing_limit)
        for idx in range(limit):
            yield idx, limit


class TorchDatasetLoaderStage(PipelineStage):
    def __init__(
        self,
        worker_id: int,
        device: int | None = None,
        dataset_args: dict | None = None,
    ):
        if dataset_args is None:
            dataset_args = {}
        self.worker_id = worker_id
        self.logger = get_logger(f"TorchDatasetLoaderStage-{worker_id}")
        self.dataset = parse_dataset(dataset_args)

    def process(self, item: Any) -> Any:
        data = self.dataset[item]  # type: ignore
        self.logger.debug(f"Loaded item {item} by worker {self.worker_id}")
        return [data]


class ProcessorStage(PipelineStage):
    def __init__(
        self,
        worker_id: int,
        device: int | None = None,
        processor_args: dict | None = None,
    ):
        if processor_args is None:
            processor_args = {}
        self.worker_id = worker_id
        self.logger = get_logger(f"ProcessorStage-{worker_id}")

        self.device = torch.device(f"cuda:{device}" if device is not None else "cpu")
        self.logger.info(f"Using device: {self.device}")
        processor_args["device"] = self.device
        self.processor = parse_processor(processor_args)
        self.logger.info(f"Initialized processor: {self.processor.__class__.__name__}")

        self.processor.load_models(["encode"], device=self.device)
        self.logger.info("Processor models loaded.")

        # Check if preprocess_batch is async
        self._is_async = inspect.iscoroutinefunction(self.processor.preprocess_batch)
        if self._is_async:
            self.logger.info("Using async preprocess_batch")

    def process(self, batch: Any) -> Any:
        batch = deep_move_to_device(batch, self.device)
        batch = self.processor.preprocess_batch(batch)
        batch = deep_move_to_device(batch, torch.device("cpu"))
        self.logger.debug(f"Processed item by worker {self.worker_id}")
        return [batch]

    async def _async_process(self, batch: Any) -> Any:
        batch = deep_move_to_device(batch, self.device)
        batch = await self.processor.preprocess_batch(batch)  # type: ignore[misc]
        batch = deep_move_to_device(batch, torch.device("cpu"))
        self.logger.debug(f"Processed item by worker {self.worker_id}")
        return [batch]

    def __getattribute__(self, name: str) -> Any:
        if name == "process" and object.__getattribute__(self, "_is_async"):
            return object.__getattribute__(self, "_async_process")
        return object.__getattribute__(self, name)


class PreprocessConfig(BaseModel):
    dataset: dict
    processor: dict
    output: dict

    num_loader_workers: int = 1
    num_sink_workers: int = 1
    processor_devices: list[int] = [0]
    processor_concurrency: int = 1  # Max concurrent async calls per processor worker
    num_threads_per_worker: int = 8
    queue_size: int = 16
    processing_limit: int | None = None  # Limit number of items to process


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset using a pipeline.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the preprocessing configuration file (YAML or JSON).",
    )
    args = parser.parse_args()
    config_path = args.config_path

    config = PreprocessConfig(**load_config_file(config_path))
    datasink_type = config.output.pop("type")

    pipeline = Pipeline(
        source=SourceConfig(
            source=TorchDatasetSource,
            name="Scanning",
            queue_size=config.queue_size,
            init_kwargs={
                "dataset_args": config.dataset,
                "processing_limit": config.processing_limit,
            },
        ),
        stages=[
            StageConfig(
                stage=TorchDatasetLoaderStage,
                num_workers=config.num_loader_workers,
                num_threads=config.num_threads_per_worker,
                queue_size=config.queue_size,
                name="Loading",
                init_kwargs={"dataset_args": config.dataset},
            ),
            StageConfig(
                stage=ProcessorStage,
                num_workers=len(config.processor_devices),
                gpu_ids=config.processor_devices,
                num_threads=config.num_threads_per_worker,
                queue_size=config.queue_size,
                max_concurrency=config.processor_concurrency,
                name="Processing",
                init_kwargs={"processor_args": config.processor},
            ),
        ],
        sink=SinkConfig(
            sink=DATASINK_REGISTRY.get(datasink_type),  # type: ignore
            name="Saving",
            num_workers=config.num_sink_workers,
            queue_size=config.queue_size,
            init_kwargs=config.output,
        ),
    )

    pipeline.run()


if __name__ == "__main__":
    main()
