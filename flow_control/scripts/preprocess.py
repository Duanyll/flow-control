from collections.abc import Iterator
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, TypeAdapter

from flow_control.datasets import (
    DATASINK_REGISTRY,
    DatasetConfig,
    DatasinkConfig,
    parse_dataset,
)
from flow_control.processors import (
    PROCESSOR_TASK_REGISTRY,
    ProcessorConfig,
    parse_processor,
)
from flow_control.utils.coercion import (
    build_type_adapter,
    coerce_record,
    get_input_typeddict,
)
from flow_control.utils.config import load_config_file
from flow_control.utils.logging import dump_if_failed, get_logger
from flow_control.utils.pipeline import (
    DataSource,
    Pipeline,
    PipelineStage,
    SinkConfig,
    SourceConfig,
    StageConfig,
)
from flow_control.utils.tensor import deep_move_to_device


class TorchDatasetSource(DataSource):
    def __init__(self, dataset_args: dict | None = None):
        if dataset_args is None:
            dataset_args = {}
        self.dataset = parse_dataset(dataset_args)
        self.total = len(self.dataset)

    def scan(self) -> Iterator[tuple[Any, int | None]]:
        for idx in range(self.total):
            yield idx, self.total


class TorchDatasetLoaderStage(PipelineStage):
    def __init__(
        self,
        worker_id: int,
        device: int | None = None,
        dataset_args: dict | None = None,
        processor_args: dict | None = None,
        processing_mode: Literal["inference", "training"] = "training",
        attachment_dir: str | None = None,
        enable_coercion: bool = True,
    ):
        if dataset_args is None:
            dataset_args = {}
        self.worker_id = worker_id
        self.logger = get_logger(f"TorchDatasetLoaderStage-{worker_id}")
        self.dataset = parse_dataset(dataset_args)
        self.attachment_dir = attachment_dir or ""
        self.type_adapter: TypeAdapter | None = None

        if enable_coercion and processor_args:
            task_name = processor_args.get("task")
            if task_name and task_name in PROCESSOR_TASK_REGISTRY:
                processor_class = PROCESSOR_TASK_REGISTRY[task_name]
                mode: Literal["training", "inference"] = processing_mode
                typed_dict_class = get_input_typeddict(processor_class, mode)
                if typed_dict_class is not None:
                    self.type_adapter = build_type_adapter(typed_dict_class)
                    self.logger.info(
                        f"Coercion enabled for {task_name}/{mode}: {typed_dict_class.__name__}"
                    )

    def process(self, item: Any) -> Any:
        data = self.dataset[item]
        if self.type_adapter is not None:
            data = coerce_record(data, self.type_adapter, self.attachment_dir)
        return [data]


class ProcessorStage(PipelineStage):
    def __init__(
        self,
        worker_id: int,
        device: int | None = None,
        save_extra: bool = False,
        processing_mode: Literal["inference", "training"] = "training",
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
        self.save_extra = save_extra
        self.processing_mode = processing_mode
        self.logger.info(
            f"Initialized processor for {self.processing_mode}: {self.processor.__class__.__name__}"
        )
        self.processor.load_models("encode", device=self.device)
        self.logger.info("Processor models loaded.")

    async def process(self, item: Any) -> list[Any]:
        with dump_if_failed(self.logger, item):
            item = deep_move_to_device(item, self.device)
            if self.processing_mode == "inference":
                output = await self.processor.prepare_inference_batch(item)
            else:
                output = await self.processor.prepare_training_batch(item)
            output["latent_length"] = self.processor.get_latent_length(output)
            if self.save_extra:
                item.update(output)
                output = item
            if "__key__" not in output:
                output["__key__"] = item.get("__key__", None)  # type: ignore
            output = deep_move_to_device(output, torch.device("cpu"))
        return [output]


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    processor: ProcessorConfig
    output: DatasinkConfig

    num_loader_workers: int = 1
    num_sink_workers: int = 1
    processor_devices: list[int] = [0]
    processor_concurrency: int = 1  # Max concurrent async calls per processor worker
    num_threads_per_worker: int = 8
    queue_size: int = 16

    processing_mode: Literal["inference", "training"]
    save_extra: bool = False
    attachment_dir: str | None = None
    enable_coercion: bool = True


def run(config_path: str) -> None:
    """Run preprocessing pipeline with the given config file."""
    config = PreprocessConfig(**load_config_file(config_path))
    datasink_type = config.output.pop("type")

    pipeline = Pipeline(
        source=SourceConfig(
            source=TorchDatasetSource,
            name="Scanning",
            queue_size=config.queue_size,
            init_kwargs={"dataset_args": config.dataset},
        ),
        stages=[
            StageConfig(
                stage=TorchDatasetLoaderStage,
                num_workers=config.num_loader_workers,
                num_threads=config.num_threads_per_worker,
                queue_size=config.queue_size,
                name="Loading",
                init_kwargs={
                    "dataset_args": config.dataset,
                    "processor_args": config.processor,
                    "processing_mode": config.processing_mode,
                    "attachment_dir": config.attachment_dir,
                    "enable_coercion": config.enable_coercion,
                },
            ),
            StageConfig(
                stage=ProcessorStage,
                num_workers=len(config.processor_devices),
                gpu_ids=config.processor_devices,
                num_threads=config.num_threads_per_worker,
                queue_size=config.queue_size,
                max_concurrency=config.processor_concurrency,
                name="Processing",
                init_kwargs={
                    "processor_args": config.processor,
                    "processing_mode": config.processing_mode,
                    "save_extra": config.save_extra,
                },
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset using a pipeline.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the preprocessing configuration file.",
    )
    args = parser.parse_args()
    run(args.config_path)


if __name__ == "__main__":
    main()
