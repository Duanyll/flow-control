import torch
from typing import Any, Iterator
from flow_control.datasets import parse_dataset, DATASINK_REGISTRY
from flow_control.processors import parse_processor
from flow_control.utils.pipeline import (
    Pipeline,
    PipelineStage,
    DataSource,
    SourceConfig,
    StageConfig,
    SinkConfig,
)
from flow_control.utils.logging import get_logger
from pydantic import BaseModel

class TorchDatasetSource(DataSource):
    def __init__(self, **kwargs):
        self.dataset = parse_dataset(kwargs)
        self.total = len(self.dataset) # type: ignore

    def scan(self) -> Iterator[tuple[Any, int | None]]:
        for idx in range(len(self.dataset)):  # type: ignore
            yield idx, self.total


class TorchDatasetLoaderStage(PipelineStage):
    def __init__(self, worker_id: int, device: int | None = None, **kwargs):
        self.worker_id = worker_id
        self.logger = get_logger(f"TorchDatasetLoaderStage-{worker_id}")
        self.dataset = parse_dataset(kwargs)

    def process(self, item: Any) -> Any:
        data = self.dataset[item]  # type: ignore
        self.logger.debug(f"Loaded item {item} by worker {self.worker_id}")
        return data


class ProcessorStage(PipelineStage):
    def __init__(self, worker_id: int, device: int | None = None, **kwargs):
        self.worker_id = worker_id
        self.logger = get_logger(f"ProcessorStage-{worker_id}")

        if device is not None:
            self.device = torch.cuda.device(device)
        else:
            self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        kwargs['device'] = self.device
        self.processor = parse_processor(kwargs)
        self.logger.info(f"Initialized processor: {self.processor.__class__.__name__}")

        self.processor.load_models(['encode'])
        self.logger.info("Processor models loaded.")

    def process(self, item: Any) -> Any:
        result = self.processor.preprocess_batch(item)
        self.logger.debug(f"Processed item by worker {self.worker_id}")
        return result


class PreprocessConfig(BaseModel):
    dataset: dict
    processor: dict
    datasink: dict

    num_loader_workers: int = 1
    processor_devices: list[int] = [0]
    num_threads_per_worker: int = 8
    queue_size: int = 16


def main(config_path: str):
    from flow_control.utils.loaders import load_config_file

    config = PreprocessConfig(**load_config_file(config_path))
    datasink_type = config.datasink.pop("type")

    pipeline = Pipeline(
        source=SourceConfig(
            source=TorchDatasetSource,
            name="Scanning",
            queue_size=config.queue_size,
            init_kwargs=config.dataset,
        ),
        stages=[
            StageConfig(
                stage=TorchDatasetLoaderStage,
                num_workers=config.num_loader_workers,
                num_threads=config.num_threads_per_worker,
                queue_size=config.queue_size,
                name="Loading",
                init_kwargs=config.dataset,
            ),
            StageConfig(
                stage=ProcessorStage,
                num_workers=len(config.processor_devices),
                num_threads=config.num_threads_per_worker,
                queue_size=config.queue_size,
                name="Processing",
                init_kwargs={
                    **config.processor,
                    "device": None,  # Device will be set in the stage
                },
            ),
        ],
        sink=SinkConfig(
            sink=DATASINK_REGISTRY.get(datasink_type), # type: ignore
            name="Saving",
            queue_size=config.queue_size,
            init_kwargs=config.datasink,
        ),
    )

    pipeline.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset using a pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the preprocessing configuration file (YAML or JSON).",
    )
    args = parser.parse_args()
    main(args.config)