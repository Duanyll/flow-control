"""
Demo script for the pipeline framework.

Run with: uv run -m flow_control.utils.pipeline_demo
"""

import random
import time
from typing import Iterator, List, Optional

from .pipeline import (
    Pipeline,
    PipelineStage,
    DataSource,
    DataSink,
    SourceConfig,
    StageConfig,
    SinkConfig,
)
from .logging import get_logger


class MockDataSource(DataSource):
    """Generate mock file paths."""

    def __init__(self, num_items: int = 50, **kwargs):
        self.num_items = num_items

    def scan(self) -> Iterator[tuple[str, int | None]]:
        for i in range(self.num_items):
            yield f"item_{i:04d}.dat", self.num_items
            time.sleep(0.01)  # Simulate scanning delay


class MockLoader(PipelineStage):
    """Simulate loading data with random filtering."""

    def __init__(self, worker_id: int, device: Optional[int] = None, filter_rate: float = 0.1, **kwargs):
        self.worker_id = worker_id
        self.filter_rate = filter_rate
        self.logger = get_logger(f"MockLoader-{worker_id}")

    def process(self, item: str) -> List[dict]:
        time.sleep(random.uniform(0.05, 0.15))  # Simulate I/O

        # Random filtering
        if random.random() < self.filter_rate:
            self.logger.debug(f"Filtered: {item}")
            return []

        # Return loaded "data"
        data = {
            "path": item,
            "data": [random.random() for _ in range(10)],
            "loaded_by": self.worker_id,
        }
        self.logger.debug(f"Loaded: {item}")
        return [data]


class MockProcessor(PipelineStage):
    """Simulate processing with random splitting."""

    def __init__(self, worker_id: int, device: Optional[int] = None, split_rate: float = 0.2, max_split: int = 3, **kwargs):
        self.worker_id = worker_id
        self.split_rate = split_rate
        self.max_split = max_split
        self.logger = get_logger(f"MockProcessor-{worker_id}")

    def process(self, item: dict) -> List[dict]:
        time.sleep(random.uniform(0.1, 0.3))  # Simulate computation

        # Determine how many outputs
        if random.random() < self.split_rate:
            num_outputs = random.randint(2, self.max_split)
        else:
            num_outputs = 1

        results = []
        for i in range(num_outputs):
            result = {
                "path": item["path"],
                "variant": i,
                "result": sum(item["data"]) * random.uniform(0.9, 1.1),
                "processed_by": self.worker_id,
            }
            results.append(result)

        self.logger.debug(f"Processed: {item['path']} -> {num_outputs} output(s)")
        return results


class MockSink(DataSink):
    """Simulate writing results."""

    def __init__(self, worker_id: int, output_dir: str = "/tmp/mock_output", **kwargs):
        self.worker_id = worker_id
        self.output_dir = output_dir
        self.logger = get_logger(f"MockSink-{worker_id}")
        self.count = 0

    def write(self, item: dict) -> bool:
        time.sleep(random.uniform(0.02, 0.08))  # Simulate I/O
        self.count += 1
        variant = item.get("variant", 0)
        result = item.get("result", 0)
        self.logger.debug(f"Written: {item['path']}:{variant} (result={result:.3f})")
        return True

    def cleanup(self):
        self.logger.info(f"Sink worker {self.worker_id} wrote {self.count} items")


class FaultyProcessor(PipelineStage):
    """Processor that throws a fatal error during setup."""

    def __init__(self, worker_id: int, device: Optional[int] = None, fail_on_setup: bool = False, **kwargs):
        if fail_on_setup:
            raise RuntimeError("Simulated setup failure!")
        self.worker_id = worker_id
        self.logger = get_logger(f"FaultyProcessor-{worker_id}")
        self.count = 0

    def process(self, item: dict) -> List[dict]:
        self.count += 1
        if self.count >= 5:
            raise RuntimeError("I am tired of processing!")
        time.sleep(random.uniform(0.1, 0.2))
        return [item]


def main(test_fatal_error: bool = False):
    print("=" * 60)
    print("Pipeline Demo: Mock Data Processing")
    if test_fatal_error:
        print("(Testing fatal error handling)")
    print("=" * 60)
    print()

    if test_fatal_error:
        # Use FaultyProcessor that will fail during setup
        stages = [
            StageConfig(
                stage=MockLoader,
                num_workers=2,
                num_threads=2,
                queue_size=4,
                name="Loading",
                init_kwargs={"filter_rate": 0.1},
            ),
            StageConfig(
                stage=FaultyProcessor,
                num_workers=2,
                num_threads=2,
                queue_size=4,
                name="FaultyProcessing",
                init_kwargs={"fail_on_setup": False},
            ),
        ]
    else:
        stages = [
            StageConfig(
                stage=MockLoader,
                num_workers=2,
                num_threads=2,
                queue_size=4,
                name="Loading",
                init_kwargs={"filter_rate": 0.1},
            ),
            StageConfig(
                stage=MockProcessor,
                num_workers=3,
                num_threads=2,
                queue_size=4,
                name="Processing",
                init_kwargs={"split_rate": 0.2, "max_split": 3},
            ),
        ]

    pipeline = Pipeline(
        source=SourceConfig(
            source=MockDataSource,
            name="Scanning",
            queue_size=8,
            init_kwargs={"num_items": 100},
        ),
        stages=stages,
        sink=SinkConfig(
            sink=MockSink,
            num_workers=2,
            num_threads=2,
            name="Writing",
            init_kwargs={"output_dir": "/tmp/mock_output"},
        ),
    )

    result = pipeline.run()

    print()
    print("=" * 60)
    print("Final Statistics:")
    print(f"  Source items: {result.source_total}")
    print(f"  Aborted: {result.aborted}")
    if result.error_message:
        print(f"  Error: {result.error_message}")
    print(f"  Elapsed time: {result.elapsed_time:.2f}s")
    if result.source_total > 0 and result.elapsed_time > 0:
        print(f"  Throughput: {result.source_total / result.elapsed_time:.1f} items/s")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    test_fatal = "--fatal" in sys.argv
    main(test_fatal_error=test_fatal)
