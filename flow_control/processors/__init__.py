from typing import Annotated

from pydantic import PlainValidator

from .base import BaseProcessor
from .flux1 import Flux1Processor

PROCESSOR_REGISTRY = {
    "base": BaseProcessor,
    "flux1": Flux1Processor,
}

def parse_processor(conf: dict) -> BaseProcessor:
    processor_type = conf.pop("type")
    processor_class = PROCESSOR_REGISTRY.get(processor_type)
    if processor_class is None:
        raise ValueError(f"Unknown processor type: {processor_type}")
    return processor_class(**conf)

Processor = Annotated[BaseProcessor, PlainValidator(parse_processor)]

__all__ = [
    "Processor",
]