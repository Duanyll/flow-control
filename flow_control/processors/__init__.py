from typing import Annotated

from pydantic import PlainValidator

from .base import BaseProcessor
from .efficient_layered import EfficientLayeredQwenImageProcessor
from .flux1 import Flux1Processor
from .kontext import KontextProcessor
from .qwen import QwenImageProcessor
from .qwen_edit import QwenImageEditProcessor
from .qwen_layered import QwenImageLayeredProcessor

PROCESSOR_REGISTRY = {
    "base": BaseProcessor,
    "flux1": Flux1Processor,
    "kontext": KontextProcessor,
    "qwen": QwenImageProcessor,
    "qwen_edit": QwenImageEditProcessor,
    "qwen_layered": QwenImageLayeredProcessor,
    "efficient_layered": EfficientLayeredQwenImageProcessor,
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
