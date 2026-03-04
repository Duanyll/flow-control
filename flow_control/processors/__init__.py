from typing import Annotated

from pydantic import PlainValidator

from .base import BaseProcessor
from .presets import (
    Flux1Preset,
    Flux2Klein4BPreset,
    Flux2Klein9BPreset,
    Flux2Preset,
    LongcatImageEditPreset,
    LongcatImagePreset,
    QwenImageEditPreset,
    QwenImageLayeredPreset,
    QwenImagePreset,
    ZImagePreset,
)
from .tasks.efficient_layered import EfficientLayeredProcessor
from .tasks.inpaint import InpaintProcessor
from .tasks.qwen_layered import QwenImageLayeredProcessor
from .tasks.t2i import T2IProcessor
from .tasks.t2i_control import T2IControlProcessor
from .tasks.tie import TIEProcessor

PROCESSOR_TASK_REGISTRY = {
    "t2i": T2IProcessor,
    "t2i_control": T2IControlProcessor,
    "inpaint": InpaintProcessor,
    "efficient_layered": EfficientLayeredProcessor,
    "qwen_layered": QwenImageLayeredProcessor,
    "tie": TIEProcessor,
}

PROCESSOR_PRESET_REGISTRY = {
    "flux1": Flux1Preset,
    "qwen_image": QwenImagePreset,
    "qwen_image_edit": QwenImageEditPreset,
    "qwen_image_layered": QwenImageLayeredPreset,
    "longcat_image": LongcatImagePreset,
    "longcat_image_edit": LongcatImageEditPreset,
    "zimage": ZImagePreset,
    "flux2": Flux2Preset,
    "flux2_klein_4b": Flux2Klein4BPreset,
    "flux2_klein_9b": Flux2Klein9BPreset,
}


def parse_processor(conf: dict) -> BaseProcessor:
    task = conf["task"]
    ctor = PROCESSOR_TASK_REGISTRY[task]
    if "preset" in conf:
        preset_name = conf["preset"]
        preset_ctor = PROCESSOR_PRESET_REGISTRY[preset_name]
        ctor = type(
            f"{preset_ctor.__name__.replace('Preset', '')}{ctor.__name__}",
            (preset_ctor, ctor),
            {},
        )
    return ctor(**conf)


Processor = Annotated[BaseProcessor, PlainValidator(parse_processor)]

__all__ = [
    "Processor",
]
