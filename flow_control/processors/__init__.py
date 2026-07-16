from typing import Annotated, Any, Literal

from flow_control.utils.registry import RegistryUnion

from .base import BaseProcessor, task_registry
from .presets import (
    Flux1Preset,
    Flux2Klein4BPreset,
    Flux2Klein9BPreset,
    Flux2Preset,
    HiDreamO1DevPreset,
    HiDreamO1FullPreset,
    Krea2RawPreset,
    Krea2TurboPreset,
    LongcatImageEditPreset,
    LongcatImagePreset,
    QwenImageEditPreset,
    QwenImageLayeredPreset,
    QwenImagePreset,
    Sd35MediumPreset,
    ZImagePreset,
    preset_registry,
)
from .tasks.inpaint import InpaintProcessor
from .tasks.qwen_layered import QwenImageLayeredProcessor
from .tasks.t2i import T2IProcessor
from .tasks.t2i_control import T2IControlProcessor
from .tasks.tie import TIEProcessor


def parse_processor(conf: dict) -> BaseProcessor:
    task = conf["task"]
    task_ctor = task_registry.get(task)
    if task_ctor is None:
        raise ValueError(
            f"Unknown processor task {task!r}; "
            f"registered: {sorted(task_registry.members())}"
        )
    ctor: type[BaseProcessor] = task_ctor

    preset_name = conf.get("preset")
    if preset_name:
        preset_ctor = preset_registry.get(preset_name)
        if preset_ctor is None:
            raise ValueError(
                f"Unknown processor preset {preset_name!r}; "
                f"registered: {sorted(preset_registry.members())}"
            )
        ctor = type(
            f"{preset_ctor.__name__.replace('Preset', '')}{task_ctor.__name__}",
            (preset_ctor, task_ctor),
            {},
        )
    return ctor(**conf)


def _validate_processor_dict(conf: dict[str, Any]) -> dict[str, Any]:
    # Validate by constructing the processor, but return the original dict
    # unchanged so the value can cross multiprocessing boundaries.
    parse_processor(conf)
    return conf


# Runtime validation dispatches through parse_processor (task x preset mixin
# composition); JSON schema is the materialized task union.
Processor = Annotated[
    BaseProcessor, RegistryUnion(task_registry, "task", parser=parse_processor)
]

# Same JSON schema as Processor, but the validated value stays a plain dict so
# it can cross multiprocessing boundaries.
ProcessorConfig = Annotated[
    dict[str, Any],
    RegistryUnion(task_registry, "task", parser=_validate_processor_dict),
]


def get_processor_input_typeddict(
    processor_class: type, mode: Literal["training", "inference"]
) -> type | None:
    """Extract the TypedDict type from a processor class's generic parameters.

    Returns the InputBatch (mode="inference") or TrainInputBatch (mode="training")
    TypedDict class, or None if it cannot be determined.
    """
    from flow_control.processors.base import BaseProcessor

    # Pydantic stores resolved generic args in __pydantic_generic_metadata__
    # on the parameterized BaseProcessor[...] class in the MRO.
    for base in processor_class.__mro__:
        meta = getattr(base, "__pydantic_generic_metadata__", None)
        if meta is None:
            continue
        origin = meta.get("origin")
        args = meta.get("args", ())
        if (
            origin is not None
            and (origin is BaseProcessor or issubclass(origin, BaseProcessor))
            and len(args) >= 2
        ):
            if mode == "inference":
                return args[0]
            else:
                return args[1]
    return None


__all__ = [
    "BaseProcessor",
    "Flux1Preset",
    "Flux2Klein4BPreset",
    "Flux2Klein9BPreset",
    "Flux2Preset",
    "HiDreamO1DevPreset",
    "HiDreamO1FullPreset",
    "InpaintProcessor",
    "Krea2RawPreset",
    "Krea2TurboPreset",
    "LongcatImageEditPreset",
    "LongcatImagePreset",
    "Processor",
    "ProcessorConfig",
    "QwenImageEditPreset",
    "QwenImageLayeredPreset",
    "QwenImageLayeredProcessor",
    "QwenImagePreset",
    "Sd35MediumPreset",
    "T2IControlProcessor",
    "T2IProcessor",
    "TIEProcessor",
    "ZImagePreset",
    "get_processor_input_typeddict",
    "parse_processor",
    "preset_registry",
    "task_registry",
]
