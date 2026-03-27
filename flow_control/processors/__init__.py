from typing import Annotated, Any, Literal

from pydantic import Discriminator, GetCoreSchemaHandler, Tag
from pydantic_core import CoreSchema, core_schema

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


# Schema-only union of task classes for JSON schema generation
_ProcessorTaskUnion = Annotated[
    Annotated[T2IProcessor, Tag("t2i")]
    | Annotated[T2IControlProcessor, Tag("t2i_control")]
    | Annotated[InpaintProcessor, Tag("inpaint")]
    | Annotated[EfficientLayeredProcessor, Tag("efficient_layered")]
    | Annotated[QwenImageLayeredProcessor, Tag("qwen_layered")]
    | Annotated[TIEProcessor, Tag("tie")],
    Discriminator("task"),
]


class _ProcessorAnnotation:
    """Custom annotation for Processor type.

    - Core schema: wraps the task union schema so $defs are shared with the parent.
    - Runtime: uses parse_processor for preset mixin composition.
    - JSON schema: the union schema's JSON schema is generated automatically.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # Generate the union schema through the handler so all $defs are
        # registered in the parent's schema generation context.
        union_schema = handler.generate_schema(_ProcessorTaskUnion)
        return core_schema.with_info_wrap_validator_function(
            cls._validate,
            union_schema,
        )

    @staticmethod
    def _validate(value: Any, handler: Any, info: Any) -> BaseProcessor:
        # Ignore the union handler — use parse_processor for runtime
        if isinstance(value, BaseProcessor):
            return value
        if isinstance(value, dict):
            return parse_processor(value)
        raise ValueError(f"Expected dict or BaseProcessor, got {type(value)}")


Processor = Annotated[BaseProcessor, _ProcessorAnnotation]


class _ProcessorConfigAnnotation:
    """Custom annotation for ProcessorConfig type.

    Same JSON schema as Processor (full task union with $defs), but the
    validated value stays as a plain dict so it can cross multiprocessing
    boundaries. Runtime validation is done via parse_processor.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        union_schema = handler.generate_schema(_ProcessorTaskUnion)
        return core_schema.with_info_wrap_validator_function(
            cls._validate,
            union_schema,
        )

    @staticmethod
    def _validate(value: Any, handler: Any, info: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            parse_processor(value)  # validate, discard the instance
            return value
        raise ValueError(f"Expected dict, got {type(value)}")


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


ProcessorConfig = Annotated[dict[str, Any], _ProcessorConfigAnnotation]

__all__ = [
    "Processor",
    "ProcessorConfig",
    "parse_processor",
]
