from typing import Annotated, Any

from flow_control.utils.registry import RegistryUnion

from .base import BaseModelAdapter, Batch, adapter_registry
from .flux1 import (
    Flux1Adapter,
    Flux1DConcatAdapter,
    Flux1FillAdapter,
    Flux1KontextAdapter,
    Flux1NConcatAdapter,
)
from .flux2 import Flux2Adapter
from .hidream import HiDreamO1Adapter
from .krea2 import Krea2Adapter
from .longcat import LongCatAdapter, LongCatEditAdapter
from .qwen import (
    QwenImageAdapter,
    QwenImageEditAdapter,
    QwenImageLayeredAdapter,
)
from .sd3 import SD3Adapter
from .zimage import ZImageAdapter

__all__ = [
    "ModelAdapter",
    "Batch",
    "parse_model_adapter",
    "BaseModelAdapter",
    "adapter_registry",
    "Flux1Adapter",
    "Flux1DConcatAdapter",
    "Flux1FillAdapter",
    "Flux1KontextAdapter",
    "Flux1NConcatAdapter",
    "Flux2Adapter",
    "HiDreamO1Adapter",
    "Krea2Adapter",
    "LongCatAdapter",
    "LongCatEditAdapter",
    "QwenImageAdapter",
    "QwenImageEditAdapter",
    "QwenImageLayeredAdapter",
    "SD3Adapter",
    "ZImageAdapter",
]


def _adapter_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return f"{v['arch']}_{v['type']}"
    return f"{v.arch}_{v.type}"


ModelAdapter = Annotated[
    BaseModelAdapter, RegistryUnion(adapter_registry, _adapter_discriminator)
]


def parse_model_adapter(conf: dict[str, Any]) -> BaseModelAdapter:
    """Parse a model adapter config dict into the appropriate adapter instance."""
    from pydantic import TypeAdapter

    ta = TypeAdapter(ModelAdapter)
    return ta.validate_python(conf)
