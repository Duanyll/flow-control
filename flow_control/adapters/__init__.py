from typing import Annotated, Any

from flow_control.utils.registry import RegistryUnion

from .arch.cosmos3 import Cosmos3Adapter, Cosmos3KVAdapter
from .arch.flux1 import (
    Flux1Adapter,
    Flux1DConcatAdapter,
    Flux1FillAdapter,
    Flux1KontextAdapter,
    Flux1NConcatAdapter,
)
from .arch.flux2 import Flux2Adapter
from .arch.flux2.kv import Flux2KVAdapter
from .arch.flux3 import Flux3Adapter
from .arch.hidream import HiDreamO1Adapter
from .arch.ideogram4 import Ideogram4Adapter
from .arch.krea2 import Krea2Adapter
from .arch.longcat import LongCatAdapter, LongCatEditAdapter
from .arch.qwen import (
    QwenImageAdapter,
    QwenImageEditAdapter,
    QwenImageLayeredAdapter,
)
from .arch.qwen21 import QwenImage21Adapter
from .arch.sd3 import SD3Adapter
from .arch.zimage import ZImageAdapter
from .base import BaseModelAdapter, Batch, adapter_registry

__all__ = [
    "Cosmos3Adapter",
    "Cosmos3KVAdapter",
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
    "Flux2KVAdapter",
    "Flux3Adapter",
    "HiDreamO1Adapter",
    "Ideogram4Adapter",
    "Krea2Adapter",
    "LongCatAdapter",
    "LongCatEditAdapter",
    "QwenImageAdapter",
    "QwenImage21Adapter",
    "QwenImageEditAdapter",
    "QwenImageLayeredAdapter",
    "SD3Adapter",
    "ZImageAdapter",
]


ModelAdapter = Annotated[BaseModelAdapter, RegistryUnion(adapter_registry, "type")]


def parse_model_adapter(conf: dict[str, Any]) -> BaseModelAdapter:
    """Parse a model adapter config dict into the appropriate adapter instance."""
    from pydantic import TypeAdapter

    ta = TypeAdapter(ModelAdapter)
    return ta.validate_python(conf)
