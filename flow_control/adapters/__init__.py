from typing import Annotated, Any

from pydantic import Discriminator, Tag

from .base import BaseModelAdapter, Batch
from .flux1 import (
    Flux1Adapter,
    Flux1DConcatAdapter,
    Flux1FillAdapter,
    Flux1KontextAdapter,
    Flux1NConcatAdapter,
)
from .flux2 import Flux2Adapter
from .longcat import LongCatAdapter, LongCatEditAdapter
from .qwen import (
    EfficientLayeredQwenImageAdapter,
    QwenImageAdapter,
    QwenImageEditAdapter,
    QwenImageLayeredAdapter,
)
from .zimage import ZImageAdapter


def _adapter_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return f"{v['arch']}_{v['type']}"
    return f"{v.arch}_{v.type}"


ModelAdapter = Annotated[
    Annotated[Flux1Adapter, Tag("flux1_base")]
    | Annotated[Flux1DConcatAdapter, Tag("flux1_d_concat")]
    | Annotated[Flux1NConcatAdapter, Tag("flux1_n_concat")]
    | Annotated[Flux1FillAdapter, Tag("flux1_fill")]
    | Annotated[Flux1KontextAdapter, Tag("flux1_kontext")]
    | Annotated[Flux2Adapter, Tag("flux2_base")]
    | Annotated[QwenImageAdapter, Tag("qwen_base")]
    | Annotated[QwenImageEditAdapter, Tag("qwen_edit")]
    | Annotated[QwenImageLayeredAdapter, Tag("qwen_layered")]
    | Annotated[EfficientLayeredQwenImageAdapter, Tag("qwen_efficient_layered")]
    | Annotated[LongCatAdapter, Tag("longcat_base")]
    | Annotated[LongCatEditAdapter, Tag("longcat_edit")]
    | Annotated[ZImageAdapter, Tag("zimage_base")],
    Discriminator(_adapter_discriminator),
]


def parse_model_adapter(conf: dict[str, Any]) -> BaseModelAdapter:
    """Parse a model adapter config dict into the appropriate adapter instance."""
    from pydantic import TypeAdapter

    ta = TypeAdapter(ModelAdapter)
    return ta.validate_python(conf)


__all__ = [
    "ModelAdapter",
    "Batch",
    "parse_model_adapter",
]
