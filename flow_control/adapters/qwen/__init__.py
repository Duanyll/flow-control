from .base import QwenImageAdapter
from .edit import QwenImageEditAdapter
from .efficient_layered import EfficientLayeredQwenImageAdapter
from .layered import QwenImageLayeredAdapter

ADAPTER_REGISTRY = {
    "base": QwenImageAdapter,
    "edit": QwenImageEditAdapter,
    "layered": QwenImageLayeredAdapter,
    "efficient_layered": EfficientLayeredQwenImageAdapter,
}


def parse_adapter(conf: dict) -> QwenImageAdapter:
    adapter_type = conf["type"]
    adapter_class = ADAPTER_REGISTRY.get(adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return adapter_class(**conf)
