from .base import BaseQwenImageAdapter
from .edit import QwenImageEditAdapter

ADAPTER_REGISTRY = {
    "base": BaseQwenImageAdapter,
    "edit": QwenImageEditAdapter,
}


def parse_adapter(conf: dict) -> BaseQwenImageAdapter:
    adapter_type = conf.pop("type")
    adapter_class = ADAPTER_REGISTRY.get(adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return adapter_class(**conf)
