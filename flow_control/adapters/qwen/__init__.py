from .base import BaseQwenImageAdapter
from .edit import QwenImageEditAdapter
from .efficient_layered import EfficientLayeredQwenImageAdapter
from .layered import QwenImageLayeredAdapter
from .peft_lora import QwenPeftLoraAdapter

ADAPTER_REGISTRY = {
    "base": BaseQwenImageAdapter,
    "edit": QwenImageEditAdapter,
    "layered": QwenImageLayeredAdapter,
    "peft_lora": QwenPeftLoraAdapter,
    "efficient_layered": EfficientLayeredQwenImageAdapter,
}


def parse_adapter(conf: dict) -> BaseQwenImageAdapter:
    adapter_type = conf.pop("type")
    if isinstance(adapter_type, list):
        # Construct mixin adapter
        adapter_class = type(
            "MixinAdapter", tuple(ADAPTER_REGISTRY[t] for t in adapter_type), {}
        )
    else:
        adapter_class = ADAPTER_REGISTRY.get(adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return adapter_class(**conf)
