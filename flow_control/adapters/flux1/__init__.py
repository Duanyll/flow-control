from .base import BaseFlux1Adapter
from .peft_lora import Flux1PeftLoraAdapter
from .d_concat import Flux1DConcatAdapter
from .n_concat import Flux1NConcatAdapter
from .fill import Flux1FillAdapter

ADAPTER_REGISTRY = {
    "base": BaseFlux1Adapter,
    "peft_lora": Flux1PeftLoraAdapter,
    "d_concat": Flux1DConcatAdapter,
    "n_concat": Flux1NConcatAdapter,
    "fill": Flux1FillAdapter,
}

def parse_adapter(conf: dict) -> BaseFlux1Adapter:
    adapter_type = conf.pop("type")
    adapter_class = ADAPTER_REGISTRY.get(adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return adapter_class(**conf)