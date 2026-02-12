from .base import Flux1Adapter
from .d_concat import Flux1DConcatAdapter
from .fill import Flux1FillAdapter
from .n_concat import Flux1NConcatAdapter

ADAPTER_REGISTRY = {
    "base": Flux1Adapter,
    "d_concat": Flux1DConcatAdapter,
    "n_concat": Flux1NConcatAdapter,
    "fill": Flux1FillAdapter,
}


def parse_adapter(conf: dict) -> Flux1Adapter:
    adapter_type = conf["type"]
    adapter_class = ADAPTER_REGISTRY.get(adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return adapter_class(**conf)
