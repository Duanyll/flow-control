from .base import Flux2Adapter

ADAPTER_REGISTRY = {
    "base": Flux2Adapter,
}


def parse_adapter(conf: dict) -> Flux2Adapter:
    adapter_type = conf["type"]
    adapter_class = ADAPTER_REGISTRY.get(adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return adapter_class(**conf)
