from .base import LongCatAdapter

ADAPTER_REGISTRY = {
    "base": LongCatAdapter,
}


def parse_adapter(conf: dict) -> LongCatAdapter:
    adapter_type = conf["type"]
    adapter_class = ADAPTER_REGISTRY.get(adapter_type)
    if adapter_class is None:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return adapter_class(**conf)
