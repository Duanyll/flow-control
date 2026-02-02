from .base import BaseLongCatAdapter

ADAPTER_REGISTRY = {
    "base": BaseLongCatAdapter,
}


def parse_adapter(conf: dict) -> BaseLongCatAdapter:
    adapter_type = conf["type"]
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
